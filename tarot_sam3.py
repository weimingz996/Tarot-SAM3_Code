import hashlib
import json
from pathlib import Path
from ref_prompt import description_variant_pipeline as ref_variants
from ref_prompt import target_contract_prompts as ref_contract
from ref_prompt import target_name_augmentation as ref_names
from src.models import (
    DINOv3Engine,
    QwenEngine,
    SAM3Engine,
    extract_best_mask,
    filter_masks_by_bboxes,
    mask_iou,
)
from src.prompts import reasoning_prompts
from src.utils import (
    ExperimentLogger,
    centroid_of_mask,
    combine_matrices,
    dilate_mask,
    extract_bbox_from_mask,
    extract_dino_points,
    find_negative_point,
    get_comparator_result,
    get_gate_score,
    largest_comp_area,
    mask_image,
    resize_image,
    sample_mask_points,
    save_full_long_short_bbox_board,
    save_mask,
    save_mask_and_point,
    save_mask_grid,
    save_point_on_image,
    save_simi_map,
    split_phases,
    visualize_dis_region,
)
from ref_prompt.full_description_pipeline import evaluate_one
from ref_prompt.target_contract_prompts import (
    reasoning_prompts as ref_prompts,
)
from ref_prompt.description_variant_pipeline import evaluate
from ref_prompt.target_name_augmentation import generate_target_name_candidates
from ref_prompt.training_free_bbox_selector import select_bbox_mask
from reason_prompt.reasonseg_best_flow import VisionQwen, run_reasonseg_candidate_flow
import os
import numpy as np
from typing import Tuple
import cv2
from PIL import Image, ImageOps

METHODS = ("full", "short", "long")


def configure_numeric_hyperparameters(cfg):
    reason = cfg.tarot_sam3.Reason
    eri = cfg.tarot_sam3.ERI

    ref_variants.THRESHOLD = eri.refer_variant_iou_threshold
    ref_variants.REFERENCE_EXACT_COPY_BLOCK_IOU = (
        eri.refer_reference_exact_copy_block_iou
    )

    ref_contract.MAX_REFERENCE_OBJECTS = reason.refer_max_reference_objects
    ref_names.TARGET_NAME_MAX_WORDS = reason.refer_target_name_max_words


def bbox_match_iou(mask, input_bbox) -> float:
    value = np.asarray(mask, dtype=bool)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    height, width = value.shape
    x1, y1, x2, y2 = (float(item) for item in input_bbox)
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    ys, xs = np.where(value)
    if len(xs) == 0:
        return 0.0
    mx1, my1 = float(xs.min()), float(ys.min())
    mx2, my2 = float(xs.max() + 1), float(ys.max() + 1)
    intersection = max(0.0, min(x2, mx2) - max(x1, mx1)) * max(
        0.0, min(y2, my2) - max(y1, my1)
    )
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    mask_area = (mx2 - mx1) * (my2 - my1)
    union = box_area + mask_area - intersection
    return intersection / union if union else 0.0


class TarotSAM3:
    reasonseg_candidate_runner = staticmethod(run_reasonseg_candidate_flow)

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        configure_numeric_hyperparameters(cfg)
        reason_cfg = cfg.tarot_sam3.Reason
        self.qwen = QwenEngine(
            qwen_cfg=cfg.qwen,
            timeout=120,
            max_tokens=256,
            temperature=0.0,
        )
        self.text_qwen = VisionQwen.from_qwen_config(cfg.qwen)
        self.dino = DINOv3Engine(dinov3_location=cfg.dinov3.dinov3_location,
                                 model_name=cfg.dinov3.model_name,
                                 device=device,
                                 weights=cfg.dinov3.weight_path)
        self.sam3 = SAM3Engine(ckpt_path=cfg.sam3.weight_path, device=device,
                               conf_thresh=cfg.tarot_sam3.Reason.refer_sam_confidence)
        self.logger = None
        self.ERI_mask = None
        self.sam3_points, self.sam3_labels = None, None
        self.description_mask = None
        self.dino_simi_map = None
        self.query = None
        self.target_scope = None
        self.full_description = None
        self.masks_info = None
        self.image_path = None
        self.image = None
        self.save_dir = None
        self.reason_seg = False
        self.sam3_point_mask = None
        self.visualize = False

    def _live_source(self):
        height, width = self.image.shape[:2]
        return {
            "sample_key": "refcoco_live_0",
            "index": 0,
            "image_id": -1,
            "image_size": [width, height],
            "query": self.query,
            "descriptions": {"initial": self.query, "guarded_v3": self.query},
            "bbox_response": {"initial": None, "guarded_v3": None},
            "bbox": {"initial": None, "guarded_v3": None},
            # The evaluators require this post-prediction field. It is never used
            # as generation or selection evidence.
            "gt_bbox": [0.0, 0.0, 0.0, 0.0],
        }

    def _run_full_v4_live(self):
        evidence_root = Path(self.save_dir)
        evidence_root.mkdir(parents=True, exist_ok=True)
        reason_cfg = self.cfg.tarot_sam3.Reason
        return evaluate_one(
            self.qwen,
            lambda: self.sam3,
            self.image,
            self._live_source(),
            ref_prompts,
            1,
            evidence_root,
            max_rounds=1,
            use_reference_masks=reason_cfg.refer_use_reference_masks,
            use_ordinal_masks=reason_cfg.refer_use_ordinal_masks,
            sam_confidence=float(reason_cfg.refer_sam_confidence),
        )

    def _full_v4_fallback(self, error):
        self.logger.log("Full V4", f"live generation unavailable: {error}")
        return {
            "identity_accepted": False,
            "target": {
                "target_name": "object",
                "target_scope": "whole_object",
                "selector_scope": None,
                "source": "fallback",
            },
            "descriptions": {"v4_full": self.query},
            "bbox": {"v4_full": None},
            "selection_route": None,
            "error": str(error),
        }

    def ref_reasoning_prompt(self):
        try:
            row = self._run_full_v4_live()
        except Exception as exc:
            row = self._full_v4_fallback(exc)

        full_description = row["descriptions"].get("v4_full") or self.query
        target = row.get("target") or {}
        target_name = str(target.get("target_name") or "object").strip()
        scope = target.get("selector_scope") or target.get("target_scope")
        self.target_entity = target_name
        self.full_description = full_description
        self.target_scope = scope
        self._full_v4_row = row
        self.logger.log("Reasoning", f"Full Description: {full_description}")
        self.logger.log("Reasoning", f"Target Scope: {self.target_scope}")
        self.logger.log("Reasoning", f"Row Information: {row}")






    def get_empty_mask(self):
        empty_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=bool)
        return empty_mask

    def description_mask_extractor(self, description, id, return_info: bool = False):
        text_results = self.sam3.predict_text(description)
        if len(text_results) == 0:
            self.logger.log("Reasoning", f"Fail to output mask for {id} Description: {description}")
            empty_mask = self.get_empty_mask()
            mask_info = {"id": id, "text": description, "mask": empty_mask,
                         "raw_masks_info": [],
                         "conf": 0.0, "source": "text"}
            return mask_info if return_info else mask_info["mask"]
        self.logger.log("Reasoning", f"Success to output mask for {id} Description: {description}")
        masks = [
            np.asarray(result["mask"]).astype(bool).squeeze()
            for result in text_results
        ]
        raw_masks_info = [
            {**result, "mask": mask, "conf": float(result.get("conf", 0.0))}
            for result, mask in zip(text_results, masks)
        ]
        fused_mask_info = None
        if self.target_scope in {"collection", "group", "group_set"}:
            sam3_text_mask = np.logical_or.reduce(masks)
            sam3_conf = max(float(result["conf"]) for result in text_results)
            fused_mask_info = {
                "mask": sam3_text_mask,
                "conf": sam3_conf,
                "is_fused": True,
            }
        else:
            best_result = max(text_results, key=lambda result: float(result["conf"]))
            sam3_text_mask = np.asarray(best_result["mask"]).astype(bool).squeeze()
            sam3_conf = float(best_result["conf"])
        mask_info = {"id": id, "text": description, "mask": sam3_text_mask,
                     "raw_masks_info": raw_masks_info,
                     "conf": sam3_conf, "source": "text"}
        if fused_mask_info is not None:
            mask_info["fused_mask_info"] = fused_mask_info
        return mask_info if return_info else sam3_text_mask

    def _run_full_sl_v2_live(self):
        row = self._full_v4_row
        full = {
            "description": row["descriptions"].get("v4_full") or self.query,
            "bbox": row["bbox"].get("v4_full"),
        }
        empty = {
            "full": full,
            "short": {"description": None, "bbox": None},
            "long": {"description": None, "bbox": None},
        }
        if full["bbox"] is None:
            return {**empty, "error": "Full V4 bbox unavailable"}
        try:
            return evaluate(
                self.qwen,
                lambda: self.sam3,
                self.image,
                float(self.cfg.tarot_sam3.Reason.refer_sam_confidence),
                row,
                sl_version="v2",
            )
        except Exception as exc:
            self.logger.log("Full-SL V2", f"live generation unavailable: {exc}")
            return {**empty, "error": str(exc)}


    def _generate_bbox_candidates(self, bboxes):
        shape = tuple(self.image.shape[:2])
        candidates = []
        for method in METHODS:
            bbox = bboxes.get(method)
            if bbox is None:
                continue
            bbox = list(bbox)
            predictions = self.sam3.predict_box(bbox)
            raw_masks_info = []
            for candidate_index, prediction in enumerate(predictions):
                raw_mask = np.asarray(prediction["mask"]).astype(bool).squeeze()
                if raw_mask.shape != shape:
                    raise ValueError(
                        f"{method} SAM3 mask shape {raw_mask.shape} "
                        f"!= image shape {shape}"
                    )
                match_iou = bbox_match_iou(raw_mask, bbox)
                if match_iou < self.cfg.tarot_sam3.ERI.refer_bbox_match_iou_thresh:
                    continue
                raw_masks_info.append({
                    **prediction,
                    "mask": raw_mask,
                    "conf": float(prediction.get("conf", 0.0)),
                    "candidate_index": candidate_index,
                    "bbox_match_iou": match_iou,
                })
            if not raw_masks_info:
                continue

            chosen = max(
                raw_masks_info,
                key=lambda raw: (
                    raw["bbox_match_iou"],
                    raw["conf"],
                    -raw["candidate_index"],
                ),
            )
            mask = chosen["mask"]
            candidates.append({
                "id": f"bbox_{method}",
                "method": method,
                "mask": mask,
                "raw_masks_info": raw_masks_info,
                "input_bbox": bbox,
                "bbox_match_iou": chosen["bbox_match_iou"],
                "conf": chosen["conf"],
                "stability_score": float(chosen.get("stability_score", 0.0)),
                "source": "bbox",
            })
        return candidates

    def _select_bbox_candidate(self, candidates, text_candidates):
        boxes = {
            candidate["method"]: candidate["mask"]
            for candidate in candidates
        }
        box_metadata = {
            candidate["method"]: {
                "sam_confidence": float(candidate.get("conf", 0.0)),
                "bbox_match_iou": float(
                    candidate.get("bbox_match_iou", 0.0)
                ),
            }
            for candidate in candidates
        }
        voters = [
            candidate
            for candidate in text_candidates
            if candidate.get("prompt_index", -1) >= 2
            and candidate.get("fused_mask_info") is None
        ]
        text_masks = [candidate["mask"] for candidate in voters]
        text_metadata = [
            {
                "prompt_index": candidate["prompt_index"],
                "confidence": float(candidate.get("conf", 0.0)),
                "mask_sha256": hashlib.sha256(
                    np.packbits(np.asarray(candidate["mask"], dtype=bool))
                    .tobytes()
                ).hexdigest(),
            }
            for candidate in voters
        ]
        selected, mask, _ = select_bbox_mask(
            boxes, text_masks, text_metadata, box_metadata
        )
        return mask, selected

    def _initial_text_candidates(self, bboxes, archive_context=None):
        object_names = generate_target_name_candidates(
            self.qwen,
            self.full_description,
            self.target_entity,
            self.target_scope,
        )
        prompt = reasoning_prompts["sam3_multi_expression"].format(
            Q=self.query,
            T=self.full_description,
            N=object_names,
        )
        expressions = split_phases(self.qwen.generate(prompt))
        self.logger.log("Reasoning", f"More Expressions: {expressions}")

        masks_info = [{
            **self.description_mask_extractor(
                self.full_description, "full_text", return_info=True
            ),
            "prompt_index": 1,
        }]
        descriptions = set(self.full_description)
        for index, description in enumerate([self.full_description] + expressions):
            id = f"more_text_{index}"
            for _ in range(3):
                mask_info = self.description_mask_extractor(
                        description, id, return_info=True
                    )
                if np.any(mask_info["mask"]):
                    descriptions.add(description)
                    masks_info.append({
                        **mask_info,
                        "prompt_index": index + 1,
                    })
                    break
                else:
                    descriptions.add(description)
                    new_prompt = reasoning_prompts["ref_refine"].format(Q=self.query,
                                                                        TS=descriptions,
                                                                        N=object_names)
                    description = self.qwen.generate(new_prompt)

        if archive_context is not None and not self.reason_seg and self.visualize:
            text_ids = []
            texts = []
            masks = []
            mask_offsets = [0]
            for mask_info in masks_info:
                if not np.any(mask_info["mask"]):
                    continue
                raw_masks = [
                    np.asarray(raw["mask"]).astype(bool).squeeze()
                    for raw in mask_info.get("raw_masks_info", [])
                ] or [np.asarray(mask_info["mask"]).astype(bool).squeeze()]
                text_ids.append(str(mask_info["id"]))
                texts.append(str(mask_info.get("text", "")))
                masks.extend(raw_masks)
                mask_offsets.append(len(masks))

            context_bboxes = archive_context["bboxes"]
            bbox_arrays = {
                f"{method}_bbox": (
                    np.asarray(context_bboxes[method], dtype=np.float32)
                    if context_bboxes.get(method) is not None
                    else np.full(4, np.nan, dtype=np.float32)
                )
                for method in ("full", "short", "long")
            }
            bbox_candidates = archive_context.get("bbox_candidates") or []
            bbox_voter_masks = [
                np.asarray(candidate["mask"]).astype(bool).squeeze()
                for candidate in bbox_candidates
            ]
            np.savez_compressed(
                os.path.join(self.save_dir, "ref_inputs_before_filter.npz"),
                full_description=np.asarray(self.full_description),
                shorten_query=np.asarray(
                    archive_context.get("shorten_query") or ""
                ),
                longer_query=np.asarray(
                    archive_context.get("longer_query") or ""
                ),
                object_names=np.asarray(object_names, dtype=str),
                text_ids=np.asarray(text_ids, dtype=str),
                texts=np.asarray(texts, dtype=str),
                mask_offsets=np.asarray(mask_offsets, dtype=np.int64),
                masks=(
                    np.stack(masks)
                    if masks
                    else np.empty((0, *self.image.shape[:2]), dtype=bool)
                ),
                bbox_voter_methods=np.asarray(
                    [candidate["method"] for candidate in bbox_candidates],
                    dtype=str,
                ),
                bbox_voter_masks=(
                    np.stack(bbox_voter_masks)
                    if bbox_voter_masks
                    else np.empty((0, *self.image.shape[:2]), dtype=bool)
                ),
                bbox_voter_bbox_match_iou=np.asarray(
                    [candidate["bbox_match_iou"] for candidate in bbox_candidates],
                    dtype=np.float32,
                ),
                bbox_voter_confidence=np.asarray(
                    [candidate["conf"] for candidate in bbox_candidates],
                    dtype=np.float32,
                ),
                bbox_voter_stability=np.asarray(
                    [candidate["stability_score"] for candidate in bbox_candidates],
                    dtype=np.float32,
                ),
                **bbox_arrays,
            )

        candidates = []
        for mask_info in masks_info:
            branch = mask_info["id"]
            raw_masks_info = mask_info.get("raw_masks_info")
            if raw_masks_info is None:
                candidates.append({**mask_info, "branch": branch})
                continue
            for raw_index, raw in enumerate(raw_masks_info):
                raw_mask = np.asarray(raw["mask"]).astype(bool).squeeze()
                if not np.any(raw_mask):
                    continue
                candidates.append({
                    **mask_info,
                    **raw,
                    "id": f"{branch}_raw_{raw_index}",
                    "branch": branch,
                    "mask": raw_mask,
                    "conf": float(raw.get("conf", 0.0)),
                    "source": "text",
                    "raw_masks_info": [],
                    "fused_mask_info": None,
                })
            fused = mask_info.get("fused_mask_info")
            if self.target_scope in {"collection", "group", "group_set"} and fused is not None:
                union_mask = np.asarray(fused["mask"]).astype(bool).squeeze()
                if np.any(union_mask):
                    candidates.append({
                        **mask_info,
                        "id": f"{branch}_union",
                        "branch": branch,
                        "mask": union_mask,
                        "conf": float(fused.get("conf", mask_info.get("conf", 0.0))),
                        "source": "text",
                    })

        return filter_masks_by_bboxes(
            candidates,
            bboxes,
            bbox_iou_thresh=(
                self.cfg.tarot_sam3.ERI.refer_filter_bbox_iou_thresh
            ),
        )




    def expression_reasoning_interpreter(self):
        if self.reason_seg:
            raise RuntimeError(
                "reason_seg=True must use the sealed V2 pipeline in process_image"
        )
        if not self.reason_seg:
            sl_result = self._run_full_sl_v2_live()
            shorten_query = sl_result["short"].get("description")
            longer_query = sl_result["long"].get("description")
            bboxes = {
                method: sl_result[method].get("bbox")
                for method in METHODS
            }
            self.logger.log("Reasoning", f"Short Description: {shorten_query}")
            self.logger.log("Reasoning", f"Long Description: {longer_query}")
            for method in ("full", "long", "short"):
                self.logger.log(
                    f"BBox Detector for {method.title()}",
                    f"Output Box: {bboxes.get(method)}",
                )
            if self.visualize:
                save_full_long_short_bbox_board(
                    self.image,
                    bboxes,
                    os.path.join(
                        self.save_dir, "full_long_short_bboxes.png"
                    ),
                )
            bbox_candidates = self._generate_bbox_candidates(bboxes)
            masks_info = self._initial_text_candidates(
                [],
                archive_context={
                    "shorten_query": shorten_query,
                    "longer_query": longer_query,
                    "bboxes": bboxes,
                    "bbox_candidates": bbox_candidates,
                } if self.visualize else None,
            )
            filter_bboxes = []
            best_bbox_candidate = None
            if bbox_candidates:
                bbox_mask, selected_method = self._select_bbox_candidate(
                    bbox_candidates, masks_info
                )
                self.logger.log(
                    "Best BBox", f"Source: {selected_method.title()}"
                )
                selected_candidate = next(
                    candidate for candidate in bbox_candidates
                    if candidate["method"] == selected_method
                )
                best_bbox_candidate = {
                    **selected_candidate,
                    "id": "Best BBox",
                    "branch": "Best BBox",
                }
                self.full_des_bbox = np.asarray(
                    extract_bbox_from_mask(bbox_mask), dtype=int
                )
                filter_bboxes.append(self.full_des_bbox)
                masks_info = filter_masks_by_bboxes(
                    masks_info,
                    filter_bboxes,
                    bbox_iou_thresh=(
                        self.cfg.tarot_sam3.ERI.refer_filter_bbox_iou_thresh
                    ),
                )
            if best_bbox_candidate is not None:
                masks_info.append(best_bbox_candidate)

            self.masks_info = masks_info
            return masks_info

    def ERI_point_extractor(self, component_mask: np.ndarray, iter_str: str):
        eri_cfg = self.cfg.tarot_sam3.ERI
        prefix = "reason_" if self.reason_seg else "refer_"
        mask_points = sample_mask_points(
            component_mask,
            getattr(eri_cfg, f"{prefix}sample_point_num"),
            getattr(eri_cfg, f"{prefix}sample_point_strategy"),
        )
        if self.visualize:
            save_point_on_image(self.image, mask_points,
                                os.path.join(self.save_dir, f"sample_points_{iter_str}.png"))
        dino_sim_maps = [self.dino.extract_cos_similarity(point) for point in mask_points]
        self.dino_simi_map = combine_matrices(
            dino_sim_maps,
            method="harmonic_mean",
        )
        if self.visualize:
            save_simi_map(self.dino_simi_map,
                          os.path.join(self.save_dir, f"dino_simi_map_{iter_str}.png"))
        if self.reason_seg:
            dino_simi_points = extract_dino_points(self.dino_simi_map)
        else:
            dino_simi_points = extract_dino_points(self.dino_simi_map, self.full_des_bbox)
        self.logger.log("ERI Point Extractor", f"DINO Positive Point: {dino_simi_points}")
        negative_point = find_negative_point(self.dino_simi_map, dino_simi_points[0],
                                             search_radius=1000,
                                             threshold=0.3,
                                             diff_threshold=0)
        if negative_point is not None:
            self.logger.log("ERI", f"DINO Negative Point: {list(negative_point)}")
            points = np.concatenate([dino_simi_points, negative_point[None, :]], axis=0)
            labels = np.array([1] * len(dino_simi_points) + [0])
        else:
            self.logger.log("ERI", "DINO Negative Point: None")
            points = dino_simi_points
            labels = np.ones(len(dino_simi_points))
        self.sam3_points = points
        self.sam3_labels = labels
        sam3_masks = self.sam3.predict_point(points, labels)
        self.sam3_point_mask = extract_best_mask(sam3_masks)
        if self.visualize:
            save_mask_and_point(self.image, self.sam3_point_mask,
                                self.sam3_points,
                                os.path.join(self.save_dir, f"sam3_point_mask_{iter_str}.png"),
                                input_labels=self.sam3_labels)

    def output_checker(self, mask: np.ndarray, iter_str: str) -> bool:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file_1:
            mask_image(self.image, mask, temp_file_1.name)
            if self.reason_seg:
                checker_prompt = reasoning_prompts["output_checker_reason"].format(
                    Q=self.query,
                    T_final=self.full_description,
                )
            else:
                checker_prompt = reasoning_prompts["output_checker"].format(T_final=self.query)
            check_result = self.qwen.generate(checker_prompt, [temp_file_1.name])
        self.logger.log(f"Object Checker for {iter_str}", f"Check Result: '{check_result}'")
        return get_gate_score(check_result) == 1

    def mask_comparator(self, mask_1: np.ndarray, mask_2: np.ndarray, iter_str: str) -> Tuple[np.ndarray, int]:
        if not np.any(mask_1):
            self.logger.log(f"Mask Comparator for {iter_str}", "Winner: B")
            return mask_2, 1
        if not np.any(mask_2):
            self.logger.log(f"Mask Comparator for {iter_str}", "Winner: A")
            return mask_1, 0
        if np.all(mask_1 == mask_2):
            self.logger.log(f"Mask Comparator for {iter_str}", "Winner: A")
            return mask_1, 0
        if mask_iou(mask_1, mask_2) > 0.8:
            if mask_1.sum() < mask_2.sum():
                self.logger.log(f"Mask Comparator for {iter_str}", "Winner: A")
                return mask_1, 0
            else:
                self.logger.log(f"Mask Comparator for {iter_str}", "Winner: B")
                return mask_2, 1
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file_1, tempfile.NamedTemporaryFile(
                suffix=".png", delete=True) as temp_file_2:
            mask_image(self.image, mask_1, temp_file_1.name)
            mask_image(self.image, mask_2, temp_file_2.name)
            if self.reason_seg:
                resize_image(temp_file_1.name, temp_file_1.name)
                resize_image(temp_file_2.name, temp_file_2.name)
            compare_prompt = reasoning_prompts["mask_comparator"].format(Q=self.full_description)
            response = self.qwen.generate(compare_prompt, [temp_file_1.name,
                                                           temp_file_2.name])
        winner = "A" if get_comparator_result(response) == "a" else "B"
        self.logger.log(f"Mask Comparator for {iter_str}",
                        f"Winner: {winner} Comparator Response: {response}")
        if winner == "A":
            return mask_1, 0
        return mask_2, 1


    def extract_point_under(self, cur_mask: np.ndarray, point_mask: np.ndarray, point_simi_map: np.ndarray,
                            iter_str: str):
        point_dis_region = ~cur_mask & point_mask
        if not point_dis_region.any():
            return False, None, None
        point_simi_thresh = point_simi_map[point_dis_region].min()
        part_mask = point_dis_region | ((point_simi_map > point_simi_thresh) & cur_mask)
        if self.visualize:
            save_path = os.path.join(self.save_dir, f"under_part_mask_{iter_str}.png")
            save_mask(part_mask, save_path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file_1:
            mask_image(self.image, part_mask, temp_file_1.name)
            if self.reason_seg:
                checker_prompt = reasoning_prompts["reason_part_checker"].format(Q=self.query, T=self.full_description)
            else:
                checker_prompt = reasoning_prompts["part_checker"].format(Q=self.full_description)
            check_result = self.qwen.generate(checker_prompt, [temp_file_1.name])
        self.logger.log(f"Under Checker for Point Mask of {iter_str}", f"Check Result: '{check_result}'")
        if float(check_result) == 1.0:
            new_point = centroid_of_mask(part_mask)[np.newaxis, :]
            # new_mask_results = self.sam3.predict_point(new_point, np.array([1]))
            # new_mask = extract_best_mask(new_mask_results)
            # new_point = centroid_of_mask(new_mask)[np.newaxis, :]
            return True, new_point, np.array([1])
        else:
            return False, None, None

    def extract_point_over(self, cur_mask: np.ndarray, point_mask: np.ndarray, cur_simi_map: np.ndarray, iter_str: str):
        cur_dis_region = cur_mask & ~point_mask
        prefix = "reason_" if self.reason_seg else "refer_"
        simi_thresh = max(
            cur_simi_map[cur_dis_region].min(),
            getattr(self.cfg.tarot_sam3.MSR, f"{prefix}simi_thresh"),
        )
        part_mask = (~(cur_mask & point_mask)) & (cur_simi_map > simi_thresh)
        if self.visualize:
            save_path = os.path.join(self.save_dir, f"over_part_mask_{iter_str}.png")
            save_mask(part_mask, save_path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file_1:
            mask_image(self.image, part_mask, temp_file_1.name)
            if self.reason_seg:
                checker_prompt = reasoning_prompts["reason_part_checker"].format(
                    Q=self.query, T=self.full_description
                )
            else:
                checker_prompt = reasoning_prompts["part_checker"].format(Q=self.full_description)
            check_result = self.qwen.generate(checker_prompt, [temp_file_1.name])
        check_result = 1 - float(check_result)
        self.logger.log(f"Over Checker for Cur Mask of {iter_str}", f"Check Result: '{check_result}'")
        if check_result == 1.0:
            new_point = centroid_of_mask(part_mask)[np.newaxis, :]
            # new_mask_results = self.sam3.predict_point(centroid, np.array([1]))
            # new_mask = extract_best_mask(new_mask_results)
            # new_point = centroid_of_mask(new_mask)[np.newaxis, :]
            return True, new_point, np.array([0])
        else:
            return False, None, None

    def mask_self_refine(self, cur_mask: np.ndarray, iter_str: str) -> np.ndarray:
        refer_mask = self.sam3_point_mask
        inter_mask = refer_mask & cur_mask
        cur_dis_region = cur_mask & ~inter_mask
        point_dis_region = refer_mask & ~inter_mask

        if not inter_mask.any():
            return self.get_empty_mask()
        try:
            cur_dis_point = centroid_of_mask(cur_dis_region)
            sam3_dis_point = centroid_of_mask(point_dis_region)
        except:
            return cur_mask
        cur_dis_simi_map = self.dino.extract_cos_similarity(cur_dis_point)
        sam3_dis_simi_map = self.dino.extract_cos_similarity(sam3_dis_point)
        if self.visualize:
            visualize_dis_region(self.image,
                                 cur_dis_region, point_dis_region,
                                 [cur_dis_simi_map, sam3_dis_simi_map],
                                 os.path.join(self.save_dir, f"dis_region_{iter_str}.png"))
        if not self.reason_seg and not (
            largest_comp_area(cur_dis_region)
            > self.cfg.tarot_sam3.MSR.refer_comp_area_thresh
            or largest_comp_area(point_dis_region)
            > self.cfg.tarot_sam3.MSR.refer_comp_area_thresh
        ):
            return cur_mask
        if_under, new_point, new_label = self.extract_point_under(cur_mask, refer_mask, sam3_dis_simi_map, iter_str)
        if if_under:
            self.sam3_points = np.concatenate((self.sam3_points, new_point))
            self.sam3_labels = np.concatenate((self.sam3_labels, new_label))
            self.logger.log(f"{iter_str}", f"Extracted New Positive Point: {new_point}")
            masks_info = self.sam3.predict_point(self.sam3_points, self.sam3_labels)
            new_mask = extract_best_mask(masks_info)
            if self.visualize:
                save_mask_and_point(self.image, new_mask, self.sam3_points,
                                    os.path.join(self.save_dir, f"refined_mask_iter_{iter_str}.png"),
                                    input_labels=self.sam3_labels)
            return new_mask
        if_over, new_point, new_label = self.extract_point_over(cur_mask, refer_mask, cur_dis_simi_map, iter_str)
        if if_over:
            self.sam3_points = np.concatenate((self.sam3_points, new_point))
            self.sam3_labels = np.concatenate((self.sam3_labels, new_label))
            self.logger.log(f"{iter_str}", f"Extracted New Negative Point: {new_point}")
            masks_info = self.sam3.predict_point(self.sam3_points, self.sam3_labels)
            new_mask = extract_best_mask(masks_info)
            if self.visualize:
                save_mask_and_point(self.image, new_mask, self.sam3_points,
                                    os.path.join(self.save_dir, f"refined_mask_iter_{iter_str}.png"),
                                    input_labels=self.sam3_labels)
            return new_mask
        else:
            return cur_mask

    def process_image(self, image_path: str, query: str, reason_seg: bool,
                      logger: ExperimentLogger, save_dir: str, visualize: bool = True):
        self.logger = logger
        self.reason_seg = reason_seg
        mode = "reason" if self.reason_seg else "refer"
        reason_cfg = self.cfg.tarot_sam3.Reason

        self.qwen.temperature = 0.0
        if not self.reason_seg:
            self.qwen.max_tokens = 256
        self.sam3.conf_thresh = getattr(reason_cfg, f"{mode}_sam_confidence")
        self.image_path = image_path
        if self.reason_seg:
            with Image.open(self.image_path) as image:
                image = np.array(ImageOps.exif_transpose(image).convert("RGB"))
            self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            self.image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if self.image is None:
            raise FileNotFoundError(f"Cannot read image: {self.image_path}")
        self.query = query
        self.save_dir = save_dir
        self.visualize = visualize
        self.target_scope = None

        if self.reason_seg:
            result = getattr(
                self, "reasonseg_candidate_runner", run_reasonseg_candidate_flow
            )(
                full_des_qwen=self.qwen,
                text_qwen=self.text_qwen,
                sam3=self.sam3,
                image_path=self.image_path,
                query=self.query,
                output_dir=os.path.join(self.save_dir, "reasonseg_best_flow"),
                sample=Path(self.image_path).stem,
                visualize=self.visualize,
            )
            full_des = result["full_des"]
            self.full_description = str(full_des["text"])
            self.target_scope = str(full_des["scope"])
            self.masks_info = result.get("candidate_masks_info") or []
            for mask_info in self.masks_info:
                self.logger.log(
                    "Reason Candidate Mask",
                    f"id={mask_info.get('id')}, pipeline={mask_info.get('pipeline')}, "
                    f"role={mask_info.get('role')}, text={mask_info.get('text')}, "
                    f"mask_pixels={int(np.asarray(mask_info['mask']).sum())}",
                )
            return self.masks_info
        else:
            self.sam3.load_image(self.image_path)
            self.qwen.load_image(self.image_path)
            self.target_entity = None
            self.full_des_bbox = None
            self.masks_info = None
            self.ref_reasoning_prompt()
            self.masks_info = self.expression_reasoning_interpreter()
            if not self.masks_info:
                self.logger.log("Main Process", "No candidate masks were generated.")
            return self.masks_info


if __name__ == "__main__":
    from src.utils import ExperimentLogger
    from src.utils import load_config
    import torch
    import shutil
    import argparse

    parser = argparse.ArgumentParser(description="Tarot-SAM3 单图推理")
    parser.add_argument("--config", type=str, default="configs/7B.yaml", help="配置文件路径（YAML）")
    parser.add_argument("--image_path", type=str, default="test-images/12548840825_70c715e3e3_o.jpg",
                        help="输入图像路径")
    parser.add_argument("--query", type=str,
                        default="In cold weather, dogs may need extra protection to keep them warm. What object in the picture can a dog wear to provide warmth during snowy walks?",
                        help="指代/描述文本")
    parser.add_argument("--save_dir", type=str, default=None, help="结果保存目录，默认与图像名一致")
    parser.add_argument("--reason_seg", action="store_true", help="是否使用 ReasonSeg 推理模式")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tarot_sam3 = TarotSAM3(cfg, device)

    image_path = args.image_path
    query = args.query
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.splitext(os.path.basename(image_path))[0]
    reason_seg = args.reason_seg
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    logger = ExperimentLogger(log_dir=save_dir, fname=save_dir, resume=False)

    output_mask = tarot_sam3.process_image(image_path, query, reason_seg, logger, save_dir)
    output_mask_path = os.path.join(save_dir, "final_output_mask.png")
    save_mask(output_mask, output_mask_path)
    logger.log("MAIN", f"Final Output Mask saved at: '{output_mask_path}'")
