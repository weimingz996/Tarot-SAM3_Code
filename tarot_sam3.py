import hashlib
from pathlib import Path
from ref_prompt import description_variant_pipeline as ref_variants
from ref_prompt import target_contract_prompts as ref_contract
from ref_prompt import target_name_augmentation as ref_names
from src.models import (
    QwenEngine,
    SAM3Engine,
    filter_masks_by_bboxes,
)
from src.prompts import reasoning_prompts
from src.utils import (
    ExperimentLogger,
    extract_bbox_from_mask,
    save_full_long_short_bbox_board,
    split_phases,
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


def filter_candidates_by_bboxes(masks_info, bboxes, bbox_iou_thresh):
    filtered = filter_masks_by_bboxes(
        masks_info, bboxes, bbox_iou_thresh=bbox_iou_thresh
    )
    return [
        candidate
        for candidate in masks_info
        if candidate.get("mask") is None
        or not np.any(candidate["mask"])
        or any(candidate is kept for kept in filtered)
    ]


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
        self.sam3 = SAM3Engine(ckpt_path=cfg.sam3.weight_path, device=device,
                               conf_thresh=cfg.tarot_sam3.Reason.refer_sam_confidence)
        self.logger = None
        self.query = None
        self.target_scope = None
        self.full_description = None
        self.masks_info = None
        self.image_path = None
        self.image = None
        self.save_dir = None
        self.reason_seg = False
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

    def ref_reasoning_prompt(self):
        row = self._run_full_v4_live()

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
        return evaluate(
            self.qwen,
            lambda: self.sam3,
            self.image,
            float(self.cfg.tarot_sam3.Reason.refer_sam_confidence),
            row,
            sl_version="v2",
        )


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
        text_masks = []
        text_metadata = []
        for candidate in text_candidates:
            if candidate.get("prompt_index", -1) < 2:
                continue
            raw_masks_info = candidate.get("raw_masks_info")
            voters = [candidate] if raw_masks_info is None else raw_masks_info
            for voter in voters:
                mask = np.asarray(voter.get("mask"), dtype=bool)
                if not np.any(mask):
                    continue
                text_masks.append(mask)
                text_metadata.append({
                    "prompt_index": candidate["prompt_index"],
                    "confidence": float(voter.get("conf", 0.0)),
                    "mask_sha256": hashlib.sha256(
                        np.packbits(mask).tobytes()
                    ).hexdigest(),
                })
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
        descriptions = {self.full_description}
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

        return filter_candidates_by_bboxes(
            masks_info,
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
                masks_info = filter_candidates_by_bboxes(
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

    def process_image(self, image_path: str, query: str, reason_seg: bool,
                      logger: ExperimentLogger, save_dir: str,
                      visualize: bool = True) -> list[dict]:
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
    from src.utils import load_config, save_candidate_masks
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

    candidate_masks = tarot_sam3.process_image(
        image_path, query, reason_seg, logger, save_dir
    )
    saved_paths = save_candidate_masks(candidate_masks, save_dir, logger)
    logger.log("MAIN", f"Saved {len(saved_paths)} candidate masks.")
