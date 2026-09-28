"""Evaluate bounded target-lock Full General V4 on the frozen 500x3 split.

Production V4 is a conditional pipeline, never a dataset-name router:
query target contract -> optional candidate ID/query plan/SAM3 evidence -> lossless
full description -> deterministic contract repair -> optional ordinal SAM3 -> bbox.
Initial and V3 are frozen metric baselines only; neither can become a V4 output.

Ground truth is read only after all generation, routing, and fallback decisions.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np


from ref_prompt import target_contract_prompts as v4_module
from src.models import bbox_iou, nms_masks
from src.utils import save_labeled_mask_overlay


METHODS = ("initial", "v3", "v4_full")
SCHEMA_VERSION = "v4_query_contract_3"
ORDINAL_NMS_IOU = 0.60
ORDINAL_BBOX_GUARD_IOU = 0.25




def dataset_of(sample_key: str) -> str:
    if sample_key.startswith("refcoco+_"):
        return "refcoco+"
    if sample_key.startswith("refcocog_"):
        return "refcocog"
    return "refcoco"


def split_label(sample_key: str) -> str:
    return sample_key.rsplit("_", 1)[0]


def is_plural_target_name(name: str) -> bool:
    return v4_module.canonical_name(name) != v4_module._norm(name)


def parse_bbox_response(response, width: int, height: int):
    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(response or "")
    )
    if len(numbers) < 4:
        return None
    x1, y1, x2, y2 = map(float, numbers[:4])
    box = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    return box if box[2] > box[0] and box[3] > box[1] else None




def call_qwen(qwen, stage: str, prompt: str, retries: int, trace: dict, image_paths=None):
    trace["logical_qwen_calls"] += 1
    last = None
    for attempt in range(retries + 1):
        trace["physical_qwen_calls"] += 1
        try:
            response = qwen.generate(prompt, image_paths=list(image_paths or []))
            trace["qwen_stages"].append({"stage": stage, "retries_used": attempt})
            return response
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(1.0)
    raise last


def _mask_bbox(mask):
    mask = np.asarray(mask).squeeze().astype(bool)
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _mask_center(mask):
    mask = np.asarray(mask).squeeze().astype(bool)
    ys, xs = np.where(mask)
    return [float(xs.mean()), float(ys.mean())] if len(xs) else None


def _strong_masks(results, confidence: float):
    valid = []
    for item in results or []:
        mask = np.asarray(item.get("mask")).squeeze().astype(bool)
        conf = float(item.get("conf", 0.0))
        if mask.ndim == 2 and mask.any() and conf >= confidence:
            valid.append({"mask": mask, "conf": conf, "source": item.get("source", "text")})
    return nms_masks(valid, ORDINAL_NMS_IOU) if valid else []


def _sam_predict(sam3, phrase: str, cache: dict, confidence: float, trace: dict):
    key = " ".join(phrase.lower().split())
    if key not in cache:
        trace["sam_text_calls"] += 1
        cache[key] = _strong_masks(sam3.predict_text(phrase), confidence)
    return cache[key]


def build_reference_evidence(sam3, image, references: list, cache: dict,
                             confidence: float, trace: dict, work: Path):
    paths, metadata = [], []
    for index, reference in enumerate(references, start=1):
        masks = _sam_predict(sam3, reference["sam_prompt"], cache, confidence, trace)
        if not masks:
            metadata.append({**reference, "status": "no_strong_mask", "mask_count": 0})
            continue
        kept = sorted(masks, key=lambda item: item["conf"], reverse=True)[:5]
        union = np.logical_or.reduce([item["mask"] for item in kept])
        path = work / f"reference_{index}.jpg"
        save_labeled_mask_overlay(
            image,
            union,
            f"REFERENCE ONLY R{index}: {reference['sam_prompt']} | NEVER TARGET",
            path,
            (0, 210, 70),
        )
        paths.append(str(path))
        metadata.append({
            **reference,
            "status": "shown",
            "mask_count": len(kept),
            "confidences": [round(float(item["conf"]), 5) for item in kept],
            "bboxes": [_mask_bbox(item["mask"]) for item in kept],
        })
    return paths, metadata


def _complete_ordinal_masks(masks):
    """Drop small part masks whose center is contained by a fuller instance."""
    items = []
    for item in masks:
        box, center = _mask_bbox(item["mask"]), _mask_center(item["mask"])
        if box and center:
            items.append({**item, "bbox": box, "center": center,
                          "box_area": (box[2] - box[0]) * (box[3] - box[1])})
    kept = []
    for item in items:
        fragment = any(
            other is not item
            and item["box_area"] < 0.20 * other["box_area"]
            and other["bbox"][0] <= item["center"][0] <= other["bbox"][2]
            and other["bbox"][1] <= item["center"][1] <= other["bbox"][3]
            and float(item["conf"]) <= float(other["conf"])
            for other in items
        )
        if not fragment:
            kept.append(item)
    return kept


def _dedupe_boundary_ordinal_masks(items, axis_index: int):
    """Collapse overlapping SAM fragments that share the ranked image edge."""
    if len(items) < 2:
        return items
    shape = np.asarray(items[0]["mask"]).squeeze().shape
    axis_size = shape[1] if axis_index == 0 else shape[0]
    perpendicular = 1 - axis_index
    groups = []
    for item in items:
        box = item["bbox"]
        low, high = box[axis_index], box[axis_index + 2]
        edge = "low" if low <= 1.0 else "high" if high >= axis_size - 1.0 else None
        placed = False
        for group in groups:
            other = group[0]
            other_box = other["bbox"]
            other_low, other_high = other_box[axis_index], other_box[axis_index + 2]
            other_edge = (
                "low" if other_low <= 1.0
                else "high" if other_high >= axis_size - 1.0
                else None
            )
            a0, a1 = box[perpendicular], box[perpendicular + 2]
            b0, b1 = other_box[perpendicular], other_box[perpendicular + 2]
            overlap = max(0.0, min(a1, b1) - max(a0, b0))
            min_span = max(1.0, min(a1 - a0, b1 - b0))
            mask = np.asarray(item["mask"]).squeeze().astype(bool)
            other_mask = np.asarray(other["mask"]).squeeze().astype(bool)
            pixel_overlap = np.logical_and(mask, other_mask).sum() / max(
                1, min(mask.sum(), other_mask.sum())
            )
            if (
                edge is not None
                and edge == other_edge
                and abs(item["center"][axis_index] - other["center"][axis_index])
                    <= 0.04 * axis_size
                and overlap / min_span >= 0.25
                and pixel_overlap >= 0.50
            ):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    return [
        max(group, key=lambda item: (item["box_area"], float(item["conf"])))
        for group in groups
    ]


def _ambiguous_ordinal_rank_masks(items, axis_index: int):
    """Abstain when duplicate/edge masks could change an ordinal rank."""
    if len(items) < 2:
        return False
    shape = np.asarray(items[0]["mask"]).squeeze().shape
    axis_size = shape[1] if axis_index == 0 else shape[0]
    for index, item in enumerate(items):
        box = item["bbox"]
        edge = "low" if box[axis_index] <= 1.0 else (
            "high" if box[axis_index + 2] >= axis_size - 1.0 else None
        )
        if edge is None:
            continue
        for other in items[index + 1:]:
            other_box = other["bbox"]
            other_edge = "low" if other_box[axis_index] <= 1.0 else (
                "high" if other_box[axis_index + 2] >= axis_size - 1.0 else None
            )
            if (
                edge == other_edge
                and abs(item["center"][axis_index] - other["center"][axis_index])
                    <= 0.08 * axis_size
            ):
                return True
            mask = np.asarray(item["mask"]).squeeze().astype(bool)
            other_mask = np.asarray(other["mask"]).squeeze().astype(bool)
            pixel_overlap = np.logical_and(mask, other_mask).sum() / max(
                1, min(mask.sum(), other_mask.sum())
            )
            if (
                abs(item["center"][axis_index] - other["center"][axis_index])
                    <= 0.04 * axis_size
                and pixel_overlap >= 0.50
            ):
                return True
    return False


def _filter_explicit_ordinal_row(items, ordinal: dict, row_position: str):
    """For an explicit upper/lower row, rank only within that visual row."""
    rank = int(ordinal["rank"])
    if row_position not in {"upper", "lower"} or ordinal["axis"] != "x" or len(items) < 2 * rank:
        return items
    ordered = sorted(items, key=lambda item: item["center"][1])
    gaps = [
        ordered[index + 1]["center"][1] - ordered[index]["center"][1]
        for index in range(len(ordered) - 1)
    ]
    if not gaps:
        return items
    split = max(range(len(gaps)), key=gaps.__getitem__) + 1
    height = np.asarray(items[0]["mask"]).squeeze().shape[0]
    if gaps[split - 1] < 0.08 * height:
        return items
    selected = ordered[:split] if row_position == "upper" else ordered[split:]
    return selected if len(selected) >= rank else items


def build_ordinal_evidence(sam3, image, target_name: str, ordinal: dict,
                           cache: dict, confidence: float, trace: dict, work: Path,
                           row_position: str = ""):
    masks = _complete_ordinal_masks(
        _sam_predict(sam3, target_name, cache, confidence, trace)
    )
    rank = int(ordinal["rank"])
    axis_index = 0 if ordinal["axis"] == "x" else 1
    if rank > 1 and _ambiguous_ordinal_rank_masks(masks, axis_index):
        return {
            "status": "ambiguous_rank_masks",
            "strong_mask_count": len(masks),
            "items": [],
        }
    if rank == 1:
        masks = _dedupe_boundary_ordinal_masks(masks, axis_index)
    masks = _filter_explicit_ordinal_row(masks, ordinal, row_position)
    if len(masks) < rank:
        return {"status": "insufficient_candidates", "strong_mask_count": len(masks), "items": []}
    ranked = []
    for item in masks:
        ranked.append(item)
    ranked.sort(key=lambda item: item["center"][axis_index], reverse=bool(ordinal["descending"]))
    if len(ranked) < rank:
        return {"status": "insufficient_candidates", "strong_mask_count": len(ranked), "items": []}
    desired = rank - 1
    start = max(0, desired - 1)
    stop = min(len(ranked), start + 3)
    start = max(0, stop - 3)
    neighborhood = []
    for ranked_index in range(start, stop):
        item = dict(ranked[ranked_index])
        item["sorted_rank"] = ranked_index + 1
        neighborhood.append(item)
    display = []
    for position, item in enumerate(neighborhood):
        candidate_id = chr(ord("A") + position)
        path = work / f"ordinal_{candidate_id}.jpg"
        save_labeled_mask_overlay(
            image,
            item["mask"],
            f"TARGET CANDIDATE {candidate_id} | RANK {item['sorted_rank']} FROM {ordinal['direction'].upper()}",
            path,
            (235, 45, 45),
        )
        display.append({
            "candidate_id": candidate_id,
            "path": str(path),
            "mask": item["mask"],
            "bbox": item["bbox"],
            "center": item["center"],
            "confidence": float(item["conf"]),
            "sorted_rank": item["sorted_rank"],
        })
    return {"status": "shown", "strong_mask_count": len(ranked), "items": display}


def _public_ordinal_evidence(evidence: dict):
    return {
        "status": evidence.get("status"),
        "strong_mask_count": evidence.get("strong_mask_count", 0),
        "items": [
            {key: item[key] for key in ("candidate_id", "bbox", "center", "confidence", "sorted_rank")}
            for item in evidence.get("items", [])
        ],
    }


def _ensure_sam3(get_sam3, state: dict):
    if "engine" not in state:
        state["engine"] = get_sam3()
    return state["engine"]


def _deterministic_single_reference_plan(target_name: str, candidates: list):
    """Select one unambiguous directed query-span reference; otherwise abstain."""
    if len(candidates) != 1:
        return {"reference_objects": [], "parse_error": None}
    candidate = candidates[0]
    relation = v4_module._norm(
        v4_module.normalize_literal_spelling(candidate.get("relation", ""))
    )
    allowed_relations = {
        "to the left of", "left of", "to the right of", "right of",
        "in front of", "next to", "close to", "beside", "behind", "under",
        "below", "above", "over", "near", "on", "holding", "holds",
        "held by", "carrying", "carried by", "used by", "being held by",
        "being carried by", "being used by", "riding", "touching", "looking at",
        "approaching", "following", "followed by", "grasping", "blocked by",
        "covered by", "occluded by",
    }
    phrase = v4_module._space(candidate.get("query_phrase", ""))
    words = v4_module._words(phrase)
    forbidden = {
        "it", "there", "one", "ones", "image", "row", "side", "edge",
        "foreground", "background", "his", "her", "their", "its", "and", "or",
    }
    spatial_only = {
        "left", "right", "top", "bottom", "front", "back", "middle", "center",
        "far", "very", "farthest", "furthest",
    }
    head = v4_module._clean_target_name(phrase)
    if (
        relation not in allowed_relations
        or not words
        or bool(set(words) & forbidden)
        or set(words) <= spatial_only | {"a", "an", "the"}
        or not v4_module.valid_locked_target_name(head)
        or v4_module.names_equivalent(head, target_name)
    ):
        return {"reference_objects": [], "parse_error": None}
    return v4_module.parse_query_plan_response(
        json.dumps({"reference_ids": [candidate["id"]]}),
        target_name,
        candidates,
    )


def resolve_identity(qwen, get_sam3, image, source: dict, prompts: dict, retries: int,
                     max_rounds: int, use_reference_masks: bool, confidence: float,
                     trace: dict, work: Path, static_plan: dict, sam_state: dict,
                     sam_cache: dict):
    del max_rounds
    query = source["query"]
    target_raw = call_qwen(
        qwen, "target_contract", prompts["v4_target_name"].format(Q=query),
        retries, trace,
    )
    target, target_error = v4_module.parse_target_response(
        target_raw,
        query,
        static_plan.get("target_hint"),
        static_plan.get("scope_hint"),
        static_plan.get("allow_visual_anchor", False),
        static_plan.get("headless_query", False),
        static_plan.get("anatomy_anchor", False),
    )
    target_retry_raw = None
    needs_category_retry = (
        target["target_name"] == "object" and static_plan.get("headless_query")
    )
    needs_anchor_retry = bool(
        static_plan.get("allow_visual_anchor") and not target.get("visual_anchor")
    )
    if needs_category_retry or needs_anchor_retry:
        retry_instruction = (
            "The query has no reliable physical noun. Return one broad visible "
            "target category from the image; never return a query adjective as target_name."
            if needs_category_retry else
            "Keep target_name=person. Return exactly one anchor of at most three words: "
            "either one absolute image position, or one visible relation such as "
            "'holding hotdog'. Do not use a person, body part, clothing, pronoun, and/or."
            if static_plan.get("anatomy_anchor") else
            "Return the same locked person category plus exactly one short anchor. "
            "For a body-part selector, prefer its visible object relation such as "
            "'holding hotdog'; for sitting/seated/lying/riding, use a physical support "
            "such as 'on couch'; otherwise use left/right/top/bottom/foreground/background. "
            "Never rename the target and never use clothing alone as the anchor."
        )
        target_retry_raw = call_qwen(
            qwen,
            "target_contract_retry",
            prompts["v4_target_name"].format(Q=query)
            + "\n" + retry_instruction,
            retries,
            trace,
        )
        retry_target, _ = v4_module.parse_target_response(
            target_retry_raw,
            query,
            static_plan.get("target_hint"),
            static_plan.get("scope_hint"),
            static_plan.get("allow_visual_anchor", False),
            static_plan.get("headless_query", False),
            static_plan.get("anatomy_anchor", False),
        )
        if needs_category_retry and retry_target["target_name"] != "object":
            target = retry_target
        elif needs_anchor_retry and retry_target.get("visual_anchor"):
            same_human_category = bool(
                v4_module.canonical_name(target["target_name"]) in v4_module.HUMAN_WORDS
                and v4_module.canonical_name(retry_target["target_name"])
                in v4_module.HUMAN_WORDS
            )
            if (
                same_human_category
                or static_plan.get("implicit_person_owner")
                or v4_module.names_equivalent(
                    target["target_name"], retry_target["target_name"]
                )
            ):
                target["visual_anchor"] = retry_target["visual_anchor"]
    if (
        target["source"] == "visual_inference"
        and v4_module.canonical_name(target["target_name"]) in v4_module.HUMAN_WORDS
    ):
        target["target_name"] = "person"
    scope_label = static_plan.get("scope_label", "")
    if static_plan.get("scope_hint") in {"part", "region"} and scope_label:
        parent_hint = static_plan.get("scope_parent_hint", "")
        if parent_hint and v4_module.valid_locked_target_name(parent_hint):
            target["target_name"] = (
                f"{parent_hint} half" if scope_label == "half" else parent_hint
            )
        target["selector_scope"] = static_plan["scope_hint"]
        target["selector_phrase"] = static_plan.get("target_role_span") or scope_label
        target["target_scope"] = "whole_object"
    role_name = static_plan.get("target_role_name", "")
    directed_role_lock = any(
        v4_module._norm(relation) != "with"
        for relation in static_plan.get("relation_triggers", [])
    )
    if (
        static_plan.get("target_role_locked")
        and static_plan.get("scope_hint") is None
        and role_name
        and v4_module.valid_locked_target_name(role_name)
        and not all(word.isdigit() for word in v4_module._words(role_name))
        and len(v4_module._words(role_name)) <= 4
        and (
            target["target_name"] == "object"
            or v4_module.names_equivalent(role_name, target["target_name"])
            or directed_role_lock
        )
    ):
        target.update({
            "target_name": role_name,
            "target_scope": "whole_object",
            "source": "explicit_query",
            "evidence_span": static_plan["target_role_span"],
            "target_span": static_plan["target_role_span"],
            "role": "query_role_lock",
            "confidence": "high",
        })
    static_hint = static_plan.get("target_hint") or {}
    if (
        static_plan.get("scope_hint") == "group"
        and static_plan.get("scope_parent_hint")
    ):
        target.update({
            "target_name": static_plan["scope_parent_hint"],
            "target_scope": "group",
            "source": "explicit_query",
            "evidence_span": static_plan["target_role_span"],
            "target_span": static_plan["target_role_span"],
            "role": "query_group_lock",
            "confidence": "high",
        })
    elif (
        static_plan.get("scope_hint") is None
        and static_hint.get("reason") in {"explicit_human_head", "comparative_head"}
        and static_hint.get("target_name")
    ):
        hint_span = static_hint.get("target_span") or static_plan["target_role_span"]
        target.update({
            "target_name": static_hint["target_name"],
            "target_scope": "whole_object",
            "source": "explicit_query",
            "evidence_span": hint_span,
            "target_span": hint_span,
            "role": f"query_{static_hint['reason']}_lock",
            "confidence": "high",
        })
    if static_plan.get("implicit_person_owner"):
        owner_proxy = static_plan.get("owner_proxy_span", "")
        target.update({
            "target_name": "person",
            "target_scope": "whole_object",
            "source": "visual_inference",
            "evidence_span": "",
            "target_span": "",
            "owner_proxy_span": owner_proxy,
            "role": "implicit_owner_lock",
            "confidence": "high",
        })
    candidate_raw, candidate_error = None, None
    if target["confidence"] == "low" and len(target["candidates"]) >= 2:
        candidate_raw = call_qwen(
            qwen,
            "target_candidate_id",
            prompts["v4_target_candidate"].format(
                Q=query,
                candidates=json.dumps(target["candidates"], ensure_ascii=False),
            ),
            retries,
            trace,
        )
        candidate, candidate_error = v4_module.parse_target_candidate_response(
            candidate_raw, target["candidates"]
        )
        if candidate:
            target = v4_module.apply_target_candidate(target, candidate)
    reference_candidates = static_plan.get("reference_candidates", [])
    directed_references = [
        item for item in reference_candidates
        if " ".join(item.get("relation", "").lower().split()) != "with"
    ]
    if len(directed_references) == 1:
        phrase = directed_references[0]["query_phrase"]
        current = target.get("reference_span", "")
        if not current or v4_module._contains_token_sequence(current, phrase):
            target["reference_span"] = phrase
    if static_plan.get("support_reference") and not target.get("reference_span"):
        target["reference_span"] = static_plan["support_reference"]

    record = {
        "iteration": 1,
        "target_raw": target_raw,
        "target_retry_raw": target_retry_raw,
        "target": target,
        "target_error": target_error,
        "candidate_raw": candidate_raw,
        "candidate_error": candidate_error,
        "query_plan_raw": None,
        "query_plan": {"reference_objects": []},
        "required_literal_cues": [],
        "reference_evidence": [],
        "description_raw": None,
        "description": None,
        "violations": [],
        "contract_repaired": False,
        "post_repair_violations": [],
        "judge_raw": None,
        "judge": None,
        "judge_error": "deterministic_contract",
        "accepted": False,
    }

    query_plan = {"reference_objects": []}
    if use_reference_masks and len(reference_candidates) == 1:
        deterministic_plan = _deterministic_single_reference_plan(
            target["target_name"], reference_candidates
        )
        if deterministic_plan["reference_objects"]:
            query_plan = deterministic_plan
            record["query_plan_raw"] = "deterministic_single_reference"
            record["query_plan"] = query_plan
    if use_reference_masks and reference_candidates and not query_plan["reference_objects"]:
        plan_raw = call_qwen(
            qwen,
            "query_plan",
            prompts["v4_query_plan"].format(
                Q=query,
                target_name=target["target_name"],
                reference_candidates=json.dumps(
                    reference_candidates, ensure_ascii=False, separators=(",", ":")
                ),
            ),
            retries,
            trace,
        )
        query_plan = v4_module.parse_query_plan_response(
            plan_raw, target["target_name"], reference_candidates
        )
        record["query_plan_raw"] = plan_raw
        record["query_plan"] = query_plan

    reference_paths, reference_meta = [], []
    if query_plan["reference_objects"]:
        sam3 = _ensure_sam3(get_sam3, sam_state)
        reference_paths, reference_meta = build_reference_evidence(
            sam3, image, query_plan["reference_objects"], sam_cache,
            confidence, trace, work,
        )
    record["reference_evidence"] = reference_meta
    shown_references = [item for item in reference_meta if item.get("status") == "shown"]
    evidence_note = (
        "; ".join(
            f"Image {index + 2} is green REFERENCE ONLY evidence for {item['sam_prompt']}"
            for index, item in enumerate(shown_references)
        ) or "none"
    )
    required_cues = v4_module.required_literal_cues(
        static_plan, target["target_name"], query, target
    )
    record["required_literal_cues"] = required_cues
    contract = {
        key: target.get(key)
        for key in (
            "target_name", "target_scope", "target_span", "reference_span",
            "visual_anchor", "owner_proxy_span", "role", "source",
        )
    }
    description_raw = call_qwen(
        qwen,
        "full_description",
        prompts["v4_full_description"].format(
            Q=query,
            target_name=target["target_name"],
            target_contract=json.dumps(contract, ensure_ascii=False),
            literal_cues=json.dumps(required_cues, ensure_ascii=False),
            references=json.dumps(query_plan["reference_objects"], ensure_ascii=False),
            evidence_note=evidence_note,
        ),
        retries,
        trace,
        reference_paths,
    ).strip()
    description = v4_module.normalize_description_prefix(
        v4_module.normalize_literal_spelling(description_raw), target["target_name"]
    )
    description = v4_module.ensure_scope_cue(
        description, target["target_name"], static_plan
    )
    description = v4_module.normalize_support_relation(description, static_plan)
    description = v4_module.normalize_side_location(
        query, target, description, static_plan
    )
    description = v4_module.sanitize_unsupported_ordinals(query, description)
    bare_plural_query = bool(
        len(v4_module._words(query)) == 1
        and is_plural_target_name(target["target_name"])
        and v4_module.names_equivalent(query, target["target_name"])
    )
    forced_raw_query = bool(
        static_plan.get("preserve_raw_query", False) or bare_plural_query
    )
    violations = (
        ["bare_plural_query_guard" if bare_plural_query else "raw_query_guard"]
        if forced_raw_query
        else v4_module.description_violations(query, target, description, static_plan)
    )
    if violations:
        description = v4_module.lossless_fallback_description(query, target)
        description = v4_module.normalize_literal_spelling(description)
        description = v4_module.normalize_support_relation(description, static_plan)
    post_repair = v4_module.description_violations(query, target, description, static_plan)
    accepted = not post_repair
    record.update({
        "description_raw": description_raw,
        "description": description,
        "violations": violations,
        "contract_repaired": bool(violations),
        "post_repair_violations": post_repair,
        "accepted": accepted,
    })
    return {
        "accepted": accepted,
        "target": target,
        "description": description,
        "reference_paths": reference_paths,
        "reference_evidence": reference_meta,
        "rounds": [record],
    }


def detect_final_bbox(qwen, prompts, query: str, target: dict, description: str,
                      width: int, height: int, retries: int, trace: dict,
                      force_query_locator: bool = False,
                      prefer_query_locator: bool = False):
    contract = {
        key: target.get(key)
        for key in (
            "target_name", "target_scope", "source", "target_span",
            "selector_scope", "selector_phrase", "reference_span", "visual_anchor",
            "owner_proxy_span", "role",
        )
    }
    scope_instruction = {
        "region": "Box only the requested target region.",
        "group": "Use one box enclosing the complete referenced group, not one member.",
    }.get(target.get("target_scope"), "")
    if target.get("selector_scope") == "part":
        scope_instruction = (
            "Use the requested part only to identify the instance; box the complete "
            "visible locked parent target."
        )
    elif target.get("owner_proxy_span"):
        scope_instruction = (
            "Box the complete visible extent of the SAME owner identified by the "
            "clothing/body/accessory selector. If that owner is partly out of frame and "
            "only the selector is visible, keep that partial owner; never switch to a "
            "different fully visible person."
        )
    elif (
        target.get("target_scope") == "whole_object"
        and re.match(r"^\s*(?:left|right)\s+side\s+of\b", query, re.I)
    ):
        scope_instruction = (
            f"Box the complete target {target['target_name']}, not only its side "
            "or a subregion."
        )
    query_words = v4_module._words(query)
    target_word_count = len(v4_module._words(target.get("target_name", "")))
    target_is_literal = any(
        v4_module.names_equivalent(
            target.get("target_name", ""),
            " ".join(query_words[index:index + target_word_count]),
        )
        for index in range(max(0, len(query_words) - target_word_count + 1))
    ) if target_word_count else False
    target_is_plural = is_plural_target_name(target.get("target_name", ""))
    preserve_short_query = bool(
        len(query_words) <= 3
        and target_is_literal
        and not target_is_plural
        and target.get("source") == "explicit_query"
        and target.get("target_scope") == "whole_object"
        and not target.get("owner_proxy_span")
        and not target.get("visual_anchor")
    )
    locator_expression = query if (
        force_query_locator or prefer_query_locator or preserve_short_query
    ) else description
    owner_selector_locator = bool(
        force_query_locator
        and target.get("owner_proxy_span")
        and not target.get("visual_anchor")
    )
    if owner_selector_locator:
        locator_expression = description
    clock_match = re.fullmatch(
        r"\s*(?P<head>.+?)\s+at\s+(?P<hour>[1-9]|1[0-2])\s*",
        query,
        re.I,
    )
    if preserve_short_query and clock_match:
        locator_expression = (
            f"{clock_match.group('head')} at the {clock_match.group('hour')} o'clock position"
        )
    trace["locator_source"] = (
        "locked_owner_full_description" if owner_selector_locator
        else "lossless_contract_query" if force_query_locator
        else "lossless_parse_query" if prefer_query_locator
        else "normalized_clock_query" if preserve_short_query and clock_match
        else "lossless_short_query" if preserve_short_query
        else "verified_full_description"
    )
    trace["locator_expression"] = locator_expression
    clean_explicit = bool(
        target.get("source") == "explicit_query"
        and target.get("confidence") == "high"
        and target.get("target_scope") == "whole_object"
        and target.get("target_span")
        and v4_module._exact_query_span(query, target.get("target_span"))
        and not target.get("selector_scope")
        and not target.get("owner_proxy_span")
        and not target.get("visual_anchor")
        and not target.get("candidates")
        and target.get("role") != "query_explicit_human_head_lock"
    )
    prompt_key = (
        "v4_final_bbox" if clean_explicit and not force_query_locator
        else "v4_contract_bbox"
    )
    prompt = prompts[prompt_key].format(
        Q=query,
        target_contract=json.dumps(contract, ensure_ascii=False),
        description=locator_expression,
        scope_instruction=scope_instruction,
        W=width,
        H=height,
    )
    if not scope_instruction:
        prompt = prompt.replace("bounding box. \nFormat:", "bounding box.\nFormat:")
    response = call_qwen(qwen, "final_bbox", prompt, retries, trace)
    bbox = parse_bbox_response(response, width, height)
    if bbox is None:
        response = call_qwen(
            qwen,
            "final_bbox_format_retry",
            prompt + "\nReturn exactly four numeric coordinates; do not return None or prose.",
            retries,
            trace,
        )
        bbox = parse_bbox_response(response, width, height)
    return response, bbox


def side_location_rule(query: str, static_plan: dict):
    if not static_plan.get("side_location"):
        return None
    match = re.match(r"^\s*(left|right)\s+side\s+of\b", query, re.I)
    if not match:
        return None
    direction = match.group(1).lower()
    return {
        "rank": 1,
        "direction": direction,
        "axis": "x",
        "descending": direction == "right",
        "evidence_span": match.group(0).strip(),
    }


def detect_ordinal_bbox(qwen, prompts, query: str, target: dict, description: str,
                        ordinal: dict, evidence: dict, width: int, height: int,
                        retries: int, trace: dict):
    prompt = prompts["v4_ordinal_bbox"].format(
        Q=query,
        target_name=target["target_name"],
        description=description,
        ordinal=json.dumps(ordinal, ensure_ascii=False),
        candidate_rank_map=json.dumps(
            {
                item["candidate_id"]: item["sorted_rank"]
                for item in evidence["items"]
            },
            ensure_ascii=False,
        ),
        candidate_bbox_map=json.dumps(
            {
                item["candidate_id"]: [round(value, 2) for value in item["bbox"]]
                for item in evidence["items"]
            },
            ensure_ascii=False,
        ),
        W=width,
        H=height,
    )
    paths = [item["path"] for item in evidence["items"]]
    response = call_qwen(qwen, "ordinal_select_bbox", prompt, retries, trace, paths)
    candidate_id, box, error = v4_module.parse_ordinal_response(response, width, height)
    selected = next((item for item in evidence["items"] if item["candidate_id"] == candidate_id), None)
    if selected is None:
        return response, None, "ordinal_invalid_choice", candidate_id
    if box is None:
        return response, selected["bbox"], "ordinal_sam_bbox_guard", candidate_id
    if bbox_iou(box, selected["bbox"]) < ORDINAL_BBOX_GUARD_IOU:
        return response, selected["bbox"], "ordinal_sam_bbox_guard", candidate_id
    return response, box, "ordinal_qwen_bbox", candidate_id


def evaluate_one(qwen, get_sam3, image, source: dict, prompts: dict, retries: int,
                 evidence_root: Path, max_rounds: int = 2,
                 use_reference_masks: bool = True, use_ordinal_masks: bool = True,
                 sam_confidence: float = 0.6):
    started = time.time()
    query = source["query"]
    width, height = [int(value) for value in source["image_size"]]
    initial_description = source["descriptions"]["initial"]
    initial_bbox = source["bbox"]["initial"]
    initial_response = source["bbox_response"]["initial"]
    v3_description = source["descriptions"]["guarded_v3"]
    v3_bbox = source["bbox"]["guarded_v3"]
    v3_response = source["bbox_response"]["guarded_v3"]
    trace = {
        "logical_qwen_calls": 0,
        "physical_qwen_calls": 0,
        "sam_text_calls": 0,
        "qwen_stages": [],
    }
    static_plan = v4_module.static_query_plan(query)
    sam_state, sam_cache = {}, {}

    with tempfile.TemporaryDirectory(prefix="evidence_", dir=str(evidence_root)) as tmp:
        work = Path(tmp)
        identity = resolve_identity(
            qwen,
            get_sam3,
            image,
            source,
            prompts,
            retries,
            max_rounds,
            use_reference_masks,
            sam_confidence,
            trace,
            work,
            static_plan,
            sam_state,
            sam_cache,
        )
        ordinal_public = {"status": "not_triggered", "strong_mask_count": 0, "items": []}
        final_response, final_bbox = None, None
        final_description = identity.get("description")
        selection_route, fallback_reason = "v4_no_prediction", "identity_not_verified"
        ordinal_choice = None
        localized = False

        if identity["accepted"]:
            target, description = identity["target"], identity["description"]
            contract_violations = identity["rounds"][0].get("violations", [])
            hard_repair = any(
                violation != "bare_plural_query_guard"
                for violation in contract_violations
            )
            force_query_locator = bool(
                (
                    hard_repair
                    and not target.get("visual_anchor")
                    and "target_spelling_normalized" not in target.get("parse_warnings", [])
                )
                or (target.get("owner_proxy_span") and not target.get("visual_anchor"))
            )
            prefer_query_locator = bool(
                not force_query_locator
                and target.get("source") == "explicit_query"
                and target.get("target_scope") == "whole_object"
                and target.get("target_span")
                and v4_module._exact_query_span(query, target.get("target_span"))
                and not target.get("owner_proxy_span")
                and not target.get("visual_anchor")
                and not target.get("selector_scope")
                and not static_plan.get("target_hint")
                and not static_plan.get("relation_triggers")
                and not static_plan.get("reference_candidates")
                and not static_plan.get("ordinal")
                and not static_plan.get("actions")
                and not static_plan.get("hard_cues")
                and bool(static_plan.get("color_phrases"))
            )
            ordinal = static_plan.get("ordinal") if use_ordinal_masks else None
            side_rule = (
                side_location_rule(query, static_plan) if use_ordinal_masks else None
            )
            if side_rule:
                sam3 = _ensure_sam3(get_sam3, sam_state)
                side_evidence = build_ordinal_evidence(
                    sam3,
                    image,
                    target["target_name"],
                    side_rule,
                    sam_cache,
                    sam_confidence,
                    trace,
                    work,
                )
                ordinal_public = _public_ordinal_evidence(side_evidence)
                selected = next(
                    (
                        item for item in side_evidence.get("items", [])
                        if item["sorted_rank"] == 1
                    ),
                    None,
                )
                if selected is not None:
                    final_bbox = selected["bbox"]
                    final_response = json.dumps({
                        "candidate_id": selected["candidate_id"],
                        "bbox": [round(value, 2) for value in final_bbox],
                    })
                    final_description = description
                    ordinal_choice = selected["candidate_id"]
                    selection_route, fallback_reason = "spatial_sam_bbox", None
                    trace["locator_source"] = "spatial_sam_mask"
                    trace["locator_expression"] = query
                    localized = True
            if not localized and ordinal:
                sam3 = _ensure_sam3(get_sam3, sam_state)
                ordinal_evidence = build_ordinal_evidence(
                    sam3,
                    image,
                    target["target_name"],
                    ordinal,
                    sam_cache,
                    sam_confidence,
                    trace,
                    work,
                    row_position=(
                        "lower" if re.search(r"\b(?:lower|bottom)\s+row\b", query, re.I)
                        else "upper" if re.search(r"\b(?:upper|top)\s+row\b", query, re.I)
                        else ""
                    ),
                )
                ordinal_public = _public_ordinal_evidence(ordinal_evidence)
                if ordinal_evidence["status"] == "shown":
                    response, box, route, ordinal_choice = detect_ordinal_bbox(
                        qwen,
                        prompts,
                        query,
                        target,
                        query if force_query_locator else description,
                        ordinal,
                        ordinal_evidence,
                        width,
                        height,
                        retries,
                        trace,
                    )
                    if box is not None:
                        final_response, final_bbox = response, box
                        final_description = description
                        selection_route, fallback_reason = route, None
                        localized = True
                if not localized:
                    response, box = detect_final_bbox(
                        qwen, prompts, query, target, description,
                        width, height, retries, trace, force_query_locator,
                        True,
                    )
                    if box is not None:
                        final_response, final_bbox, final_description = response, box, description
                        selection_route, fallback_reason = "v4_ordinal_fallback_aligned", None
                        localized = True
                    else:
                        fallback_reason = "ordinal_and_aligned_bbox_invalid"
            elif not localized:
                response, box = detect_final_bbox(
                    qwen, prompts, query, target, description,
                    width, height, retries, trace, force_query_locator,
                    prefer_query_locator,
                )
                if box is not None:
                    final_response, final_bbox, final_description = response, box, description
                    repaired = identity["rounds"][0].get("contract_repaired")
                    selection_route = "v4_contract_repaired_bbox" if repaired else "v4_generated_bbox"
                    fallback_reason = "description_contract_repaired" if repaired else None
                    localized = True
                else:
                    fallback_reason = "bbox_invalid"
        elif identity.get("target"):
            # Query-preserving V4 fallback: rejected model text never reaches bbox.
            target = identity["target"]
            description = v4_module.lossless_fallback_description(query, target)
            response, box = detect_final_bbox(
                qwen, prompts, query, target, description,
                width, height, retries, trace, True,
            )
            final_response, final_bbox = response, box
            final_description = description
            if box is not None:
                selection_route = "v4_query_contract_fallback_bbox"
                fallback_reason = "generated_description_rejected"
            else:
                fallback_reason = "query_contract_bbox_invalid"

    # GT is deliberately touched only after every generation and selection above.
    gt_bbox = source["gt_bbox"]
    boxes = {"initial": initial_bbox, "v3": v3_bbox, "v4_full": final_bbox}
    iou = {
        method: bbox_iou(boxes[method], gt_bbox) if boxes[method] else 0.0
        for method in METHODS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_key": source["sample_key"],
        "dataset": dataset_of(source["sample_key"]),
        "split_label": split_label(source["sample_key"]),
        "index": source["index"],
        "image_id": source["image_id"],
        "image_size": source["image_size"],
        "query": query,
        "gt_bbox": gt_bbox,
        "static_query_plan": static_plan,
        "identity_accepted": identity["accepted"],
        "identity_rounds": identity["rounds"],
        "target": identity.get("target"),
        "reference_evidence": identity.get("reference_evidence", []),
        "ordinal_evidence": ordinal_public,
        "ordinal_choice": ordinal_choice,
        "selection_route": selection_route,
        "fallback_used": selection_route in {
            "v4_contract_repaired_bbox", "v4_query_contract_fallback_bbox",
            "v4_ordinal_fallback_aligned", "v4_no_prediction",
        },
        "fallback_reason": fallback_reason,
        "descriptions": {
            "initial": initial_description,
            "v3": v3_description,
            "v4_candidate": identity.get("description"),
            "v4_full": final_description,
        },
        "bbox_response": {"initial": initial_response, "v3": v3_response, "v4_full": final_response},
        "bbox": boxes,
        "iou": iou,
        "calls": trace,
        "elapsed_seconds": round(time.time() - started, 3),
        "error": None,
    }
