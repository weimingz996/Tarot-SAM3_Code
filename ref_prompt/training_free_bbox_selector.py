"""Frozen training-free bbox-mask selector for the online RefCOCO path."""

import numpy as np


METHODS = ("full", "short", "long")
BITS = np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def _overlap(left, right):
    left = np.asarray(left, np.uint8)
    right = np.asarray(right, np.uint8)
    intersection = np.array([
        [BITS[a & b].sum() for b in right]
        for a in left
    ], dtype=float)
    union = (
        BITS[left].sum(axis=1)[:, None]
        + BITS[right].sum(axis=1)[None, :]
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def _evidence(boxes, packed_text, text_metadata, box_metadata):
    valid = np.array([method in boxes for method in METHODS])
    assert valid.any()
    anchor = int(valid.argmax())
    shape = next(iter(boxes.values())).shape
    boxbits = np.array([
        np.packbits(boxes[method])
        if method in boxes
        else np.zeros((np.prod(shape) + 7) // 8, np.uint8)
        for method in METHODS
    ])
    pairs = _overlap(boxbits, boxbits)
    indices = [
        index
        for index, metadata in enumerate(text_metadata)
        if metadata["prompt_index"] >= 2
    ]
    seen = set()
    unique = []
    for index in indices:
        key = (
            text_metadata[index]["prompt_index"],
            text_metadata[index]["mask_sha256"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(index)
    text = np.asarray(packed_text, np.uint8)[unique]
    confidence = np.array([
        text_metadata[index]["confidence"] for index in unique
    ], dtype=float)
    prompts = np.array([
        text_metadata[index]["prompt_index"] for index in unique
    ], dtype=int)
    matrix = _overlap(text, boxbits) if len(text) else np.zeros((0, 3))
    same = _overlap(text, text) if len(text) else np.zeros((0, 0))
    box_confidence = np.array([
        box_metadata[method]["sam_confidence"] if method in boxes else 0.0
        for method in METHODS
    ])
    return {
        "valid": valid,
        "anchor": anchor,
        "full_iou": pairs[anchor],
        "confidence": box_confidence,
        "iou": matrix,
        "text_iou": same,
        "text_confidence": confidence,
        "prompt_ids": prompts,
    }


def _select_evidence(evidence):
    anchor = evidence["anchor"]
    valid = evidence["valid"]
    if not len(evidence["prompt_ids"]):
        return METHODS[anchor], {
            "votes": [0.0, 0.0, 0.0],
            "scores": [0.0, 0.0, 0.0],
            "reason": "no_augmentation_masks",
        }
    iou = evidence["iou"].reshape(-1, 3)
    prompts = evidence["prompt_ids"]
    confidence = evidence["text_confidence"]
    prompt_names = sorted(set(prompts))
    count = np.array([np.sum(prompts == name) for name in prompts])
    support = np.array([
        np.mean([
            evidence["text_iou"][index, prompts == name].max()
            for name in prompt_names
            if name != prompts[index]
        ])
        if len(prompt_names) > 1 else 1.0
        for index in range(len(iou))
    ])
    weight = (
        confidence ** 2
        * iou.max(1) ** 2
        * (0.1 + support)
        * (0.1 + iou[:, anchor])
        / count
    )
    vote = (iou * weight[:, None]).sum(0) / max(1e-12, weight.sum())
    box_confidence = evidence["confidence"]
    eligible = valid & (vote >= vote[anchor])
    if box_confidence[anchor] >= 0.85:
        eligible &= evidence["full_iou"] >= 0.25
    eligible[anchor] = True
    score = np.where(eligible, vote + 0.5 * box_confidence, -np.inf)
    method = METHODS[int(score.argmax())]
    return method, {
        "votes": vote.tolist(),
        "scores": [
            float(value) if np.isfinite(value) else None for value in score
        ],
        "text_weights": weight.tolist(),
        "eligible": eligible.tolist(),
    }


def select_bbox_mask(boxes, text_masks, text_metadata, box_metadata):
    """Return one original bbox mask using augmentation-only text votes."""
    assert len(text_masks) == len(text_metadata)
    assert all(
        np.asarray(mask).shape == next(iter(boxes.values())).shape
        for mask in text_masks
    )
    packed = np.array([
        np.packbits(np.asarray(mask, bool)) for mask in text_masks
    ], dtype=np.uint8)
    method, details = _select_evidence(
        _evidence(boxes, packed, text_metadata, box_metadata)
    )
    return method, boxes[method], details
