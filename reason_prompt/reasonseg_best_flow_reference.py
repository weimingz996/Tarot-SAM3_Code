#!/usr/bin/env python3
"""Per-image ReasonSeg pipeline through pre-vote mask candidates.

The inference path is GT-free.  Text augmentation always makes exactly two
independent vision-language calls; SAM prompts are deduplicated per image.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.parallel import run_parallel_calls


STAGE_ORDER = (
    "full_des", "text_augmentation", "bbox_augmentation", "candidate_export"
)
IDENTITY_ROLES = (
    "canonical_name", "detector_synonym", "number_group",
    "query_noun", "full_des_evidence",
)
RELATION_ROLES = (
    "visual_discriminator", "extent_relation", "contrast_view",
    "owner_relation", "concise_alias",
)
FULL_DES_ROLES = (
    "full_des_canonical", "full_des_visual", "full_des_extent",
)
QWEN_TEMPERATURE = 0.0
QWEN_MAX_TOKENS = 900
QWEN_TIMEOUT_SECONDS = 120
QWEN_MAX_RETRIES = 0
IMAGE_MAX_SIDE = 2048


IDENTITY_INSTRUCTIONS = """Independently resolve the direct visible target of
one ReasonSeg query from the QUERY and image. You are intentionally not given
the FullDes target. Do not guess from
dataset habits. Before writing prompts, compare exactly three plausible
concrete target nouns. A target is exact only when segmenting that visible
entity alone answers the query. Reject an owner, container, content, support,
attribute value, other participant, and any noun that is broader or narrower
than the requested extent. Prefer the common detector-friendly object or part
name visible in the image over a functional paraphrase. For a functional or
causal query, explicitly compare the whole object with the visible part that
performs the function. For a hypothetical use, choose the visible object that
supplies the requested scene or affordance, not an object mentioned only
inside the imagined scene. If the query asks for a region, area, portion, or
patch, target that extent rather than its whole owner. Preserve requested
number and extent.

Use the five roles as follows: canonical_name is the plain target head;
detector_synonym is a common visual synonym; number_group explicitly names the
target with the requested count/group meaning; query_noun is the shortest noun
phrase that directly fills the query answer slot; full_des_evidence combines
the target with the strongest visible FullDes discriminator. Except for a true
detector synonym, every text must explicitly name target_head.

After generating all detector prompts, localize the same query-required target
with one tight full-image box. Use absolute pixel coordinates on the supplied
IMAGE_WIDTH by IMAGE_HEIGHT image in left, top, right, bottom order, not
normalized coordinates. Replace the four zeros in the box schema with measured
integer coordinates. Output four separate JSON numbers, never a quoted string.
Enclose all requested instances for a group, but only the
requested part or region for a part/region query. The box is evidence for a
later vote and must not change any detector phrase.

Return exactly this bare JSON shape, filling every string and all three checks:
{"candidate_checks":[
{"target_head":"candidate noun","verdict":"exact|too_broad|too_narrow|wrong_participant","reason":"brief visual or query evidence"},
{"target_head":"candidate noun","verdict":"exact|too_broad|too_narrow|wrong_participant","reason":"brief visual or query evidence"},
{"target_head":"candidate noun","verdict":"exact|too_broad|too_narrow|wrong_participant","reason":"brief visual or query evidence"}],
"target_contract":{"target_head":"best exact direct target noun"},"prompts":[
{"role":"canonical_name","target_head":"same direct target","text":"short noun phrase"},
{"role":"detector_synonym","target_head":"same direct target","text":"short noun phrase"},
{"role":"number_group","target_head":"same direct target","text":"short noun phrase"},
{"role":"query_noun","target_head":"same direct target","text":"short noun phrase"},
{"role":"full_des_evidence","target_head":"same direct target","text":"short noun phrase"}],
"bbox_xyxy_pixels":[0,0,0,0]}
Do not rename keys, nest prompts elsewhere, add prose, or use Markdown."""


RELATION_INSTRUCTIONS = """Independently audit the direct visible target of
one ReasonSeg query, then generate five target-preserving detector views.
Derive exactly three plausible concrete target nouns from the QUERY and image,
then compare them with FULLDES_TARGET_CONTRACT. You do not see the first
model's answer. The QUERY is authoritative. FullDes is a proposal, not a
label. Correct a proposal that is functional, generic, too broad, too narrow,
or names another participant. Prefer the common detector-friendly object or
part name visible in the image. Never substitute an owner, container, content,
support, attribute value, or another participant. For a functional or causal
query, compare the whole object with the visible part that performs the
function. For a hypothetical use, identify the visible object supplying the
requested scene or affordance. If the query asks for a region, area, portion,
or patch, keep that extent in the target noun.

Use the five roles as follows: visual_discriminator adds appearance;
extent_relation preserves whole/part/material/group extent;
contrast_view distinguishes the target from visible distractors;
owner_relation keeps only a query-required spatial or ownership relation; and
concise_alias is a short alternate detector phrase. Every text must explicitly
name target_head and must be a noun phrase rather than a sentence. A visual
discriminator must add an actually observed color, material, shape, or spatial
position; for comparative or superlative queries, do not merely repeat words
such as largest or widest. In every full_des_prompts item, copy
FULLDES_TARGET_HEAD_TO_COPY character for character into target_head; never put
the independently audited target there.

After generating all detector prompts, return two tight full-image boxes using
absolute pixel coordinates on the supplied IMAGE_WIDTH by IMAGE_HEIGHT image,
not normalized coordinates. Use left, top, right, bottom order. Replace the four
zeros in each box schema with measured integer coordinates. Output four separate
JSON numbers, never a quoted string: bbox_xyxy_pixels for the independently
audited target, and full_des_bbox_xyxy_pixels for FULLDES_TARGET_HEAD_TO_COPY.
Each box must preserve requested count and whole/part/region extent. These boxes
are later vote evidence and must not change any detector phrase.

Return exactly this bare JSON shape, filling every string and all three checks:
{"candidate_checks":[
{"target_head":"candidate noun","verdict":"exact|too_broad|too_narrow|wrong_participant","reason":"brief visual or query evidence"},
{"target_head":"candidate noun","verdict":"exact|too_broad|too_narrow|wrong_participant","reason":"brief visual or query evidence"},
{"target_head":"candidate noun","verdict":"exact|too_broad|too_narrow|wrong_participant","reason":"brief visual or query evidence"}],
"target_head":"best exact direct target noun",
"full_des_prompts":[
{"role":"full_des_canonical","target_head":"copy FULLDES_TARGET_HEAD_TO_COPY exactly","text":"plain FullDes target noun"},
{"role":"full_des_visual","target_head":"copy FULLDES_TARGET_HEAD_TO_COPY exactly","text":"observed appearance or position plus FullDes target"},
{"role":"full_des_extent","target_head":"copy FULLDES_TARGET_HEAD_TO_COPY exactly","text":"FullDes target with required count, extent, or relation"}],
"prompts":[
{"role":"visual_discriminator","target_head":"same direct target","text":"short noun phrase"},
{"role":"extent_relation","target_head":"same direct target","text":"short noun phrase"},
{"role":"contrast_view","target_head":"same direct target","text":"short noun phrase"},
{"role":"owner_relation","target_head":"same direct target","text":"short noun phrase"},
{"role":"concise_alias","target_head":"same direct target","text":"short noun phrase"}],
"bbox_xyxy_pixels":[0,0,0,0],
"full_des_bbox_xyxy_pixels":[0,0,0,0]}
Do not rename keys, nest prompts elsewhere, add prose, or use Markdown."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()




def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)




def clean_text(value: Any, field: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        raise ValueError(f"missing {field}")
    return text


def compact_prompt(value: Any, limit: int = 6) -> str:
    return " ".join(clean_text(value, "prompt").split()[:limit]).rstrip(",;:")


def target_key(value: Any) -> str:
    words = re.findall(r"[a-z0-9]+", compact_prompt(value, 6).casefold())
    if words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    if words:
        last = words[-1]
        if last.endswith("ies") and len(last) > 3:
            words[-1] = last[:-3] + "y"
        elif re.search(r"(?:ches|shes|xes|zes|ses)$", last):
            words[-1] = last[:-2]
        elif last.endswith("s") and not last.endswith("ss"):
            words[-1] = last[:-1]
    return " ".join(words)


def target_head(value: Any) -> str:
    words = compact_prompt(value, 6).split()
    if words and words[0].casefold() in {"a", "an", "the"}:
        words = words[1:]
    if not words:
        raise ValueError("missing target_head")
    return " ".join(words)


def targets_compatible(left: Any, right: Any) -> bool:
    """Treat ordinary noun modifiers as the same target, not a new entity."""

    left_words = target_key(left).split()
    right_words = target_key(right).split()
    if left_words == right_words:
        return True
    relation_words = {
        "at", "behind", "beside", "between", "by", "from", "in", "inside",
        "near", "of", "on", "over", "under", "with", "within",
    }
    if relation_words.intersection(left_words) or relation_words.intersection(right_words):
        return False
    shorter, longer = sorted((left_words, right_words), key=len)
    if not shorter:
        return False
    return shorter[-1] == longer[-1] or (
        len(shorter) == 1 and shorter[0] in longer
    )


def parse_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response is not a JSON object")
    fragment = text[start : end + 1]
    value, consumed = json.JSONDecoder().raw_decode(fragment)
    if fragment[consumed:].strip("} \t\r\n"):
        raise ValueError("response contains multiple objects or trailing data")
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value


_JSON_STRING = r'"(?:\\.|[^"\\])*"'


def salvage_identity_response(raw: str) -> dict[str, Any]:
    """Recover the declared identity and prompt objects from malformed outer JSON."""

    text = str(raw or "")
    contract = re.search(
        rf'"target_contract"\s*:\s*\{{[^{{}}]*?"target_head"\s*:\s*({_JSON_STRING})',
        text,
        re.DOTALL,
    )
    if contract is None:
        raise ValueError("malformed identity response has no recoverable target")
    head = target_head(json.loads(contract.group(1)))
    pattern = re.compile(
        rf'\{{\s*"role"\s*:\s*({_JSON_STRING})\s*,\s*'
        rf'"target_head"\s*:\s*({_JSON_STRING})\s*,\s*'
        rf'"text"\s*:\s*({_JSON_STRING})\s*\}}',
        re.DOTALL,
    )
    prompts = []
    for match in pattern.finditer(text):
        role, prompt_head, prompt_text = map(json.loads, match.groups())
        if target_key(prompt_head) == target_key(head):
            prompts.append({
                "role": role, "target_head": head, "text": prompt_text,
            })
    return {
        "target_contract": {"target_head": head},
        "candidate_checks": [{"target_head": head, "verdict": "exact"}],
        "prompts": prompts,
    }


class VisionQwen:
    """Independent OpenAI-compatible vision client; not the repo singleton."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.qwen_cfg = SimpleNamespace(model=model)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=QWEN_TIMEOUT_SECONDS,
            max_retries=QWEN_MAX_RETRIES,
        )
        self.ori_image: str | None = None
        self.image_size: tuple[int, int] | None = None

    @classmethod
    def from_config(cls, path: Path) -> "VisionQwen":
        import yaml

        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["qwen"]
        return cls(value["base_url"], value["api_key"], value["model"])

    def load_image(self, path: Path, max_side: int | None = None) -> None:
        max_side = IMAGE_MAX_SIDE if max_side is None else max_side
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            # Qwen2.5-VL grounds in pixels on a 28-pixel-aligned canvas.
            self.image_size = tuple(max(28, round(side / 28) * 28) for side in image.size)
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=92)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        self.ori_image = f"data:image/jpeg;base64,{encoded}"

    def call_json(
        self, instructions: str, payload: Mapping[str, Any],
        max_tokens: int | None = None,
    ) -> str:
        if not self.ori_image:
            raise RuntimeError("text image is not loaded")
        response = self.client.chat.completions.create(
            model=self.qwen_cfg.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": self.ori_image}},
                    {"type": "text", "text": json.dumps(
                        payload, ensure_ascii=False, sort_keys=True
                    )},
                ]},
            ],
            max_tokens=QWEN_MAX_TOKENS if max_tokens is None else max_tokens,
            temperature=QWEN_TEMPERATURE,
        )
        return (response.choices[0].message.content or "").strip()


def full_des_contract(result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("ground_truth_loaded") is not False:
        raise ValueError("FullDes result must be GT-free")
    full = result.get("full_des") or {}
    target = clean_text(full.get("target_entity"), "FullDes target")
    number = {"singular": "single", "plural": "multiple"}.get(
        str(full.get("number") or "").casefold(), "query_defined"
    )
    scope = str(full.get("scope") or "").casefold()
    role, extent = {
        "whole_object": ("object", "whole"),
        "physical_part": ("part", "part"),
        "visible_region": ("region", "region"),
        "place_support": ("region", "region"),
        "material_substance": ("material", "region"),
        "group_set": ("group", "group"),
    }.get(scope, ("object", "query_defined"))
    return {
        "direct_target_role": role,
        "target_head": target,
        "required_predicate": clean_text(
            full.get("selector") or "none", "FullDes selector"
        ),
        "number": number,
        "extent": extent,
        "relation": "none",
        "exclude": [],
    }


def full_des_context(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query": result["query"],
        "query_contract": dict(result.get("query_contract") or {}),
        "visual_binding": dict(result.get("visual_binding") or {}),
        "published_full_des": dict(result.get("full_des") or {}),
        "text_candidate_hypotheses": [
            {
                "prompt": clean_text(item.get("prompt"), "candidate prompt"),
                "target_claim": dict(item.get("target_claim") or {}),
            }
            for item in result.get("candidates") or []
            if (item.get("sam_input") or {}).get("type") == "text"
            and str(item.get("prompt") or "").strip()
        ],
    }


def join_prompt(target: str, modifier: Any) -> str:
    value = " ".join(str(modifier or "").split()).strip()
    if not value:
        return target
    target_words = target.casefold().split()
    value_words = value.casefold().split()
    return compact_prompt(
        value if all(word in value_words for word in target_words)
        else f"{target} {value}"
    )


def extent_fallback(target: str, contract: Mapping[str, Any]) -> str:
    if contract.get("number") == "multiple":
        return compact_prompt(f"all {target}", 8)
    if contract.get("extent") == "region":
        return compact_prompt(f"{target} area", 8)
    return compact_prompt(target, 8)


def fallback_texts(
    target: str,
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
    group: str,
) -> list[str]:
    full = result.get("full_des") or {}
    visual, selector = full.get("visible_discriminator"), full.get("selector")
    full_text = full.get("text") or target
    if group == "identity":
        return [
            compact_prompt(target, 8),
            join_prompt(target, visual or selector),
            extent_fallback(target, contract),
            join_prompt(target, selector),
            compact_prompt(full_text, 8),
        ]
    return [
        join_prompt(target, visual),
        join_prompt(target, selector),
        compact_prompt(full_text, 8),
        extent_fallback(target, contract),
        compact_prompt(target, 8),
    ]


def query_evidence_prompt(target: str, result: Mapping[str, Any]) -> str:
    """Make one deterministic target-preserving view from FullDes query evidence."""

    full = result.get("full_des") or {}
    enriched = target
    for value in (full.get("visible_discriminator"), full.get("selector")):
        if (
            str(value or "").strip()
            and targets_compatible(target, value)
            and prompt_mentions_target(str(value), target)
        ):
            enriched = target_head(value)
            break
    query_contract = result.get("query_contract") or {}
    evidence = clean_text(
        query_contract.get("query_evidence")
        or query_contract.get("answer_role_phrase")
        or full.get("selector")
        or target,
        "query evidence",
    )
    if len(evidence.split()) > 6 and query_contract.get("answer_role_phrase"):
        evidence = clean_text(query_contract["answer_role_phrase"], "answer role")
    evidence = target_key(evidence)
    return join_prompt(enriched, evidence)


def prompt_mentions_target(text: str, target: str) -> bool:
    def stem(word: str) -> str:
        if word.endswith("ies") and len(word) > 3:
            return word[:-3] + "y"
        if re.search(r"(?:ches|shes|xes|zes|ses)$", word):
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    prompt_words = {stem(word) for word in re.findall(r"[a-z0-9]+", text.casefold())}
    target_words = {stem(word) for word in re.findall(r"[a-z0-9]+", target.casefold())}
    return bool(target_words) and target_words <= prompt_words


def anchor_prompt(role: str, text: str, target: str) -> str:
    if role == "canonical_name":
        return compact_prompt(target, 8)
    if role == "detector_synonym" or prompt_mentions_target(text, target):
        return compact_prompt(text, 8)
    if role == "number_group":
        return compact_prompt(f"{text} {target}", 8)
    return compact_prompt(f"{target} {text}", 8)


def sanitize_prompt(role: str, text: str, target: str) -> str:
    if role != "contrast_view":
        return text
    positive = re.split(
        r"(?:[,;]|\band\b)\s*(?:not|no|without|except)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,;:-")
    return positive or target


def prompt_group(
    value: Mapping[str, Any],
    roles: Sequence[str],
    target: str,
    fallbacks: Sequence[str],
) -> list[dict[str, str]]:
    raw = value.get("prompts")
    if not isinstance(raw, list):
        raw = []
    usable: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        try:
            if target_key(item.get("target_head") or target) != target_key(target):
                continue
            role = str(item.get("role") or "")
            text = item.get("text") or item.get(role)
            if not str(text or "").strip():
                continue
            usable.append({
                "role": role,
                "text": compact_prompt(sanitize_prompt(role, str(text), target), 8),
            })
        except ValueError:
            continue
    selected, used = [], set()
    for index, role in enumerate(roles):
        match = next(
            (i for i, item in enumerate(usable) if i not in used and item["role"] == role),
            None,
        )
        if match is None:
            match = next((i for i in range(len(usable)) if i not in used), None)
        text = fallbacks[index] if match is None else usable[match]["text"]
        if match is not None:
            used.add(match)
        selected.append({
            "role": role,
            "target_head": target,
            "text": anchor_prompt(role, text, target),
        })
    return selected


def exact_target(value: Mapping[str, Any], target: str) -> bool:
    for check in value.get("candidate_checks") or []:
        if not isinstance(check, Mapping) or str(check.get("verdict")).casefold() != "exact":
            continue
        try:
            if target_key(check.get("target_head")) == target_key(target):
                return True
        except ValueError:
            pass
    return False


def normalized_box(
    value: Any, image_size: tuple[int, int] = (1000, 1000),
) -> list[float] | None:
    """Validate internal normalized boxes; raw pixel boxes require their canvas."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [
            float(coordinate) * 1000.0 / side
            for coordinate, side in zip(value, (*image_size, *image_size))
        ]
    except (TypeError, ValueError):
        return None
    if (
        any(not math.isfinite(coordinate) for coordinate in box)
        or any(coordinate < 0.0 or coordinate > 1000.0 for coordinate in box)
        or box[0] >= box[2]
        or box[1] >= box[3]
    ):
        return None
    return box


def compile_dual_responses(
    identity_raw: str,
    relation_raw: str,
    result: Mapping[str, Any],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Normalize two independent model responses without consulting GT."""

    locked = full_des_contract(result)
    locked_target = locked["target_head"]
    errors: list[str] = []
    notes: list[str] = []
    try:
        identity = parse_object(identity_raw)
        proposed = target_head((identity.get("target_contract") or {}).get("target_head"))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
        try:
            identity = salvage_identity_response(identity_raw)
            proposed = target_head(identity["target_contract"]["target_head"])
            notes.append(f"identity JSON locally recovered: {error}")
        except (TypeError, ValueError, json.JSONDecodeError):
            identity, proposed = {}, locked_target
            errors.append(f"identity: {error}")
    try:
        relation = parse_object(relation_raw)
        relation_target = target_head(relation.get("target_head"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        relation, relation_target = {}, locked_target
        errors.append(f"relation: {error}")

    agreed = target_key(proposed) == target_key(relation_target)
    target = (
        locked_target
        if agreed and target_key(proposed) == target_key(locked_target)
        else proposed if agreed else locked_target
    )
    if not agreed:
        notes.append("independent target disagreement; kept both exact views")
    contract = {**locked, "target_head": target}
    identity_native = prompt_group(
        identity, IDENTITY_ROLES, proposed,
        fallback_texts(proposed, contract, result, "identity"),
    )
    relation_native = prompt_group(
        relation, RELATION_ROLES, relation_target,
        fallback_texts(relation_target, contract, result, "relation"),
    )
    identity_locked = prompt_group(
        {}, IDENTITY_ROLES, locked_target,
        fallback_texts(locked_target, locked, result, "identity"),
    )
    relation_locked = prompt_group(
        {}, RELATION_ROLES, locked_target,
        fallback_texts(locked_target, locked, result, "relation"),
    )
    identity_prompts = (
        identity_native if agreed or exact_target(identity, proposed)
        else identity_locked
    )
    relation_prompts = (
        relation_native if agreed or exact_target(relation, relation_target)
        else relation_locked
    )
    relation_prompts.extend(prompt_group(
        {"prompts": relation.get("full_des_prompts")}
        if isinstance(relation.get("full_des_prompts"), list) else {},
        FULL_DES_ROLES,
        locked_target,
        [
            locked_target,
            fallback_texts(locked_target, locked, result, "relation")[0],
            extent_fallback(locked_target, locked),
        ],
    ))
    identity_prompts.append({
        "role": "query_evidence_anchor",
        "target_head": proposed,
        "text": query_evidence_prompt(proposed, result),
    })
    relation_prompts.append({
        "role": "query_evidence_anchor",
        "target_head": relation_target,
        "text": query_evidence_prompt(relation_target, result),
    })
    bbox_proposals = []
    for source, head, value, independently_exact in (
        (
            "identity", proposed, identity.get("bbox_xyxy_pixels"),
            exact_target(identity, proposed),
        ),
        (
            "relation", relation_target, relation.get("bbox_xyxy_pixels"),
            exact_target(relation, relation_target),
        ),
        (
            "full_des", locked_target,
            relation.get("full_des_bbox_xyxy_pixels"),
            True,
        ),
    ):
        box = normalized_box(value, image_size)
        if box is not None and independently_exact:
            bbox_proposals.append({
                "source": source,
                "target_head": head,
                "bbox_xyxy_1000": box,
            })
    return {
        "target_contract": contract,
        "identity_prompts": identity_prompts,
        "relation_prompts": relation_prompts,
        "bbox_proposals": bbox_proposals,
        "bbox_image_size": list(image_size),
        "raw_bbox_coordinate_space": "qwen_image_pixels",
        "raw_responses": [identity_raw, relation_raw],
        "compile_status": "fallback" if errors else "recovered" if notes else "model",
        "compile_notes": errors + notes,
        "qwen_calls": 2,
        "ground_truth_loaded": False,
    }


def generate_text_augmentation(
    qwen: VisionQwen,
    image_path: Path,
    full_result: Mapping[str, Any],
) -> dict[str, Any]:
    qwen.load_image(image_path)
    contract = full_des_contract(full_result)
    width, height = qwen.image_size
    canvas = {"IMAGE_WIDTH": width, "IMAGE_HEIGHT": height}
    identity_payload = {"QUERY": full_result["query"], **canvas}
    relation_payload = {
        "QUERY": full_result["query"],
        **canvas,
        "FULLDES_TARGET_CONTRACT": contract,
        "FULLDES_TARGET_HEAD_TO_COPY": contract["target_head"],
        "FULLDES_CONTEXT": full_des_context(full_result),
    }
    calls = [
        lambda: qwen.call_json(IDENTITY_INSTRUCTIONS, identity_payload),
        lambda: qwen.call_json(RELATION_INSTRUCTIONS, relation_payload),
    ]
    identity_raw, relation_raw = (
        [call() for call in calls]
        if max(width, height) > 1500
        else run_parallel_calls(calls)
    )
    compiled = compile_dual_responses(identity_raw, relation_raw, full_result, qwen.image_size)
    if compiled["qwen_calls"] != 2:
        raise AssertionError("text augmentation must use exactly two Qwen calls")
    return compiled


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("mask shapes differ")
    union = int(np.logical_or(left, right).sum())
    return int(np.logical_and(left, right).sum()) / union if union else 0.0


def thumbnail(mask: np.ndarray, size: int = 128) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not all(mask.shape):
        raise ValueError("mask must be non-empty and two-dimensional")
    if max(mask.shape) <= size:
        return mask
    rows = np.minimum(
        (np.arange(size) * mask.shape[0] / size).astype(int), mask.shape[0] - 1
    )
    columns = np.minimum(
        (np.arange(size) * mask.shape[1] / size).astype(int), mask.shape[1] - 1
    )
    return mask[np.ix_(rows, columns)]


def deduplicate_prompts(
    identity: Sequence[Mapping[str, Any]],
    relation: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    unique: list[dict[str, Any]] = []
    by_text: dict[str, int] = {}
    indices = {"identity": [], "relation": []}
    for route, prompts in (("identity", identity), ("relation", relation)):
        for prompt in prompts:
            key = clean_text(prompt.get("text"), "text prompt")
            if key not in by_text:
                by_text[key] = len(unique)
                unique.append(dict(prompt))
            indices[route].append(by_text[key])
    return unique, indices


def normalize_sam_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask).squeeze().astype(bool)
    if value.shape != shape:
        value = cv2.resize(
            value.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return np.ascontiguousarray(value)


def namespace_prompt_index(prompt_index: int, route: str) -> int:
    if route == "identity":
        return prompt_index // 3 if prompt_index < 9 else 6 + prompt_index - 9
    if route == "relation":
        return 3 + prompt_index if prompt_index < 3 else 20 + prompt_index
    raise ValueError(f"unknown text route: {route}")


def load_sam_image(sam3: Any, image_path: Path) -> None:
    """Use the same EXIF-corrected pixels as the text Qwen calls."""

    with Image.open(image_path) as image:
        if int(image.getexif().get(274, 1) or 1) == 1:
            sam3.load_image(str(image_path))
            return
        with tempfile.TemporaryDirectory(prefix="reasonseg-oriented-") as directory:
            path = Path(directory) / "image.png"
            ImageOps.exif_transpose(image).convert("RGB").save(path)
            sam3.load_image(str(path))


def run_text_masks(
    sam3: Any,
    image_path: Path,
    augmentation: Mapping[str, Any],
    shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Run one SAM call per unique text and return Top-3 plus all candidates."""

    relation_prompts = [
        prompt for prompt in augmentation["relation_prompts"]
        if prompt.get("role") != "concise_alias"
    ]
    unique, route_indices = deduplicate_prompts(
        augmentation["identity_prompts"], relation_prompts
    )
    load_sam_image(sam3, image_path)
    unique_results: list[list[dict[str, Any]]] = []
    for prompt in unique:
        results = []
        for item in sam3.predict_text(prompt["text"]) or []:
            mask = normalize_sam_mask(item["mask"], shape)
            results.append({
                "mask": mask,
                "confidence": float(item.get("conf", 0.0)),
                "stability": float(item.get("stability_score", 0.0)),
            })
        unique_results.append(results)

    candidates: list[dict[str, Any]] = []
    for route, prompts in (
        ("identity", augmentation["identity_prompts"]),
        ("relation", relation_prompts),
    ):
        for prompt_index, (prompt, unique_index) in enumerate(
            zip(prompts, route_indices[route])
        ):
            namespaced_index = namespace_prompt_index(prompt_index, route)
            raw = unique_results[unique_index]
            for candidate_index, item in enumerate(raw):
                if not item["mask"].any():
                    continue
                candidates.append({
                    **item,
                    "mask_sha256": mask_sha256(item["mask"]),
                    "source": "augmentation_raw",
                    "pipeline": route,
                    "prompt_index": namespaced_index,
                    "original_prompt_index": prompt_index,
                    "candidate_index": candidate_index,
                    "prompt": dict(prompt),
                })
            union = (
                np.logical_or.reduce([item["mask"] for item in raw])
                if raw else np.zeros(shape, dtype=bool)
            )
            if not union.any():
                continue
            candidates.append({
                "mask": np.ascontiguousarray(union),
                "mask_sha256": mask_sha256(union),
                "confidence": max((item["confidence"] for item in raw), default=0.0),
                "stability": max((item["stability"] for item in raw), default=0.0),
                "source": "augmentation_union",
                "pipeline": route,
                "prompt_index": namespaced_index,
                "original_prompt_index": prompt_index,
                "candidate_index": -1,
                "prompt": dict(prompt),
            })
    top3 = rank_text_top3(candidates, reserved_role="query_evidence_anchor")
    return top3, candidates, len(unique)


def rank_candidates(
    candidates: Sequence[Mapping[str, Any]], threshold: float = 0.90
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate["mask_sha256"]), []).append(candidate)
    values = []
    for occurrences in grouped.values():
        representative = max(
            occurrences,
            key=lambda item: (
                item.get("source") == "augmentation_union",
                float(item.get("confidence", 0.0)),
            ),
        )
        values.append({
            "candidate": dict(representative),
            "occurrences": occurrences,
            "thumbnail": thumbnail(representative["mask"]),
        })
    def route_key(item: Mapping[str, Any]) -> tuple[str, int]:
        return (
            str(item.get("pipeline") or ""),
            int(item.get("original_prompt_index", item.get("prompt_index", 0))),
        )

    routes = sorted({route_key(item) for item in candidates})
    route_members = {
        current_route: [
            index
            for index, value in enumerate(values)
            if any(
                route_key(item) == current_route
                for item in value["occurrences"]
            )
        ]
        for current_route in routes
    }
    pipelines = sorted({str(item.get("pipeline") or "") for item in candidates})
    pipeline_members = {
        pipeline: [
            index
            for index, value in enumerate(values)
            if any(
                str(item.get("pipeline") or "") == pipeline
                for item in value["occurrences"]
            )
        ]
        for pipeline in pipelines
    }
    pairwise = [
        [
            mask_iou(left["thumbnail"], right["thumbnail"])
            for right in values
        ]
        for left in values
    ]
    ranked = []
    for index, value in enumerate(values):
        by_route = []
        for current_route in routes:
            similarities = [
                pairwise[index][other]
                for other in route_members[current_route]
            ]
            by_route.append(max(similarities, default=0.0))
        by_pipeline = [
            max(
                (pairwise[index][other] for other in pipeline_members[pipeline]),
                default=0.0,
            )
            for pipeline in pipelines
        ]
        candidate = value["candidate"]
        candidate["independent_consensus_support"] = sum(
            score >= threshold for score in by_pipeline
        )
        candidate["independent_consensus_mean_iou"] = (
            sum(by_pipeline) / len(by_pipeline) if by_pipeline else 0.0
        )
        candidate["consensus_support"] = sum(score >= threshold for score in by_route)
        candidate["consensus_mean_iou"] = (
            sum(by_route) / len(by_route) if by_route else 0.0
        )
        ranked.append(candidate)
    ranked.sort(
        key=lambda item: (
            int(item["consensus_support"]),
            float(item["consensus_mean_iou"]),
            float(item.get("confidence", 0.0)),
            int(item["independent_consensus_support"]),
            float(item["independent_consensus_mean_iou"]),
        ),
        reverse=True,
    )
    return ranked


def distinct(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    threshold: float,
) -> bool:
    return all(
        candidate["mask_sha256"] != other["mask_sha256"]
        and mask_iou(thumbnail(candidate["mask"]), thumbnail(other["mask"])) < threshold
        for other in selected
    )


def reserve_prompt_role(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    role: str,
    threshold: float = .90,
) -> list[dict[str, Any]]:
    """Reserve the third slot for one distinct semantic view when requested."""

    selected = [dict(item) for item in selected]
    if any((item.get("prompt") or {}).get("role") == role for item in selected):
        return selected
    role_candidates = [
        item for item in candidates
        if (item.get("prompt") or {}).get("role") == role
        and item.get("source") == "augmentation_union"
    ]
    anchor = next((
        item for item in rank_candidates(role_candidates)
        if distinct(item, selected[:2], threshold)
    ), None)
    if anchor is None:
        return selected
    return [*selected[:2], anchor] if len(selected) >= 2 else [*selected, anchor]


def rank_text_top3(
    candidates: Sequence[Mapping[str, Any]], threshold: float = 0.90,
    reserved_role: str | None = None,
) -> list[dict[str, Any]]:
    unions = [item for item in candidates if item.get("source") == "augmentation_union"]
    consensus = rank_candidates(unions or candidates, threshold)
    if consensus:
        base = consensus[0]
        base_thumb = thumbnail(base["mask"])
        cluster = [
            item for item in consensus
            if mask_iou(base_thumb, thumbnail(item["mask"])) >= 0.90
        ]
        if len(cluster) > 1:
            combined = np.logical_or.reduce([item["mask"] for item in cluster])
            composed = {
                **base,
                "mask": np.ascontiguousarray(combined),
                "mask_sha256": mask_sha256(combined),
                "source": "augmentation_cluster_union",
                "cluster_size": len(cluster),
            }
            consensus = [composed, *consensus]
    fallback = rank_candidates(candidates, threshold)
    selected: list[dict[str, Any]] = []
    for ranking in (consensus, consensus, consensus, fallback):
        candidate = next(
            (item for item in ranking if distinct(item, selected, threshold)), None
        )
        if candidate is not None:
            selected.append(candidate)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        for candidate in rank_candidates(candidates, 0.95):
            if distinct(candidate, selected, 0.95):
                selected.append(candidate)
            if len(selected) == 3:
                break
    if reserved_role:
        selected = reserve_prompt_role(
            selected, candidates, reserved_role, threshold,
        )
    return selected


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(rows):
        return None
    return [
        int(columns.min()), int(rows.min()),
        int(columns.max()) + 1, int(rows.max()) + 1,
    ]


def consensus_bbox_top3(sam, shape, proposals):
    height, width = shape
    candidates, cache = [], {}
    for proposal in proposals:
        box = normalized_box(proposal['bbox_xyxy_1000'])
        if box is None:
            continue
        pixels = tuple(round(value * scale / 1000) for value, scale in zip(box, (width, height, width, height)))
        if pixels not in cache:
            cache[pixels] = [{'mask': normalize_sam_mask(item['mask'], shape),
                'confidence': float(item.get('conf', 0.)), 'stability': float(item.get('stability_score', 0.))}
                for item in sam.predict_box(list(pixels)) or []]
        raw = [item for item in cache[pixels] if item['mask'].any()]
        if not raw:
            continue
        union = {'mask': np.logical_or.reduce([item['mask'] for item in raw]),
                 'confidence': max(item['confidence'] for item in raw),
                 'stability': max(item['stability'] for item in raw)}
        for item in [*raw, union]:
            candidates.append({**item, 'mask_sha256': mask_sha256(item['mask']),
                'pipeline': proposal['source'], 'original_prompt_index': 0,
                'bbox_source': proposal['source'], 'bbox': list(pixels),
                'source': 'bbox_refinement', 'vote_eligible': True})
    selected = []
    for candidate in rank_candidates(candidates):
        if distinct(candidate, selected, .9):
            selected.append(candidate)
        if len(selected) == 3:
            break
    return selected


def full_des_roi_proposals(full, selected_mask):
    if full.get('ground_truth_loaded') is not False:
        raise ValueError('FullDes is not GT-free')
    masks = [('selected', selected_mask)]
    for item in full.get('candidates', []):
        if not ((item.get('source') == 'planner' and item.get('variant') == 'top1')
                or item.get('variant') in ('top2_union', 'all_union')):
            continue
        info = item['mask']
        path = Path(info['path'])
        if file_sha256(path) != info['sha256']:
            raise ValueError('FullDes candidate file hash mismatch')
        with Image.open(path) as image:
            mask = np.asarray(image) > 0
        if list(mask.shape) != info['shape'] or mask_sha256(mask) != info['raw_sha256']:
            raise ValueError('FullDes candidate mask integrity mismatch')
        masks.append((item['source'] + '_' + item['variant'],
                      orient_mask_to_image(mask, Path(full['image_path']))))
    height, width = selected_mask.shape
    proposals, seen = [], set()
    for source, mask in masks:
        if mask.shape != selected_mask.shape:
            raise ValueError('FullDes ROI shape mismatch')
        box = bbox_from_mask(mask)
        if box is not None and tuple(box) not in seen:
            seen.add(tuple(box))
            proposals.append({'source': 'full_des_roi_' + source,
                'bbox_xyxy_1000': [value * 1000 / scale
                    for value, scale in zip(box, (width, height, width, height))]})
    return proposals


def run_bbox_augmentation(sam3, full_result, full_des_mask, augmentation):
    """Refine two-call text boxes and FullDes geometry; make no extra Qwen call."""
    proposals = [
        dict(item) for item in augmentation.get("bbox_proposals") or []
        if normalized_box(item.get("bbox_xyxy_1000")) is not None
    ]
    proposals.extend(full_des_roi_proposals(full_result, full_des_mask))
    return consensus_bbox_top3(sam3, full_des_mask.shape, proposals)


def _candidate_masks_info(
    text_top3: Sequence[Mapping[str, Any]],
    full_des_mask: np.ndarray,
    full_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for rank, candidate in enumerate(text_top3, 1):
        prompt = dict(candidate.get("prompt") or {})
        records.append({
            **candidate,
            "id": f"Text Top{rank}",
            "source": "text",
            "role": prompt.get("role"),
            "text": prompt.get("text"),
            "mask": np.ascontiguousarray(candidate["mask"], dtype=bool),
        })
    full_des = dict(full_result.get("full_des") or {})
    records.append({
        "id": "FullDes",
        "source": "full_des",
        "role": "full_des",
        "text": full_des.get("text"),
        "target_entity": full_des.get("target_entity"),
        "scope": full_des.get("scope"),
        "mask": np.ascontiguousarray(full_des_mask, dtype=bool),
    })
    return records



def orient_mask_to_image(mask: np.ndarray, image_path: Path) -> np.ndarray:
    """Match a raw-image mask to the EXIF-corrected image seen by Qwen/SAM."""

    value = np.ascontiguousarray(mask, dtype=bool)
    with Image.open(image_path) as image:
        orientation = int(image.getexif().get(274, 1) or 1)
        raw_shape = (image.height, image.width)
        canonical = ImageOps.exif_transpose(image)
        canonical_shape = (canonical.height, canonical.width)
    transforms = {
        1: None,
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    if orientation not in transforms:
        raise ValueError(f"unsupported EXIF orientation: {orientation}")
    if orientation != 1 and (
        value.shape == raw_shape or raw_shape == canonical_shape
    ):
        value = np.asarray(
            Image.fromarray(value.astype(np.uint8)).transpose(
                transforms[orientation]
            ),
            dtype=bool,
        )
    if value.shape != canonical_shape:
        raise ValueError(
            f"FullDes mask shape {value.shape} != canonical image {canonical_shape}"
        )
    return np.ascontiguousarray(value)


def load_full_des_mask(
    result: Mapping[str, Any], image_path: Path
) -> np.ndarray:
    path = Path(clean_text((result.get("mask") or {}).get("path"), "FullDes mask"))
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise ValueError(f"unreadable FullDes mask: {path}")
    raw_mask = np.ascontiguousarray(value > 0)
    record = result.get("mask") or {}
    expected = record.get("raw_sha256")
    if expected and mask_sha256(raw_mask) != expected:
        raise ValueError("FullDes mask hash mismatch")
    return orient_mask_to_image(raw_mask, image_path)


def candidate_metadata(candidate: Mapping[str, Any], array_key: str) -> dict[str, Any]:
    return {
        "array_key": array_key,
        "mask_sha256": candidate["mask_sha256"],
        "area": int(np.asarray(candidate["mask"], dtype=bool).sum()),
        "source": candidate.get("source"),
        "pipeline": candidate.get("pipeline"),
        "prompt_index": candidate.get("prompt_index"),
        "prompt": candidate.get("prompt"),
        "confidence": float(candidate.get("confidence", 0.0)),
        "stability": float(candidate.get("stability", 0.0)),
        "consensus_support": candidate.get("consensus_support"),
        "consensus_mean_iou": candidate.get("consensus_mean_iou"),
        "independent_consensus_support": candidate.get(
            "independent_consensus_support"
        ),
        "independent_consensus_mean_iou": candidate.get(
            "independent_consensus_mean_iou"
        ),
        "bbox": candidate.get("bbox"),
        "bbox_source": candidate.get("bbox_source"),
        "vote_eligible": bool(candidate.get("vote_eligible", False)),
        "source_text_rank": candidate.get("source_text_rank"),
    }


def save_candidate_archive(
    path: Path,
    text_top3: Sequence[Mapping[str, Any]],
    bbox_candidates: Sequence[Mapping[str, Any]],
    full_des_mask: np.ndarray,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {"full_des": full_des_mask.astype(np.uint8)}
    text_records, bbox_records = [], []
    for index, candidate in enumerate(text_top3, 1):
        key = f"text_{index}"
        arrays[key] = np.asarray(candidate["mask"], dtype=np.uint8)
        text_records.append(candidate_metadata(candidate, key))
    for index, candidate in enumerate(bbox_candidates, 1):
        key = f"bbox_{index}"
        arrays[key] = np.asarray(candidate["mask"], dtype=np.uint8)
        bbox_records.append(candidate_metadata(candidate, key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "text_top3": text_records,
        "bbox_top3": bbox_records,
    }


@dataclass
class Engines:
    full_des_qwen: Any
    text_qwen: VisionQwen
    sam3: Any
    full_des_api: Any


def full_des_stage_dir(full_des_api: Any, sample_output_dir: Path) -> Path:
    """Allocate a new per-run FullDes directory inside its required root."""

    required_root = getattr(full_des_api, "FINAL_OUTPUT_ROOT", None)
    if required_root is None:
        return sample_output_dir / "full_des"
    batch_output = sample_output_dir.parents[1]
    run_hash = hashlib.sha256(str(batch_output).encode("utf-8")).hexdigest()[:12]
    run_name = f"{safe_name(batch_output.name)}_{run_hash}"
    return (
        Path(required_root).resolve()
        / "clean_e2e_runs"
        / run_name
        / sample_output_dir.name
    )


def run_candidate_sample(
    image_path: Path,
    query: str,
    output_dir: Path,
    engines: Engines,
) -> dict[str, Any]:
    """Generate and persist every pre-vote mask candidate for one image."""

    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    query = clean_text(query, "query")
    output_dir = Path(output_dir).resolve()
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"candidate output already exists: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"incomplete sample output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    full_stage = full_des_stage_dir(engines.full_des_api, output_dir)
    full_result = engines.full_des_api.generate(
        str(image_path), query, engines.full_des_qwen, engines.sam3,
        str(full_stage),
    )
    full_mask = load_full_des_mask(full_result, image_path)
    shape = tuple(int(value) for value in full_mask.shape)

    augmentation = generate_text_augmentation(
        engines.text_qwen, image_path, full_result
    )
    write_json(output_dir / "text_augmentation.json", augmentation)
    text_top3, all_text, sam_text_calls = run_text_masks(
        engines.sam3, image_path, augmentation, shape
    )

    bbox_candidates = run_bbox_augmentation(
        engines.sam3, full_result, full_mask, augmentation,
    )
    del all_text
    archive = save_candidate_archive(
        output_dir / "candidates.npz", text_top3, bbox_candidates, full_mask
    )
    candidate_masks_info = _candidate_masks_info(
        text_top3, full_mask, full_result
    )

    result = {
        "schema_version": "reasonseg-candidates-v1",
        "image_path": str(image_path),
        "image_sha256": file_sha256(image_path),
        "query": query,
        "stage_order": list(STAGE_ORDER),
        "full_des_result": str(Path(full_result["result_path"]).resolve()),
        "full_des_result_sha256": file_sha256(Path(full_result["result_path"])),
        "text_qwen_model": engines.text_qwen.qwen_cfg.model,
        "text_qwen_calls": 2,
        "sam_text_prompt_calls": sam_text_calls,
        "bbox_qwen_calls": 0,
        "bbox_source": "two_call_text_boxes_and_fresh_full_des_roi",
        "candidates": archive,
        "ground_truth_loaded": False,
    }
    write_json(result_path, result)
    write_json(output_dir / "prediction_seal.json", {
        "schema_version": "reasonseg-candidates-seal-v1",
        "result_sha256": file_sha256(result_path),
        "candidate_archive_sha256": archive["sha256"],
        "text_qwen_calls": 2,
        "stage_order": list(STAGE_ORDER),
        "ground_truth_loaded": False,
    })
    return {
        **result,
        "result_path": str(result_path.resolve()),
        "candidate_masks_info": candidate_masks_info,
        "full_des": dict(full_result.get("full_des") or {}),
    }






def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return name[:80] or "sample"
