"""Full S/L experiments: identity-locked alternatives to frozen V4 Full.

One file intentionally owns prompts, gates, evaluation, and summaries so later
S/L iterations do not create a version/file tree. No visualization is persisted.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path


from ref_prompt import full_description_pipeline as v4_eval
from ref_prompt import target_contract_prompts as v4
from src.models import bbox_iou
from src.utils import (
    save_labeled_mask_overlay,
    save_selected_target_views,
    save_target_candidate_views,
)


THRESHOLD = 0.50
STRICT_THRESHOLD = 0.75
SCHEMA_VERSION = "full-sl"
SYSTEM_REFERENCE = "Ground one subordinate reference instance without changing the locked target. Return JSON only."
SYSTEM_TARGET_SELECTION = "Select exactly one locked target candidate from query semantics. Return JSON only."
SYSTEM_VISUAL_CUE = "Report optional visible evidence for an already frozen target candidate. Return JSON only."
SYSTEM_TARGET_BBOX = "Project the frozen highlighted target evidence to one bbox. Never reselect an instance. Output coordinates only."
SYSTEM_QUERY_REFERENCE = "Ground one query-native NEVER TARGET reference without changing the locked target. Return JSON only."
GRAMMAR_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
    "to", "of", "in", "on", "at", "by", "for", "from", "with", "and",
    "or", "that", "which", "who", "whose", "this", "it", "its", "as", "without",
    "has", "have", "having", "their", "there", "one",
}
REFERENCE_QWEN_FROZEN_MIN_IOU = 0.50
REFERENCE_SAM_FROZEN_MIN_IOU = 0.70
REFERENCE_QWEN_SAM_MIN_IOU = 0.50
REFERENCE_UNIQUENESS_RATIO = 0.60
REFERENCE_UNIQUENESS_MARGIN = 0.15
REFERENCE_SHIFT_BLOCK_IOU = 0.80
REFERENCE_EXACT_COPY_BLOCK_IOU = 0.90
NEGATION_RE = re.compile(
    r"\b(?:no|not|without|never|doesn\s*'?\s*t|isn\s*'?\s*t|aren\s*'?\s*t)\b",
    re.I,
)
TARGET_OWNED_WORDS = v4.IMPLICIT_OWNER_WORDS | {
    "back", "beard", "beards", "ear", "ears", "eye", "eyes", "mouth",
    "nose", "tail", "tails", "torso", "chest", "waist", "wrist", "wrists",
    "ankle", "ankles", "legging", "leggings", "boot", "boots", "clothing",
}
SHORT_DELETABLE_WORDS = {"a", "an", "the"}
V2_TARGET_SAM_MIN_IOU = 0.50
V2_TARGET_FINAL_MIN_IOU = 0.50
V2_TARGET_FINAL_OTHER_MARGIN = 0.05
V2_MAX_TARGET_CANDIDATES = 12
QUERY_REFERENCE_SAM_MIN_IOU = 0.45
QUERY_REFERENCE_SAME_CLASS_MIN_IOU = 0.70
QUERY_REFERENCE_TARGET_OVERLAP_IOU = 0.50
V2_CUE_TYPES = {
    "color", "pattern", "material", "shape", "size", "clothing",
    "body_feature", "state",
}
V2_UNSAFE_CUE_WORDS = {
    "left", "right", "top", "bottom", "middle", "center", "front", "back",
    "foreground", "background", "behind", "beside", "between", "near", "next",
    "above", "below", "under", "over", "inside", "outside", "first", "second",
    "third", "fourth", "last", "farthest", "nearest", "holding", "carrying",
    "riding", "touching", "looking", "following", "approaching", "wearing",
    "catching", "feeding", "pulling", "pushing", "with", "without",
}
V2_UI_CUE_WORDS = {
    "mask", "masked", "overlay", "overlaid", "panel", "candidate",
    "highlight", "highlighted", "outline", "outlined", "label", "labeled",
    "bounding", "bbox", "box",
}
QUERY_REFERENCE_INVALID_HEADS = {
    "edge", "side", "corner", "front", "back", "picture", "image",
    "row", "top", "bottom", "middle", "left", "right", "one",
}
QUERY_REFERENCE_SAME_CLASS_PROXIES = {
    "baby", "adult", "larger", "smaller", "bigger", "small", "big",
    "another", "other", "one", "ones",
}
def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", (text or "").lower())


def normalized(text: str) -> str:
    return " ".join(words(text))


def content_tokens(text: str) -> set[str]:
    output = set()
    for token in words(text):
        if token in GRAMMAR_WORDS:
            continue
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        output.add(token)
    return output


def content_sequence(text: str) -> list[str]:
    output = []
    for token in words(text):
        if token in GRAMMAR_WORDS:
            continue
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        output.append(token)
    return output


def ordered_subsequence_deletions(source: list[str], candidate: list[str]) -> list[str] | None:
    deleted, cursor = [], 0
    for token in source:
        if cursor < len(candidate) and token == candidate[cursor]:
            cursor += 1
        else:
            deleted.append(token)
    return deleted if cursor == len(candidate) else None


def exact_target_prefix(description: str, target_name: str) -> bool:
    required = ["the", *words(target_name)]
    return words(description)[:len(required)] == required


def mandatory_cues(row: dict) -> list[str]:
    return v4.required_literal_cues(
        row["static_query_plan"], row["target"]["target_name"],
        row["query"], row["target"],
    )


def frozen_reference_objects(row: dict) -> list[dict]:
    rounds = row.get("identity_rounds") or []
    if rounds:
        references = (rounds[0].get("query_plan") or {}).get("reference_objects") or []
        if references:
            return references
    return [
        {key: item.get(key) for key in ("candidate_id", "name", "sam_prompt", "relation")}
        for item in row.get("reference_evidence") or []
        if item.get("candidate_id") and item.get("sam_prompt")
    ]


def gate_target(row: dict) -> dict:
    target = dict(row["target"])
    references = frozen_reference_objects(row)
    if references:
        target["reference_span"] = " ".join(
            str(item.get("name") or item.get("sam_prompt") or "") for item in references
        ).strip()
    return target


def semantic_violations(row: dict, description: str) -> list[str]:
    return v4.description_violations(
        row["query"], gate_target(row), description, row["static_query_plan"]
    )


def short_role_problems(row: dict, description: str) -> list[str]:
    problems = []
    full_tokens = content_tokens(row["descriptions"]["v4_full"])
    description_tokens = content_tokens(description)
    for candidate in row["static_query_plan"].get("reference_candidates") or []:
        signature = content_tokens(str(candidate.get("query_phrase", ""))) & full_tokens
        missing = sorted(signature - description_tokens)
        if missing:
            problems.append(
                f"reference_cue_lost:{str(candidate.get('id', '?')).upper()}:" + ",".join(missing)
            )
    source = f"{row['query']} {row['descriptions']['v4_full']}"
    if NEGATION_RE.search(source):
        if not NEGATION_RE.search(description):
            problems.append("negation_lost")
        else:
            match = NEGATION_RE.search(row["query"]) or NEGATION_RE.search(row["descriptions"]["v4_full"])
            anchor_tokens = []
            if match:
                anchor_source = (
                    row["query"][match.end():]
                    if match.string == row["query"]
                    else row["descriptions"]["v4_full"][match.end():]
                )
                for token in words(anchor_source):
                    for content in content_tokens(token):
                        if content not in anchor_tokens:
                            anchor_tokens.append(content)
                    if len(anchor_tokens) >= 4:
                        break
            missing = sorted(set(anchor_tokens) - description_tokens)
            if missing:
                problems.append("negation_anchor_lost:" + ",".join(missing))
    return problems


def deterministic_short_raw(row: dict) -> str:
    """Delete one internal determiner; never let Short rewrite semantic content."""
    full = row["descriptions"]["v4_full"]
    target_name = row["target"]["target_name"]
    prefix = re.match(
        r"^\s*the\s+" + r"\s+".join(re.escape(token) for token in words(target_name)) + r"\b",
        full,
        re.I,
    )
    if not prefix:
        return json.dumps({
            "object_name": target_name,
            "description": full,
            "source": "deterministic_grammar",
            "deleted_tokens": [],
        })
    matches = [
        match for match in re.finditer(r"\b(?:a|an|the)\b", full, re.I)
        if match.start() >= prefix.end()
    ]
    if not matches:
        return json.dumps({
            "object_name": target_name,
            "description": full,
            "source": "deterministic_grammar",
            "deleted_tokens": [],
        })
    match = matches[-1]
    description = (full[:match.start()] + full[match.end():]).strip()
    description = re.sub(r"\s+", " ", description)
    description = re.sub(r"\s+([,.;:!?])", r"\1", description)
    return json.dumps({
        "object_name": target_name,
        "description": description,
        "source": "deterministic_grammar",
        "deleted_tokens": [match.group(0).lower()],
    })


def reference_prompt(row: dict, safe_references: list[dict]) -> str:
    target = row["target"]
    width, height = row["image_size"]
    allowed_ids = [item["candidate_id"] for item in safe_references] + ["none"]
    return f"""The original image is loaded.
ORIGINAL QUERY: {row['query']}
FROZEN FULL DESCRIPTION: {row['descriptions']['v4_full']}
LOCKED TARGET NAME: {target['target_name']}
LOCKED TARGET SCOPE: {target['target_scope']}
SAFE FROZEN V4 REFERENCE INSTANCES: {json.dumps(safe_references, ensure_ascii=False)}
ALLOWED CANDIDATE ID STRINGS: {json.dumps(allowed_ids, ensure_ascii=False)}
IMAGE SIZE: width={width}, height={height}

Choose at most one listed physical REFERENCE instance used only to locate the
locked target. Each listed name, SAM prompt, and relation is immutable: do not
rewrite them and do not choose anything outside this list.
Return a tight Qwen bbox for that exact listed reference instance. It is a
NEGATIVE INSTANCE and must NEVER become the target. If target and reference
share a noun class, localize only the specific reference instance identified
by its listed attributes/relation. candidate_id must be one exact string from
ALLOWED CANDIDATE ID STRINGS. Never output a combined phrase such as
"R1 or none". If uncertain, use candidate_id="none".

Return exactly:
{{"object_name":"{target['target_name']}","candidate_id":"{safe_references[0]['candidate_id']}","reference_bbox":[x1,y1,x2,y2]}}
"""


def parse_response(raw: str) -> dict:
    value = v4.extract_json(raw)
    return value if isinstance(value, dict) else {}


def common_problems(row: dict, parsed: dict, normalize_prefix: bool = True) -> tuple[str, list[str]]:
    target_name = row["target"]["target_name"]
    description = str(parsed.get("description", "")).strip()
    description = (
        v4.normalize_description_prefix(description, target_name)
        if normalize_prefix else re.sub(r"\s+", " ", description)
    )
    problems = []
    if normalized(str(parsed.get("object_name", ""))) != normalized(target_name):
        problems.append("object_name_shift")
    if not description:
        problems.append("empty_description")
    elif not exact_target_prefix(description, target_name):
        problems.append("object_prefix_shift")
    return description, problems


def gate_short(row: dict, raw: str) -> tuple[str | None, dict]:
    full = row["descriptions"]["v4_full"]
    parsed = parse_response(raw)
    description, problems = common_problems(row, parsed, normalize_prefix=False)
    if description:
        full_words, short_words = words(full), words(description)
        deleted = ordered_subsequence_deletions(full_words, short_words)
        if deleted is None:
            problems.append("not_full_ordered_subsequence")
            deleted = []
        elif len(deleted) != 1 or any(token not in SHORT_DELETABLE_WORDS for token in deleted):
            problems.append("unsafe_grammar_edit:" + ",".join(deleted or ["none"]))
        if content_sequence(description) != content_sequence(full):
            problems.append("content_sequence_changed")
        for cue in mandatory_cues(row):
            cue_content = content_tokens(cue)
            if cue_content and not cue_content <= content_tokens(description):
                problems.append("mandatory_cue_lost:" + normalized(cue))
        problems.extend(semantic_violations(row, description))
        problems.extend(short_role_problems(row, description))
        if normalized(description) == normalized(full):
            problems.append("duplicate_full")
        if len(words(description)) >= len(words(full)):
            problems.append("not_shorter")
    problems = sorted(set(problems))
    return (None if problems else description), {
        "accepted": not problems,
        "problems": problems,
        "object_name": parsed.get("object_name"),
        "source": parsed.get("source"),
        "deleted_tokens": parsed.get("deleted_tokens") or [],
        "full_tokens": len(words(full)),
        "short_tokens": len(words(description)),
        "content_sequence_equal": content_sequence(description) == content_sequence(full),
        "prefix_repaired": False,
    }


def phrase_match(text: str, phrase: str):
    tokens = words(phrase)
    if not tokens:
        return None
    return re.search(
        r"\b" + r"\s+".join(re.escape(token) for token in tokens) + r"\b",
        text or "", re.I,
    )




def query_role_selector(row: dict) -> tuple[str, str]:
    """Replace only the locked target occurrence; preserve the query as a role graph."""
    query = re.sub(r"\s+", " ", row["query"]).strip()
    target_name = row["target"]["target_name"]
    target_match = None
    source_span = ""
    for field in ("target_span", "evidence_span"):
        span = str(row["target"].get(field, "")).strip()
        anchor = phrase_match(query, span)
        if not anchor:
            continue
        local = phrase_match(anchor.group(0), target_name)
        if local:
            target_match = (
                anchor.start() + local.start(), anchor.start() + local.end(),
            )
            source_span = local.group(0)
            break
    if target_match is None:
        match = phrase_match(query, target_name)
        if match:
            target_match = (match.start(), match.end())
            source_span = match.group(0)
    if target_match is not None:
        start, end = target_match
        article = re.search(r"\b(?:a|an|the)\s*$", query[:start], re.I)
        if article:
            start = article.start()
        role_query = re.sub(
            r"\s+", " ", query[:start] + " TARGET " + query[end:]
        ).strip()
        return role_query, source_span

    owner_proxy = str(row["target"].get("owner_proxy_span", "")).strip()
    if owner_proxy:
        return (
            f'whole TARGET owning the cue in "{query}"; '
            f'owner-proxy "{owner_proxy}" is cue-only',
            owner_proxy,
        )
    return (
        f'whole TARGET identified by the part or context cue "{query}"; '
        "the cue phrase itself is cue-only",
        query,
    )


def short_v2_payloads(row: dict) -> list[dict]:
    """Build one query-native hypothesis independent of frozen Full wording."""
    target_name = row["target"]["target_name"]
    query_selector, source_span = query_role_selector(row)
    return [{
        "object_name": target_name,
        "mode": "query_role_target",
        "description": (
            f"The {target_name}; ROLE QUERY: {query_selector}; localize the "
            f"{row['target'].get('target_scope', 'whole_object')} TARGET only; "
            "all remaining object mentions are cues or references, NEVER TARGET"
        ),
        "source_span": source_span,
        "selector": query_selector,
        "source": "original_query_role_placeholder",
    }]


def gate_short_v2(row: dict, raw: str) -> tuple[str | None, dict]:
    parsed = parse_response(raw)
    description, problems = common_problems(row, parsed, normalize_prefix=False)
    expected_by_mode = {item["mode"]: item for item in short_v2_payloads(row)}
    expected = expected_by_mode.get(str(parsed.get("mode", "")))
    if expected is None:
        problems.append("invalid_cue_core_mode")
    else:
        for field in (
            "object_name", "description", "source_span", "selector", "source",
        ):
            if parsed.get(field) != expected[field]:
                problems.append(field + "_shift")
    full = row["descriptions"]["v4_full"]
    if description:
        selector = str(parsed.get("selector", ""))
        semantic_description = f"The {row['target']['target_name']}; {selector}"
        for cue in mandatory_cues(row):
            cue_content = content_tokens(cue)
            if cue_content:
                if not cue_content <= content_tokens(semantic_description):
                    problems.append("mandatory_cue_lost:" + normalized(cue))
            elif not contains_phrase(semantic_description, cue):
                problems.append("mandatory_cue_lost:" + normalized(cue))
        problems.extend(semantic_violations(row, semantic_description))
        problems.extend(short_role_problems(row, semantic_description))
        if normalized(description) == normalized(full):
            problems.append("duplicate_full")
    problems = sorted(set(problems))
    return (None if problems else description), {
        "accepted": not problems,
        "problems": problems,
        "mode": parsed.get("mode"),
        "object_name": parsed.get("object_name"),
        "source_span": parsed.get("source_span"),
        "selector": parsed.get("selector"),
        "source": parsed.get("source"),
        "full_tokens": len(words(full)),
        "short_tokens": len(words(description)),
        "must_be_token_shorter": False,
        "prefix_repaired": False,
    }


def choose_short_v2(row: dict):
    attempts = []
    for payload in short_v2_payloads(row):
        raw = json.dumps(payload, ensure_ascii=False)
        description, gate = gate_short_v2(row, raw)
        attempts.append({
            "mode": payload["mode"], "description": payload["description"],
            "problems": gate["problems"],
        })
        if description:
            gate["attempts"] = attempts
            return raw, description, gate
    raw = json.dumps({
        "object_name": row["target"]["target_name"], "mode": "abstained",
        "description": "", "source_span": "", "selector": "",
        "source": "original_query_role_placeholder",
    }, ensure_ascii=False)
    return raw, None, {
        "accepted": False,
        "problems": ["no_safe_cue_core"],
        "mode": "abstained",
        "object_name": row["target"]["target_name"],
        "source_span": "",
        "selector": "",
        "source": "original_query_role_placeholder",
        "full_tokens": len(words(row["descriptions"]["v4_full"])),
        "short_tokens": 0,
        "must_be_token_shorter": False,
        "prefix_repaired": False,
        "attempts": attempts,
    }


def confirmed_reference_note(reference: dict) -> dict | None:
    if reference.get("status") != "confirmed":
        return None
    return {
        key: reference.get(key)
        for key in (
            "candidate_id", "reference_name", "sam_prompt", "relation",
            "relation_direction", "class_relation", "mask_bbox",
            "same_category", "confirmation_source",
        )
    }


def safe_role_phrase(row: dict, value: str) -> str:
    phrase = str(value or "").strip()
    supported = content_tokens(f"{row['query']} {row['descriptions']['v4_full']}")
    return phrase if phrase and content_tokens(phrase) <= supported else ""


def long_role_payload(row: dict, reference: dict) -> dict:
    target = row["target"]
    full = row["descriptions"]["v4_full"].strip()
    prefix = re.match(
        r"^\s*the\s+" + r"\s+".join(
            re.escape(token) for token in words(target["target_name"])
        ) + r"\b",
        full,
        re.I,
    )
    selector_text = (
        full[prefix.end():].strip(" .;,:\t\n") if prefix else full.strip(" .;,:\t\n")
    )
    evidence = []
    for field in ("evidence_span", "target_span", "owner_proxy_span", "visual_anchor"):
        phrase = safe_role_phrase(row, target.get(field, ""))
        if phrase and normalized(phrase) != normalized(target["target_name"]):
            if normalized(phrase) not in {normalized(item) for item in evidence}:
                evidence.append(phrase)
    cues = list(dict.fromkeys(cue for cue in mandatory_cues(row) if normalized(cue)))
    reference_span = safe_role_phrase(row, target.get("reference_span", ""))
    confirmed = confirmed_reference_note(reference)
    mode = "confirmed_reference_negative" if confirmed else "role_decomposition"
    clauses = [
        f"The {target['target_name']} is the TARGET INSTANCE",
        f"TARGET SCOPE is {str(target['target_scope']).replace('_', ' ')}",
    ]
    if selector_text:
        clauses.append("FROZEN SELECTOR TEXT is " + selector_text)
    if evidence:
        clauses.append("TARGET EVIDENCE is " + " | ".join(evidence))
    if cues:
        clauses.append(
            "REQUIRED CUES with original target-reference direction are "
            + " | ".join(cues)
        )
    if confirmed:
        clauses.append(
            f"CONFIRMED REFERENCE {confirmed['candidate_id']} "
            f"({confirmed['reference_name']}) is REFERENCE ONLY and NEVER TARGET"
        )
    elif reference_span:
        clauses.append(
            f"REFERENCE-ONLY PHRASE is {reference_span}; only its subordinate "
            "instance in the directed relation is NEVER TARGET"
        )
    description = "; ".join(clauses)
    return {
        "object_name": target["target_name"],
        "mode": mode,
        "description": description,
        "target_scope": target["target_scope"],
        "selector_text": selector_text,
        "target_evidence": evidence,
        "required_cues": cues,
        "reference_span": reference_span,
        "reference_candidate_id": confirmed.get("candidate_id") if confirmed else "none",
        "reference_name": confirmed.get("reference_name") if confirmed else "",
        "novel_visual_cue": "none",
    }


def deterministic_long_raw(row: dict, reference: dict) -> str:
    return json.dumps(long_role_payload(row, reference), ensure_ascii=False)


def gate_long_role(row: dict, raw: str, reference: dict) -> tuple[str | None, dict]:
    parsed = parse_response(raw)
    expected = long_role_payload(row, reference)
    description = str(parsed.get("description", "")).strip()
    full = row["descriptions"]["v4_full"]
    problems = []
    if normalized(str(parsed.get("object_name", ""))) != normalized(expected["object_name"]):
        problems.append("object_name_shift")
    if parsed.get("mode") != expected["mode"]:
        problems.append("mode_shift")
    if description != expected["description"]:
        problems.append("role_payload_changed")
    if not exact_target_prefix(description, expected["object_name"]):
        problems.append("object_prefix_shift")
    if not content_tokens(full) <= content_tokens(description):
        problems.append("full_content_lost")
    if expected["selector_text"] and not contains_phrase(
        description, expected["selector_text"]
    ):
        problems.append("selector_text_lost")
    if len(words(description)) <= len(words(full)):
        problems.append("not_longer")
    for field in (
        "target_scope", "selector_text", "target_evidence", "required_cues", "reference_span",
        "reference_candidate_id", "reference_name", "novel_visual_cue",
    ):
        if parsed.get(field) != expected[field]:
            problems.append(field + "_shift")
    problems = sorted(set(problems))
    return (None if problems else description), {
        "accepted": not problems,
        "problems": problems,
        "mode": expected["mode"],
        "object_name": parsed.get("object_name"),
        "full_tokens": len(words(full)),
        "long_tokens": len(words(description)),
        "target_evidence": expected["target_evidence"],
        "selector_text": expected["selector_text"],
        "required_cues": expected["required_cues"],
        "reference_span": expected["reference_span"],
        "reference_candidate_id": expected["reference_candidate_id"],
        "reference_name": expected["reference_name"],
        "novel_visual_cue": "none",
        "support_level": (
            "confirmed_reference" if expected["mode"] == "confirmed_reference_negative"
            else "frozen_role_contract"
        ),
    }


def reference_shift_check(candidate_bbox: list[float] | None, reference: dict):
    overlaps = {}
    if candidate_bbox is not None and reference.get("status") == "confirmed":
        for name, box in (
            ("qwen_reference", reference.get("reference_bbox")),
            ("frozen_reference", reference.get("frozen_reference_bbox")),
            ("sam_reference", reference.get("mask_bbox")),
        ):
            if box is not None:
                overlaps[name] = bbox_iou(candidate_bbox, box)
    relation = normalized(str(reference.get("relation", "")))
    containment = relation in {
        "in", "inside", "on", "with", "holding", "carrying", "wearing",
        "riding", "has", "under", "over",
    }
    maximum_overlap = max(overlaps.values(), default=0.0)
    blocked = bool(overlaps) and (
        maximum_overlap >= REFERENCE_EXACT_COPY_BLOCK_IOU
        or (
            reference.get("same_category", False)
            and maximum_overlap >= REFERENCE_SHIFT_BLOCK_IOU
        )
        or (
            not reference.get("same_category", False)
            and not containment
            and (
                overlaps.get("qwen_reference", 0.0) >= 0.95
                or overlaps.get("frozen_reference", 0.0) >= 0.90
                or overlaps.get("sam_reference", 0.0) >= 0.90
            )
        )
    )
    return blocked, overlaps


def selected_reference_objects(row: dict) -> list[dict]:
    rounds = row.get("identity_rounds") or []
    if not rounds:
        return []
    return (rounds[0].get("query_plan") or {}).get("reference_objects") or []


def category_head(text: str) -> str:
    modifiers = (
        GRAMMAR_WORDS | set(v4.COLOR_WORDS) | set(v4.APPEARANCE_WORDS)
        | set(v4.ACTION_WORDS) | {"fully", "visible", "another", "other"}
    )
    for token in reversed(words(text)):
        if token not in modifiers and token not in {"his", "her", "its", "their", "one", "ones", "it"}:
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            return token
    return ""


def contains_phrase(text: str, phrase: str) -> bool:
    haystack, needle = words(text), words(phrase)
    return bool(needle) and any(
        haystack[index:index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def phrase_occurrence_count(text: str, phrase: str) -> int:
    haystack, needle = words(text), words(phrase)
    if not needle:
        return 0
    return sum(
        haystack[index:index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def target_owned_reference(row: dict, candidate: dict) -> bool:
    candidate_words = set(words(str(candidate.get("query_phrase", ""))))
    relation = normalized(str(candidate.get("relation", "")))
    return bool(candidate_words & TARGET_OWNED_WORDS) and (
        bool(candidate_words & {"his", "her", "its", "their"})
        or relation in {"with", "on", "in", "wearing", "has"}
    )


def safe_frozen_references(row: dict) -> tuple[list[dict], list[str]]:
    selected = selected_reference_objects(row)
    if len(selected) != 1:
        return [], [f"selected_reference_count:{len(selected)}"]
    selected_item = selected[0]
    candidate_id = str(selected_item.get("candidate_id", "")).strip().upper()
    candidates = {
        str(item.get("id", "")).strip().upper(): item
        for item in row["static_query_plan"].get("reference_candidates") or []
        if item.get("id")
    }
    evidence_items = [
        item for item in row.get("reference_evidence") or []
        if str(item.get("candidate_id", "")).strip().upper() == candidate_id
    ]
    problems = []
    candidate = candidates.get(candidate_id)
    if not candidate:
        problems.append("missing_static_candidate")
    if len(evidence_items) != 1:
        problems.append(f"reference_evidence_count:{len(evidence_items)}")
    evidence = evidence_items[0] if len(evidence_items) == 1 else {}
    for field in ("name", "sam_prompt", "relation"):
        if normalized(str(selected_item.get(field, ""))) != normalized(str(evidence.get(field, ""))):
            problems.append(f"selected_evidence_mismatch:{field}")
    if candidate and normalized(str(candidate.get("relation", ""))) != normalized(str(evidence.get("relation", ""))):
        problems.append("candidate_relation_mismatch")
    if evidence.get("status") != "shown":
        problems.append("frozen_reference_not_shown")

    width, height = row["image_size"]
    frozen_bboxes = []
    for box in evidence.get("bboxes") or []:
        parsed_box = v4_eval.parse_bbox_response(json.dumps(box), width, height)
        if parsed_box is not None:
            frozen_bboxes.append(parsed_box)
    if not frozen_bboxes:
        problems.append("missing_frozen_bbox")
    name = str(evidence.get("name", "")).strip()
    sam_prompt = str(evidence.get("sam_prompt", "")).strip()
    candidate_phrase = str((candidate or {}).get("query_phrase", ""))
    candidate_content = content_tokens(candidate_phrase)
    if not content_tokens(name) or not content_tokens(name) <= candidate_content:
        problems.append("reference_name_not_candidate_bound")
    if not content_tokens(sam_prompt) or not content_tokens(sam_prompt) <= candidate_content:
        problems.append("sam_prompt_not_candidate_bound")
    if NEGATION_RE.search(sam_prompt) or set(words(sam_prompt)) & {"any", "anything", "it", "one", "ones"}:
        problems.append("non_specific_sam_prompt")
    relation = str(evidence.get("relation", "")).strip()
    if relation and contains_phrase(sam_prompt, relation):
        problems.append("relation_contaminated_sam_prompt")
    if candidate and target_owned_reference(row, candidate):
        problems.append("target_owned_reference")
    if problems:
        return [], sorted(set(problems))
    return [{
        "candidate_id": candidate_id,
        "reference_name": name,
        "sam_prompt": sam_prompt,
        "relation": relation,
        "frozen_bboxes": frozen_bboxes,
        "same_category": bool(
            category_head(name) and category_head(name) == category_head(row["target"]["target_name"])
        ),
    }], []


def unique_box_match(anchor: list[float], boxes: list[list[float]], minimum: float):
    ranked = sorted(
        ((bbox_iou(anchor, box), index) for index, box in enumerate(boxes)),
        reverse=True,
    )
    if not ranked:
        return None, None, None, "no_boxes"
    best_iou, best_index = ranked[0]
    second_iou = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_iou < minimum:
        return None, best_iou, second_iou, "weak_match"
    if len(ranked) > 1 and (
        best_iou - second_iou < REFERENCE_UNIQUENESS_MARGIN
        or second_iou > best_iou * REFERENCE_UNIQUENESS_RATIO
    ):
        return None, best_iou, second_iou, "ambiguous_match"
    return best_index, best_iou, second_iou, None


def parse_reference_response(row: dict, raw: str, safe_references: list[dict]) -> dict:
    parsed = parse_response(raw)
    target_name = row["target"]["target_name"]
    candidates = {item["candidate_id"]: item for item in safe_references}
    candidate_id = str(parsed.get("candidate_id", "")).strip().upper()
    problems = []
    if normalized(str(parsed.get("object_name", ""))) != normalized(target_name):
        problems.append("object_name_shift")
    if candidate_id == "NONE":
        return {
            "status": "abstained", "accepted": False, "problems": problems,
            "raw": raw, "candidate_id": "none",
        }
    candidate = candidates.get(candidate_id)
    if candidate is None:
        problems.append("invalid_candidate_id")
    width, height = row["image_size"]
    qwen_bbox = v4_eval.parse_bbox_response(json.dumps(parsed.get("reference_bbox")), width, height)
    if qwen_bbox is None:
        problems.append("invalid_reference_bbox")
    match_index = best_iou = second_iou = None
    match_error = None
    if candidate and qwen_bbox is not None:
        match_index, best_iou, second_iou, match_error = unique_box_match(
            qwen_bbox, candidate["frozen_bboxes"], REFERENCE_QWEN_FROZEN_MIN_IOU
        )
        if match_error:
            problems.append("qwen_frozen_" + match_error)
    problems = sorted(set(problems))
    frozen_bbox = candidate["frozen_bboxes"][match_index] if candidate and match_index is not None else None
    return {
        "status": "grounded" if not problems else "rejected",
        "accepted": not problems,
        "problems": problems,
        "raw": raw,
        "candidate_id": candidate_id,
        "reference_name": candidate.get("reference_name") if candidate else None,
        "sam_prompt": candidate.get("sam_prompt") if candidate else None,
        "relation": candidate.get("relation") if candidate else None,
        "same_category": candidate.get("same_category", False) if candidate else False,
        "reference_bbox": qwen_bbox,
        "reference_bbox_source": "qwen",
        "frozen_reference_bbox": frozen_bbox,
        "qwen_frozen_iou": best_iou,
        "qwen_frozen_second_iou": second_iou,
    }


def build_reference_assist(qwen, get_sam3, image, row: dict, confidence: float, work: Path):
    if not row["static_query_plan"].get("reference_candidates"):
        return {"status": "no_candidates", "accepted": False, "problems": []}, None
    safe_references, safety_problems = safe_frozen_references(row)
    if not safe_references:
        return {
            "status": "no_safe_frozen_reference", "accepted": False,
            "problems": safety_problems,
        }, None
    public_references = [{
        key: item[key]
        for key in (
            "candidate_id", "reference_name", "sam_prompt", "relation",
            "same_category",
        )
    } for item in safe_references]
    try:
        raw = qwen.generate(
            reference_prompt(row, public_references), sys_prompt=SYSTEM_REFERENCE
        )
    except Exception as exc:
        return {
            "status": "qwen_error", "accepted": False,
            "problems": ["reference_qwen_error"], "qwen_error": str(exc),
        }, None
    reference = parse_reference_response(row, raw, safe_references)
    if not reference["accepted"]:
        return reference, None
    try:
        sam3 = get_sam3()
        masks = v4_eval._strong_masks(sam3.predict_text(reference["sam_prompt"]), confidence)
    except Exception as exc:
        reference.update({"status": "sam_error", "accepted": False, "sam_error": str(exc)})
        return reference, None
    mask_items, mask_boxes = [], []
    for item in masks:
        box = v4_eval._mask_bbox(item["mask"])
        if box is not None:
            mask_items.append(item)
            mask_boxes.append(box)
    best_index, best_iou, second_iou, match_error = unique_box_match(
        reference["frozen_reference_bbox"], mask_boxes, REFERENCE_SAM_FROZEN_MIN_IOU
    )
    if match_error:
        reference.update({
            "status": "sam_" + match_error, "accepted": False,
            "sam_mask_count": len(mask_boxes), "frozen_sam_iou": best_iou,
            "frozen_sam_second_iou": second_iou,
        })
        return reference, None
    mask_bbox, best = mask_boxes[best_index], mask_items[best_index]
    qwen_sam_iou = bbox_iou(reference["reference_bbox"], mask_bbox)
    if qwen_sam_iou < REFERENCE_QWEN_SAM_MIN_IOU:
        reference.update({
            "status": "qwen_sam_disagreement", "accepted": False,
            "sam_mask_count": len(mask_boxes), "frozen_sam_iou": best_iou,
            "frozen_sam_second_iou": second_iou, "qwen_sam_iou": qwen_sam_iou,
        })
        return reference, None
    evidence_path = work / "confirmed_reference.jpg"
    save_labeled_mask_overlay(
        image, best["mask"],
        f"CONFIRMED REFERENCE {reference['candidate_id']}: "
        f"{reference['reference_name']} | NEVER TARGET",
        evidence_path, (0, 210, 70),
    )
    reference.update({
        "status": "confirmed", "accepted": True,
        "source": "frozen_v4", "confirmation_source": "frozen_qwen_sam",
        "sam_mask_count": len(mask_boxes), "frozen_sam_iou": best_iou,
        "frozen_sam_second_iou": second_iou, "qwen_sam_iou": qwen_sam_iou,
        "mask_bbox": mask_bbox, "mask_confidence": float(best.get("conf", 0.0)),
    })
    return reference, str(evidence_path)


def query_reference_prompt(row: dict) -> str:
    target = row["target"]
    role_hint = {
        key: target.get(key)
        for key in (
            "target_name", "target_scope", "target_span", "reference_span",
            "owner_proxy_span", "role",
        )
    }
    return f"""The original image is loaded. This call only proposes one
query-native NEGATIVE reference; it never selects or renames the target.
ORIGINAL QUERY: {row['query']}
LOCKED TARGET ROLE: {json.dumps(role_hint, ensure_ascii=False)}
IMAGE SIZE: width={row['image_size'][0]}, height={row['image_size'][1]}

Find at most one physical REFERENCE instance whose location or identity helps
distinguish the locked target. The reference is a NEGATIVE instance and is
NEVER TARGET. Resolve direction literally: in "the elephant the baby points
towards", baby is reference and the pointed-to elephant is target; in "target
closest to crowd", crowd is reference. A body part, clothing item, or held
object owned by the target is only a cue unless the query uses a separate
object as the relation endpoint. Prefer an explicit lamp, fence, crowd, strap,
cat, car, or separately modified same-class object. Do not use vague regions
such as edge, side, corner, picture, row, left, or right as an object reference.

reference_phrase and relation_phrase must each copy one contiguous phrase from
ORIGINAL QUERY. reference_name and sam_prompt must be short intrinsic noun
phrases derived from reference_phrase: no relation or image position. For "the
baby" referring to a baby of the target class, sam_prompt may append the locked
target noun, e.g. "baby elephant". Return a tight bbox for the reference itself.
If there is no safe visually localizable reference, return status="none", put
"none" in every text field, and use an empty bbox.

Return one JSON object only with exactly these keys:
status, reference_phrase, reference_name, sam_prompt, relation_phrase,
relation_direction, class_relation, reference_bbox.
status is "reference" or "none". relation_direction is one of
"target_to_reference", "reference_to_target", "target_proxy_to_reference", or
"none". class_relation is "different", "same_or_overlapping", or "uncertain".
Use "reference_to_target" for a baby pointing toward the target;
"target_to_reference" for target next/closest to a lamp, fence, crowd, or cat;
and "target_proxy_to_reference" when a hand/part cue links the whole target to
a separate strap or object. Never return the locked target as the reference.
"""


def parse_query_reference_response(row: dict, raw: str) -> dict:
    parsed = parse_response(raw)
    target_name = row["target"]["target_name"]
    problems = []
    status = normalized(str(parsed.get("status", "")))
    if status == "none":
        return {
            "status": "query_abstained", "accepted": False,
            "problems": problems, "raw": raw, "candidate_id": "none",
            "object_name": target_name, "source": "query_native",
        }
    if status != "reference":
        problems.append("invalid_query_reference_status")
    phrase = re.sub(r"\s+", " ", str(parsed.get("reference_phrase", ""))).strip()
    reference_name = re.sub(
        r"\s+", " ", str(parsed.get("reference_name", ""))
    ).strip()
    sam_prompt = re.sub(r"\s+", " ", str(parsed.get("sam_prompt", ""))).strip()
    relation_phrase = re.sub(
        r"\s+", " ", str(parsed.get("relation_phrase", ""))
    ).strip()
    relation_direction = str(parsed.get("relation_direction", "")).strip().lower()
    class_relation = str(parsed.get("class_relation", "")).strip().lower()
    if phrase_occurrence_count(row["query"], phrase) != 1:
        problems.append("reference_phrase_not_query_bound")
    if phrase_occurrence_count(row["query"], relation_phrase) != 1:
        problems.append("relation_phrase_not_query_bound")
    if relation_direction not in {
        "target_to_reference", "reference_to_target",
        "target_proxy_to_reference",
    }:
        problems.append("invalid_relation_direction")
    if class_relation not in {"different", "same_or_overlapping", "uncertain"}:
        problems.append("invalid_class_relation")
    phrase_content = content_tokens(phrase)
    name_content = content_tokens(reference_name)
    sam_content = content_tokens(sam_prompt)
    proxy_reference = bool(
        set(words(phrase)) & QUERY_REFERENCE_SAME_CLASS_PROXIES
    )
    allowed_sam_content = set(phrase_content)
    if proxy_reference:
        allowed_sam_content |= content_tokens(target_name)
    if not name_content or not name_content <= phrase_content:
        problems.append("reference_name_not_query_bound")
    if not sam_content or not sam_content <= allowed_sam_content:
        problems.append("sam_prompt_not_query_bound")
    if not 1 <= len(words(sam_prompt)) <= 6:
        problems.append("unsafe_sam_prompt_length")
    reference_head = category_head(sam_prompt)
    if not reference_head or reference_head in QUERY_REFERENCE_INVALID_HEADS:
        problems.append("non_object_reference")
    lexical_same_category = bool(
        category_head(target_name)
        and (
            reference_head == category_head(target_name)
            or proxy_reference
        )
    )
    if class_relation == "different" and lexical_same_category:
        problems.append("class_relation_contradiction")
    same_category = lexical_same_category or class_relation in {
        "same_or_overlapping", "uncertain",
    }
    target_anchors = [
        str(row["target"].get(field, "")).strip()
        for field in ("target_span", "evidence_span")
        if str(row["target"].get(field, "")).strip()
    ]
    if any(normalized(phrase) == normalized(anchor) for anchor in target_anchors):
        problems.append("reference_phrase_is_target_anchor")
    if same_category and any(contains_phrase(anchor, phrase) for anchor in target_anchors):
        problems.append("same_class_reference_within_target_anchor")
    if normalized(sam_prompt) == normalized(target_name) or normalized(
        reference_name
    ) == normalized(target_name):
        problems.append("unqualified_same_class_reference")
    reference_candidate = {
        "query_phrase": phrase, "relation": relation_phrase,
    }
    owner_proxy = str(row["target"].get("owner_proxy_span", ""))
    if target_owned_reference(row, reference_candidate) or (
        owner_proxy and content_tokens(phrase) <= content_tokens(owner_proxy)
    ):
        problems.append("target_owned_part_as_reference")
    if len(words(relation_phrase)) > 8:
        problems.append("unsafe_relation_length")
    width, height = row["image_size"]
    qwen_bbox = v4_eval.parse_bbox_response(
        json.dumps(parsed.get("reference_bbox")), width, height
    )
    if qwen_bbox is None:
        problems.append("invalid_reference_bbox")
    elif (
        (qwen_bbox[2] - qwen_bbox[0]) * (qwen_bbox[3] - qwen_bbox[1])
        >= 0.90 * width * height
    ):
        problems.append("reference_bbox_near_full_image")
    problems = sorted(set(problems))
    return {
        "status": "query_grounded" if not problems else "query_rejected",
        "accepted": not problems,
        "problems": problems,
        "raw": raw,
        "candidate_id": "QREF1",
        "object_name": target_name,
        "reference_phrase": phrase,
        "reference_name": reference_name,
        "sam_prompt": sam_prompt,
        "relation": relation_phrase,
        "relation_direction": relation_direction,
        "class_relation": class_relation,
        "same_category": same_category,
        "reference_bbox": qwen_bbox,
        "reference_bbox_source": "query_qwen",
        "source": "query_native",
    }


def build_query_reference_assist(qwen, get_sam3, image, row: dict, confidence: float,
                                 work: Path):
    try:
        raw = qwen.generate(
            query_reference_prompt(row), sys_prompt=SYSTEM_QUERY_REFERENCE
        )
    except Exception as exc:
        return {
            "status": "query_qwen_error", "accepted": False,
            "problems": ["query_reference_qwen_error"],
            "qwen_error": str(exc), "source": "query_native",
        }, None
    reference = parse_query_reference_response(row, raw)
    if not reference["accepted"]:
        return reference, None

    mask_items, mask_boxes, sam_error = [], [], None
    try:
        sam3 = get_sam3()
        masks = v4_eval._strong_masks(
            sam3.predict_text(reference["sam_prompt"]), confidence
        )
        for item in masks:
            box = item.get("bbox") or v4_eval._mask_bbox(item["mask"])
            if box is not None:
                mask_items.append(item)
                mask_boxes.append(box)
    except Exception as exc:
        sam_error = str(exc)

    best_index = best_iou = second_iou = None
    match_error = "sam_error" if sam_error else "no_boxes"
    if mask_boxes:
        best_index, best_iou, second_iou, match_error = unique_box_match(
            reference["reference_bbox"], mask_boxes,
            (
                QUERY_REFERENCE_SAME_CLASS_MIN_IOU
                if reference["same_category"] else QUERY_REFERENCE_SAM_MIN_IOU
            ),
        )
    if best_index is not None:
        best = mask_items[best_index]
        mask_bbox = mask_boxes[best_index]
        mask = best["mask"]
        source = "query_qwen_sam"
    elif reference["same_category"]:
        reference.update({
            "status": "query_sam_" + str(match_error), "accepted": False,
            "problems": ["same_class_reference_not_sam_confirmed"],
            "sam_error": sam_error, "sam_mask_count": len(mask_boxes),
            "qwen_sam_iou": best_iou,
            "qwen_sam_second_iou": second_iou,
        })
        return reference, None
    else:
        width, height = row["image_size"]
        mask = v4_eval.np.zeros((height, width), dtype=bool)
        x1, y1, x2, y2 = reference["reference_bbox"]
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(width, int(x2 + 0.999)), min(height, int(y2 + 0.999))
        mask[top:bottom, left:right] = True
        mask_bbox = reference["reference_bbox"]
        source = "query_qwen_bbox"

    evidence_path = work / "query_reference_never_target.jpg"
    save_labeled_mask_overlay(
        image, mask,
        f"QUERY REFERENCE: {reference['reference_name']} | NEVER TARGET",
        evidence_path, (0, 210, 70),
    )
    reference.update({
        "status": "confirmed", "accepted": True,
        "confirmation_source": source,
        "mask_bbox": mask_bbox,
        "sam_mask_count": len(mask_boxes),
        "qwen_sam_iou": best_iou,
        "qwen_sam_second_iou": second_iou,
        "sam_error": sam_error,
    })
    return reference, str(evidence_path)


def build_target_candidate_evidence(get_sam3, image, row: dict, reference: dict,
                                    confidence: float, work: Path):
    target_name = row["target"]["target_name"]
    if row["target"].get("target_scope") != "whole_object":
        return {
            "status": "unsupported_scope", "accepted": False,
            "problems": ["target_scope_not_whole_object"], "candidates": [],
        }, [], {}
    if NEGATION_RE.fullmatch(str(target_name).strip()):
        return {
            "status": "invalid_target_name", "accepted": False,
            "problems": ["negation_as_target_name"], "candidates": [],
        }, [], {}
    try:
        sam3 = get_sam3()
        masks = v4_eval._strong_masks(sam3.predict_text(target_name), confidence)
        masks = v4_eval._complete_ordinal_masks(masks)
    except Exception as exc:
        return {
            "status": "sam_error", "accepted": False,
            "problems": ["target_sam_error"], "sam_error": str(exc),
            "candidates": [],
        }, [], {}

    items = []
    reference_bbox = (
        reference.get("mask_bbox")
        if reference.get("status") == "confirmed" and reference.get("same_category")
        else None
    )
    for item in masks:
        box = item.get("bbox") or v4_eval._mask_bbox(item["mask"])
        if box is None:
            continue
        items.append({**item, "bbox": box})
    if not items:
        return {
            "status": "no_target_candidates", "accepted": False,
            "problems": ["no_strong_target_mask"], "candidates": [],
        }, [], {}
    candidate_count_total = len(items)
    truncated = candidate_count_total > V2_MAX_TARGET_CANDIDATES
    if truncated:
        items = sorted(items, key=lambda item: (
            -float(item.get("conf", 0.0)),
            -((item["bbox"][2] - item["bbox"][0])
              * (item["bbox"][3] - item["bbox"][1])),
        ))[:V2_MAX_TARGET_CANDIDATES]
    items.sort(key=lambda item: (
        round(item["bbox"][1], 3), round(item["bbox"][0], 3),
        -round((item["bbox"][2] - item["bbox"][0])
               * (item["bbox"][3] - item["bbox"][1]), 3),
    ))

    palette = [
        (255, 92, 92), (90, 210, 255), (255, 205, 70), (120, 240, 130),
        (210, 130, 255), (255, 145, 70), (80, 235, 220), (255, 110, 200),
        (180, 230, 80), (110, 155, 255), (245, 185, 120), (170, 170, 255),
    ]
    public, render_items, by_id = [], [], {}
    reference_candidate_ious = {}
    never_target_candidate_ids = []
    for index, item in enumerate(items, 1):
        candidate_id = f"T{index}"
        reference_iou = (
            bbox_iou(item["bbox"], reference_bbox)
            if reference_bbox is not None else 0.0
        )
        reference_candidate_ious[candidate_id] = reference_iou
        if reference_iou >= 0.70:
            never_target_candidate_ids.append(candidate_id)
        public.append({
            "id": candidate_id, "bbox": item["bbox"],
            "confidence": float(item.get("conf", 0.0)),
            "confirmed_reference_iou": reference_iou,
            "never_target": reference_iou >= 0.70,
            "image_panel": (
                f"Image 2 global map plus Image 3 context crop labeled {candidate_id}"
            ),
        })
        by_id[candidate_id] = item
        render_items.append({
            "id": candidate_id, "bbox": item["bbox"], "mask": item["mask"],
        })
    overview_path = work / "target_candidate_global.jpg"
    montage_path = work / "target_candidate_panel.jpg"
    save_target_candidate_views(
        image, render_items, target_name,
        overview_path, montage_path, palette,
    )
    status = (
        "single_target_candidate" if len(items) == 1
        else "enumerated_truncated" if truncated else "enumerated"
    )
    warnings = []
    if truncated:
        warnings.append(f"candidate_pool_truncated:{candidate_count_total}->{len(items)}")
    if never_target_candidate_ids:
        warnings.append(
            "same_class_reference_marked_never_target:"
            + ",".join(never_target_candidate_ids)
        )
    return {
        "status": status, "accepted": True, "problems": [],
        "warnings": warnings, "sam_prompt": target_name,
        "target_scope": row["target"].get("target_scope"),
        "candidate_count_total": candidate_count_total,
        "candidate_count_used": len(items), "truncated": truncated,
        "same_class_reference_removed": 0,
        "same_class_reference_filter_reverted": False,
        "reference_candidate_ious": reference_candidate_ious,
        "never_target_candidate_ids": never_target_candidate_ids,
        "candidates": public,
    }, [str(overview_path), str(montage_path)], by_id


def target_selection_prompt(row: dict, evidence: dict, reference: dict,
                            reference_image_index: int | None) -> str:
    candidate_ids = [item["id"] for item in evidence["candidates"]]
    image_map = {
        item["id"]: (
            f"Image 2 same-coordinate global mark and Image 3 crop {item['id']}"
        )
        for item in evidence["candidates"]
    }
    candidate_bboxes = {
        item["id"]: item["bbox"] for item in evidence["candidates"]
    }
    reference_image = (
        f"Image {reference_image_index} green mask"
        if reference_image_index is not None else "none"
    )
    reference_candidates = row["static_query_plan"].get("reference_candidates") or []
    role_locks = [
        {
            "candidate_id": item.get("id"),
            "reference_phrase": item.get("query_phrase"),
            "directed_relation": item.get("relation"),
        }
        for item in reference_candidates
    ]
    query_reference_lock = (
        {
            "reference_phrase": reference.get("reference_phrase"),
            "reference_name": reference.get("reference_name"),
            "relation_phrase": reference.get("relation"),
            "relation_direction": reference.get("relation_direction"),
            "class_relation": reference.get("class_relation"),
            "bbox": reference.get("mask_bbox"),
            "never_target": True,
        }
        if reference.get("status") == "confirmed" else None
    )
    plan = row["static_query_plan"]
    safe_relation_reference = any(
        category_head(str(item.get("query_phrase", "")))
        and category_head(str(item.get("query_phrase", "")))
        not in QUERY_REFERENCE_INVALID_HEADS
        and not target_owned_reference(row, item)
        for item in reference_candidates
    )
    explicit_geometry = bool(
        plan.get("absolute_positions")
        or plan.get("ordinal") is not None
        or plan.get("ordinal_cues")
        or set(words(row["query"])) & {
            "leftmost", "rightmost", "topmost", "bottommost",
            "closest", "nearest", "farthest",
            "larger", "smaller", "bigger", "across", "opposite",
        }
    )
    relation_prompt_needed = bool(
        query_reference_lock or safe_relation_reference or explicit_geometry
    )
    extended_lock_lines = (
        "QUERY-NATIVE CONFIRMED REFERENCE LOCK: "
        f"{json.dumps(query_reference_lock, ensure_ascii=False)}\n"
        "TARGET IDS OVERLAPPING THE CONFIRMED SAME-CLASS REFERENCE "
        f"(NEVER TARGET): {json.dumps(evidence.get('never_target_candidate_ids') or [])}\n"
        if relation_prompt_needed else ""
    )
    selection_instructions = (
        """and scope without reversing target/reference roles. For leftmost/rightmost,
top/bottom, closest/farthest, or larger/smaller, compare the listed bboxes in
the original coordinate system (bbox centers, distances to the confirmed
reference bbox, and bbox extents respectively); do not infer order from crop
placement. For pointing or owner-part language, follow relation_direction
literally from reference/proxy to target. Select exactly one ID. Do not default
to T1, a larger containing mask, the reference, or the visually most salient
item. Any ID explicitly listed as overlapping the confirmed same-class
reference is forbidden. Do not invent or report a new appearance cue in this
step."""
        if relation_prompt_needed else
        """and scope without reversing target/reference roles. Select exactly one ID. Do
not default to T1, a larger containing mask, or the visually most salient item.
Do not invent or report a new appearance cue in this step."""
    )
    return f"""Image 1 is the original image. Image 2 is the authoritative global
candidate map in the exact same coordinate system: every SAM3 candidate has a
colored mask, thin bbox, and T-ID. Image 3 contains enlarged context crops for
appearance only; never infer global left/right/front/order from crop placement.
The separately listed green reference image, when present, is NEVER TARGET.
ORIGINAL QUERY: {row['query']}
LOCKED TARGET NAME: {row['target']['target_name']}
LOCKED TARGET SCOPE: {row['target'].get('target_scope')}
REFERENCE ROLE LOCKS: {json.dumps(role_locks, ensure_ascii=False)}
{extended_lock_lines}TARGET CANDIDATE IDS: {json.dumps(candidate_ids)}
TARGET CANDIDATE IMAGE MAP: {json.dumps(image_map)}
TARGET CANDIDATE SAM3 BBOXES: {json.dumps(candidate_bboxes)}
CONFIRMED REFERENCE IMAGE: {reference_image}

Use ONLY the ORIGINAL QUERY and locked target head for selection; the failed
Full hypothesis is deliberately withheld. First locate every query reference
and mark it mentally as NEVER TARGET. Then scan every T-ID on Image 2, applying
all target-owned attributes, directed relations, negation, cardinality, order,
{selection_instructions}

Return one JSON object only, with exactly these keys:
selected_id, checked_ids. checked_ids must copy every TARGET CANDIDATE ID once
in the listed order. selected_id must be one exact listed ID. Keep this compact;
return no explanation, object_name, cue, status list, or coverage boolean.
"""


def target_visual_cue_prompt(row: dict, selected_id: str) -> str:
    return f"""Image 1 is the original image. Image 2 is a clean original-color
crop of the already selected target: target pixels keep their natural colors
and surrounding pixels are dimmed. It contains no colored overlay, ID text, or
annotation. The candidate selection is frozen and MUST NOT be revised.
ORIGINAL QUERY: {row['query']}
FROZEN FULL DESCRIPTION: {row['descriptions']['v4_full']}
LOCKED TARGET NAME: {row['target']['target_name']}
FROZEN SELECTED TARGET ID: {selected_id}

Optionally report one short visible cue that belongs inside the frozen selected
target crop and is absent from both the query and Full. If alternatives exist,
the cue must be unique to the selected candidate. It may be color, pattern,
material, shape, size, clothing, body feature, or intrinsic state. It must NOT
be a relation, action, ordinal, absolute image position, held object,
surrounding object, or property of a reference. If no safe cue exists, use
cue="none", cue_type="none", and an empty cue_present_on_ids list.

Return one JSON object only with exactly these keys:
cue, cue_type, cue_owner_id, cue_present_on_ids, reference_owned.
cue_owner_id must be the frozen selected ID when cue is not none.
reference_owned is a JSON boolean. Never return or revise selected_id.
"""


def reference_content_tokens(row: dict, reference: dict) -> set[str]:
    values = [str(row["target"].get("reference_span", ""))]
    for item in row["static_query_plan"].get("reference_candidates") or []:
        values.append(str(item.get("query_phrase", "")))
    for item in frozen_reference_objects(row):
        values.extend(str(item.get(key, "")) for key in ("name", "sam_prompt"))
    values.extend(str(reference.get(key, "")) for key in ("reference_name", "sam_prompt"))
    return content_tokens(" ".join(values))


def gate_target_visual_cue(row: dict, raw: str, evidence: dict,
                           reference: dict) -> dict:
    parsed = parse_response(raw)
    expected_ids = [item["id"] for item in evidence.get("candidates") or []]
    selection_problems = []
    if normalized(str(parsed.get("object_name", ""))) != normalized(
        row["target"]["target_name"]
    ):
        selection_problems.append("object_name_shift")
    checks = parsed.get("candidate_checks")
    checks = checks if isinstance(checks, list) else []
    check_ids = [str(item.get("id", "")) for item in checks if isinstance(item, dict)]
    if len(check_ids) != len(set(check_ids)) or set(check_ids) != set(expected_ids):
        selection_problems.append("candidate_coverage_mismatch")
    status_by_id = {
        str(item.get("id", "")): str(item.get("status", "")).lower()
        for item in checks if isinstance(item, dict)
    }
    selected_id = str(parsed.get("selected_id", ""))
    matched_ids = [key for key, value in status_by_id.items() if value == "match"]
    if len(matched_ids) != 1 or matched_ids[0] != selected_id:
        selection_problems.append("non_unique_target_match")
    if selected_id not in expected_ids:
        selection_problems.append("invalid_selected_id")
    if selected_id in set(evidence.get("never_target_candidate_ids") or []):
        selection_problems.append("selected_confirmed_reference_never_target")
    if any(value not in {"match", "reject"} for value in status_by_id.values()):
        selection_problems.append("invalid_candidate_status")
    if parsed.get("coverage_complete") is not True:
        selection_problems.append("incomplete_candidate_coverage")

    cue_problems = []
    present_ids = parsed.get("cue_present_on_ids")
    present_ids = present_ids if isinstance(present_ids, list) else []
    cue = re.sub(r"\s+", " ", str(parsed.get("cue", ""))).strip()
    cue_type = str(parsed.get("cue_type", "")).strip().lower()
    cue_is_none = not cue or normalized(cue) in {"none", "unknown", "uncertain"}
    cue_content = content_tokens(cue)
    source_content = content_tokens(
        f"{row['query']} {row['descriptions']['v4_full']}"
    )
    novel_tokens = sorted(cue_content - source_content)
    if cue_is_none:
        cue_problems.append("no_visual_cue")
    else:
        if str(parsed.get("cue_owner_id", "")) != selected_id:
            cue_problems.append("cue_owner_shift")
        if len(present_ids) != 1 or present_ids[0] != selected_id:
            cue_problems.append("cue_not_unique_to_selected_target")
        if parsed.get("reference_owned") is not False:
            cue_problems.append("reference_owned_cue")
        if not 1 <= len(words(cue)) <= 8:
            cue_problems.append("unsafe_cue_length")
        if re.search(r"[^a-zA-Z0-9 '\-]", cue):
            cue_problems.append("unsafe_cue_characters")
        if cue_type not in V2_CUE_TYPES:
            cue_problems.append("unsafe_cue_type")
        if set(words(cue)) & V2_UNSAFE_CUE_WORDS or NEGATION_RE.search(cue):
            cue_problems.append("relational_or_negative_cue")
        if set(words(cue)) & V2_UI_CUE_WORDS:
            cue_problems.append("annotation_artifact_cue")
        if not cue_content or not novel_tokens:
            cue_problems.append("cue_not_novel")
        if set(novel_tokens) & reference_content_tokens(row, reference):
            cue_problems.append("cue_overlaps_reference_content")
    selection_problems = sorted(set(selection_problems))
    cue_problems = sorted(set(cue_problems))
    selection_accepted = not selection_problems
    cue_accepted = selection_accepted and not cue_problems
    selected = next(
        (item for item in evidence.get("candidates") or [] if item["id"] == selected_id),
        None,
    )
    return {
        "status": (
            "selected_with_cue" if cue_accepted else
            "selected_without_cue" if selection_accepted else "rejected_selection"
        ),
        "accepted": selection_accepted,
        "problems": selection_problems,
        "all_problems": sorted(set(selection_problems + cue_problems)),
        "raw": raw,
        "object_name": parsed.get("object_name"),
        "candidate_checks": checks,
        "selected_id": selected_id,
        "selected_mask_bbox": selected.get("bbox") if selected else None,
        "cue": cue,
        "cue_type": cue_type,
        "cue_owner_id": parsed.get("cue_owner_id"),
        "cue_present_on_ids": present_ids,
        "reference_owned": parsed.get("reference_owned"),
        "coverage_complete": parsed.get("coverage_complete"),
        "novel_tokens": novel_tokens,
        "selection": {
            "accepted": selection_accepted, "problems": selection_problems,
            "selected_id": selected_id,
        },
        "cue_gate": {
            "accepted": cue_accepted, "problems": cue_problems,
            "cue": cue, "cue_type": cue_type,
        },
        "cue_accepted": cue_accepted,
    }


def build_target_visual_cue(qwen, get_sam3, image, row: dict, reference: dict,
                            reference_path: str | None, confidence: float,
                            work: Path):
    evidence, candidate_paths, masks_by_id = build_target_candidate_evidence(
        get_sam3, image, row, reference, confidence, work
    )
    if (
        not evidence["accepted"]
        and reference.get("confirmation_source") == "query_qwen_bbox"
    ):
        reference.update({
            "status": "query_reference_unverified_without_target_pool",
            "accepted": False,
            "problems": sorted(set(
                reference.get("problems", [])
                + ["query_reference_unverified_without_target_pool"]
            )),
        })
        reference_path = None
    if not evidence["accepted"]:
        return {
            "status": "unavailable", "accepted": False,
            "problems": evidence["problems"], "raw": None,
            "target_candidate_evidence": evidence,
        }, masks_by_id, {}
    if reference.get("status") != "confirmed":
        reference_path = None
    elif not reference.get("same_category") and reference.get("mask_bbox"):
        reference_target_ious = {
            item["id"]: bbox_iou(
                reference["mask_bbox"], item["bbox"]
            )
            for item in evidence.get("candidates") or []
        }
        maximum_target_iou = max(reference_target_ious.values(), default=0.0)
        overlap_limit = (
            QUERY_REFERENCE_TARGET_OVERLAP_IOU
            if reference.get("confirmation_source") == "query_qwen_bbox"
            else REFERENCE_EXACT_COPY_BLOCK_IOU
        )
        reference["target_candidate_overlap_ious"] = reference_target_ious
        if maximum_target_iou >= overlap_limit:
            reference.update({
                "status": "query_reference_overlaps_target_candidate",
                "accepted": False,
                "problems": sorted(set(
                    reference.get("problems", [])
                    + ["query_reference_overlaps_target_candidate"]
                )),
            })
            reference_path = None
    image_paths = list(candidate_paths)
    if reference_path:
        image_paths.append(reference_path)
    expected_ids = [item["id"] for item in evidence.get("candidates") or []]
    never_target_ids = set(evidence.get("never_target_candidate_ids") or [])

    selection_attempts = []
    if len(expected_ids) == 1:
        selection_compact = {
            "selected_id": expected_ids[0], "checked_ids": expected_ids,
        }
        selection_raw = json.dumps(selection_compact, ensure_ascii=False)
        selection_attempts.append(selection_raw)
        selection_source = "sam3_single_candidate"
    else:
        selection_raw = None
        selection_compact = {}
        selection_prompt = target_selection_prompt(
            row, evidence, reference,
            2 + len(candidate_paths) if reference_path else None,
        )
        for attempt_index in range(2):
            retry_instruction = (
                "\nYour prior response was invalid or truncated. Return the two-key compact "
                "JSON now; checked_ids must exactly copy the full listed ID array, and "
                "selected_id must not be a NEVER TARGET reference-overlap ID."
                if attempt_index else ""
            )
            try:
                raw_attempt = qwen.generate(
                    selection_prompt + retry_instruction,
                    image_paths=image_paths, sys_prompt=SYSTEM_TARGET_SELECTION,
                )
            except Exception as exc:
                if not selection_attempts:
                    return {
                        "status": "qwen_error", "accepted": False,
                        "problems": ["target_selection_qwen_error"], "raw": None,
                        "qwen_error": str(exc), "target_candidate_evidence": evidence,
                    }, masks_by_id, {}
                break
            selection_attempts.append(raw_attempt)
            parsed_attempt = parse_response(raw_attempt)
            checked_ids = parsed_attempt.get("checked_ids")
            checked_ids = (
                [str(item) for item in checked_ids]
                if isinstance(checked_ids, list) else []
            )
            selection_compact = {
                "selected_id": str(parsed_attempt.get("selected_id", "")),
                "checked_ids": checked_ids,
            }
            selection_raw = raw_attempt
            if (
                selection_compact["selected_id"] in expected_ids
                and checked_ids == expected_ids
                and selection_compact["selected_id"] not in never_target_ids
            ):
                break
        if selection_raw is None:
            return {
                "status": "qwen_error", "accepted": False,
                "problems": ["target_selection_qwen_error"], "raw": None,
                "target_candidate_evidence": evidence,
            }, masks_by_id, {}
        selection_source = "qwen_query_global_map_selection"

    selected_id = str(selection_compact.get("selected_id", ""))
    checked_ids = selection_compact.get("checked_ids") or []
    coverage_complete = (
        checked_ids == expected_ids and len(checked_ids) == len(set(checked_ids))
    )
    selection_payload = {
        "object_name": row["target"]["target_name"],
        "candidate_checks": [
            {
                "id": candidate_id,
                "status": "match" if candidate_id == selected_id else "reject",
            }
            for candidate_id in expected_ids
        ],
        "selected_id": selected_id,
        "coverage_complete": coverage_complete,
    }

    combined = {
        **selection_payload,
        "cue": "none", "cue_type": "none",
        "cue_owner_id": selected_id,
        "cue_present_on_ids": [], "reference_owned": False,
    }
    selection_frozen_raw = json.dumps(combined, ensure_ascii=False)
    cue = gate_target_visual_cue(
        row, selection_frozen_raw, evidence, reference
    )
    cue["selection"].update({
        "source": selection_source,
        "checked_ids": checked_ids,
        "attempt_count": len(selection_attempts),
    })
    cue["selection_raw"] = selection_raw
    cue["selection_attempts_raw"] = selection_attempts
    cue["cue_raw"] = None
    cue["target_candidate_evidence"] = evidence
    if not cue["selection"]["accepted"]:
        return cue, masks_by_id, {}

    selected = masks_by_id.get(cue.get("selected_id"))
    target_assets = (
        save_selected_target_views(
            image, selected["mask"], selected["bbox"], work,
        )
        if selected else {}
    )
    try:
        cue_raw = qwen.generate(
            target_visual_cue_prompt(row, selected_id),
            image_paths=(
                [target_assets["clean_crop"]]
                if target_assets.get("clean_crop") else []
            ),
            sys_prompt=SYSTEM_VISUAL_CUE,
        )
        parsed_cue = parse_response(cue_raw)
        combined.update({
            key: parsed_cue.get(key)
            for key in (
                "cue", "cue_type", "cue_owner_id",
                "cue_present_on_ids", "reference_owned",
            )
        })
        final_raw = json.dumps(combined, ensure_ascii=False)
        cue = gate_target_visual_cue(row, final_raw, evidence, reference)
        cue["selection"].update({
            "source": selection_source,
            "checked_ids": checked_ids,
            "attempt_count": len(selection_attempts),
        })
        cue["selection_raw"] = selection_raw
        cue["selection_attempts_raw"] = selection_attempts
        cue["cue_raw"] = cue_raw
        cue["target_candidate_evidence"] = evidence
    except Exception as exc:
        cue["cue_gate"] = {
            "accepted": False, "problems": ["visual_cue_qwen_error"],
            "cue": "none", "cue_type": "none",
        }
        cue["cue_accepted"] = False
        cue["all_problems"] = sorted(set(
            cue.get("all_problems", []) + ["visual_cue_qwen_error"]
        ))
        cue["cue_qwen_error"] = str(exc)
    return cue, masks_by_id, target_assets


def gate_short_target_mask(short_bbox: list[float] | None, evidence: dict) -> dict:
    if short_bbox is None:
        return {
            "accepted": False, "status": "invalid_bbox",
            "problems": ["invalid_target_bbox"], "matched_id": None,
            "best_iou": None, "candidate_count": 0,
        }
    candidates = evidence.get("candidates") or []
    usable_status = evidence.get("status") in {
        "enumerated", "enumerated_truncated", "single_target_candidate",
    }
    if not usable_status or not candidates:
        return {
            "accepted": True, "status": "retained_unverified",
            "problems": [], "warnings": ["target_class_evidence_unavailable"],
            "matched_id": None,
            "best_iou": None, "candidate_count": len(candidates),
            "evidence_status": evidence.get("status"),
        }
    best = max(
        candidates,
        key=lambda item: bbox_iou(short_bbox, item["bbox"]),
    )
    best_iou = bbox_iou(short_bbox, best["bbox"])
    if evidence.get("status") == "enumerated_truncated" and best_iou < V2_TARGET_SAM_MIN_IOU:
        return {
            "accepted": True, "status": "retained_unverified_truncated",
            "problems": [],
            "warnings": ["truncated_target_evidence_cannot_prove_mask_mismatch"],
            "matched_id": best["id"], "best_iou": best_iou,
            "candidate_count": len(candidates),
            "evidence_status": evidence.get("status"),
        }
    accepted = best_iou >= V2_TARGET_SAM_MIN_IOU
    return {
        "accepted": accepted,
        "status": "confirmed" if accepted else "mask_mismatch",
        "problems": [] if accepted else ["target_class_mask_mismatch"],
        "warnings": [],
        "matched_id": best["id"], "best_iou": best_iou,
        "candidate_count": len(candidates),
        "evidence_status": evidence.get("status"),
    }


def long_v2_payload(row: dict, cue: dict) -> dict:
    target_name = row["target"]["target_name"]
    full = row["descriptions"]["v4_full"].strip().rstrip(".")
    if cue.get("cue_accepted"):
        description = (
            f"{full}; additionally, the same {target_name} has this visible "
            f"appearance: {cue['cue']}. This appearance belongs to the target, "
            "not to any referenced object."
        )
        novel_cue = cue["cue"]
        cue_type = cue["cue_type"]
        support = "sam3_target_mask_plus_qwen_selection_and_visual_cue"
    else:
        description = (
            f"{full}; visual hypothesis: the same {target_name} is the visually "
            "confirmed target instance, while every referenced object and all "
            "other same-class instances are excluded."
        )
        novel_cue = "none"
        cue_type = "none"
        support = "sam3_target_mask_plus_candidate_selection"
    return {
        "object_name": target_name,
        "mode": "grounded_target_visual",
        "description": description,
        "novel_visual_cue": novel_cue,
        "cue_type": cue_type,
        "cue_owner_id": cue["selected_id"],
        "cue_source": support,
    }


def gate_long_v2(row: dict, raw: str, cue: dict) -> tuple[str | None, dict]:
    parsed = parse_response(raw)
    expected = long_v2_payload(row, cue)
    description = str(parsed.get("description", "")).strip()
    full = row["descriptions"]["v4_full"]
    problems = []
    for field in (
        "object_name", "mode", "description", "novel_visual_cue", "cue_type",
        "cue_owner_id", "cue_source",
    ):
        if parsed.get(field) != expected[field]:
            problems.append(field + "_shift")
    if not exact_target_prefix(description, row["target"]["target_name"]):
        problems.append("object_prefix_shift")
    if not content_tokens(full) <= content_tokens(description):
        problems.append("full_content_lost")
    if len(words(description)) <= len(words(full)):
        problems.append("not_longer")
    if not cue.get("selection", {}).get("accepted", cue.get("accepted", False)):
        problems.append("target_candidate_selection_not_grounded")
    if cue.get("cue_accepted") and expected["novel_visual_cue"] == "none":
        problems.append("grounded_visual_cue_lost")
    problems = sorted(set(problems))
    return (None if problems else description), {
        "accepted": not problems,
        "problems": problems,
        "mode": expected["mode"],
        "object_name": parsed.get("object_name"),
        "full_tokens": len(words(full)),
        "long_tokens": len(words(description)),
        "novel_visual_cue": expected["novel_visual_cue"],
        "cue_type": expected["cue_type"],
        "cue_owner_id": expected["cue_owner_id"],
        "support_level": expected["cue_source"],
    }


def role_long_fallback(row: dict, reference: dict, reasons: list[str]):
    raw = deterministic_long_raw(row, reference)
    description, gate = gate_long_role(row, raw, reference)
    gate["base_mode"] = gate["mode"]
    gate["mode"] = "role_decomposition_fallback"
    gate["fallback_reason"] = sorted(set(reasons))
    return raw, description, gate


def verify_visual_target_bbox(candidate_bbox: list[float] | None, cue: dict) -> dict:
    candidates = cue.get("target_candidate_evidence", {}).get("candidates") or []
    if candidate_bbox is None:
        return {
            "accepted": False, "problems": ["invalid_target_bbox"],
            "selected_target_candidate_id": cue.get("selected_id"),
            "matched_target_candidate_id": None,
            "target_candidate_iou": None,
            "target_candidate_second_iou": None,
            "target_candidate_match_error": "invalid_bbox",
        }
    selected = next(
        (item for item in candidates if item["id"] == cue.get("selected_id")), None
    )
    if selected is None:
        return {
            "accepted": False, "problems": ["invalid_selected_target_candidate"],
            "selected_target_candidate_id": cue.get("selected_id"),
            "matched_target_candidate_id": None,
            "target_candidate_iou": None,
            "target_candidate_second_iou": None,
            "target_candidate_match_error": "invalid_selected_id",
        }
    selected_iou = bbox_iou(candidate_bbox, selected["bbox"])
    other_ious = [
        bbox_iou(candidate_bbox, item["bbox"])
        for item in candidates if item["id"] != selected["id"]
    ]
    best_other_iou = max(other_ious) if other_ious else 0.0
    problems = []
    if selected_iou < V2_TARGET_FINAL_MIN_IOU:
        problems.append("grounded_target_candidate_mismatch")
    if best_other_iou > selected_iou + V2_TARGET_FINAL_OTHER_MARGIN:
        problems.append("other_target_candidate_dominates")
    problems = sorted(set(problems))
    return {
        "accepted": not problems, "problems": problems,
        "selected_target_candidate_id": cue.get("selected_id"),
        "matched_target_candidate_id": selected["id"] if not problems else None,
        "target_candidate_iou": selected_iou,
        "target_candidate_second_iou": best_other_iou,
        "target_candidate_match_error": (
            None if not problems else
            "other_candidate_dominates"
            if "other_target_candidate_dominates" in problems
            else "weak_selected_match"
        ),
        "target_candidate_min_iou": V2_TARGET_FINAL_MIN_IOU,
        "target_candidate_other_margin": V2_TARGET_FINAL_OTHER_MARGIN,
    }


def detect_locked_bbox(qwen, row: dict, description: str, width: int, height: int,
                       variant: str, reference: dict | None = None,
                       evidence_path: str | None = None,
                       target_evidence_path: str | None = None,
                       selected_target_id: str | None = None,
                       selected_target_bbox: list[float] | None = None,
                       mask_retry: bool = False):
    target = gate_target(row)
    contract_fields = (
        ("target_name", "target_scope") if variant == "short" else
        (
            "target_name", "target_scope", "target_span", "reference_span",
            "visual_anchor", "owner_proxy_span", "role", "source",
        )
    )
    contract = {key: target.get(key) for key in contract_fields}
    reference_note = (
        json.dumps({
            key: reference.get(key)
            for key in (
                "candidate_id", "reference_name", "relation",
                "relation_direction", "mask_bbox",
            )
        }, ensure_ascii=False)
        if reference and reference.get("status") == "confirmed" else "none"
    )
    role_locks = [] if variant == "short" or target_evidence_path else [
        {
            "candidate_id": item.get("id"),
            "reference_phrase": item.get("query_phrase"),
            "directed_relation": item.get("relation"),
        }
        for item in row["static_query_plan"].get("reference_candidates") or []
    ]
    original_query = (
        "withheld because target candidate selection is already frozen"
        if target_evidence_path else
        row["query"] if variant == "long" else
        "withheld to preserve Short diversity"
    )
    image_paths = []
    next_image = 2
    target_image_note = "none"
    if target_evidence_path:
        target_image_note = (
            f"Image {next_image} {'selected pixels isolated on dark background' if mask_retry else 'orange selected-target mask with dimmed context'}, "
            f"selected ID={selected_target_id}"
        )
        image_paths.append(target_evidence_path)
        next_image += 1
    reference_image_note = "none"
    if evidence_path:
        reference_image_note = f"Image {next_image} green mask"
        image_paths.append(evidence_path)

    if target_evidence_path:
        decision_procedure = (
            "Candidate selection is FINAL and must not be reevaluated. Image 2 and the numeric "
            "SAM3 ROI are authoritative positive evidence for the same physical target. Trace "
            "that exact instance back to Image 1 and output a bbox strongly overlapping the ROI. "
            "Refine visible object extent according to the locked target scope; never switch to "
            "another same-class instance even if the text seems to fit it better."
        )
    elif variant == "long":
        decision_procedure = (
            "Before answering, internally enumerate every plausible instance of target_name; "
            "exclude only the instance(s) playing a locked REFERENCE role; apply target scope, "
            "every required cue, and each directed relation in order; then compare the surviving "
            "instances and choose one target. Do not assume or imitate any prior Full bbox."
        )
    elif "ROLE QUERY:" in description:
        decision_procedure = (
            "Use the query-native Short role graph directly. TARGET is a control placeholder for "
            "the exact locked target_name, not a visible word or a second object. Localize the "
            "requested target scope around TARGET. Every other noun in ROLE QUERY is a cue or "
            "reference and is NEVER the output target."
        )
    else:
        decision_procedure = (
            "Use the provided Short localization description while obeying the locked target "
            "contract. The exact target_name is the only output target; never promote a cue, "
            "subordinate object, or reference to target."
        )
    retry_note = (
        "MASK-DIRECTED RETRY: the prior answer selected the wrong instance or wrong extent. "
        "Ignore that prior answer. Use the new dark-background focus evidence and numeric ROI; "
        "return the highlighted instance only."
        if mask_retry else "none"
    )
    selected_target_block = (
        "SELECTED TARGET EVIDENCE: " + target_image_note + "\n"
        "SELECTED TARGET SAM3 ROI (POSITIVE ANCHOR, NOT A PRECOMPUTED FINAL "
        "ANSWER): " + json.dumps(selected_target_bbox) + "\n"
        if target_evidence_path else ""
    )
    prompt = f"""Image 1 is the original image.
VARIANT: {variant}
ORIGINAL QUERY: {original_query}
LOCKED TARGET CONTRACT: {json.dumps(contract, ensure_ascii=False)}
REFERENCE ROLE LOCKS: {json.dumps(role_locks, ensure_ascii=False)}
TARGET-FIRST LOCALIZATION DESCRIPTION: {description}
CONFIRMED REFERENCE INSTANCE: {reference_note}
{selected_target_block}CONFIRMED REFERENCE EVIDENCE: {reference_image_note}
IMAGE SIZE: width={width}, height={height}
DECISION PROCEDURE: {decision_procedure}
RETRY INSTRUCTION: {retry_note}

Return one tight bbox for the physical instance named by target_name. The bbox
may differ completely from the frozen Full bbox. Never promote a subordinate
or reference object. The selected evidence and ROI, when listed, are positive
TARGET anchors and the returned bbox must substantially overlap them.
The green mask, when listed, is a specific reference instance and is NEVER
TARGET, even when it shares the target noun class. Use each directed relation
without reversing target/reference roles. Output only
[x1, y1, x2, y2]."""
    try:
        generation_kwargs = (
            {"sys_prompt": SYSTEM_TARGET_BBOX} if target_evidence_path else {}
        )
        response = qwen.generate(
            prompt, image_paths=image_paths, **generation_kwargs
        )
    except Exception as exc:
        return f"QWEN_ERROR: {exc}", None
    return response, v4_eval.parse_bbox_response(response, width, height)


def bbox_behavior(full_bbox: list[float], variant_bbox: list[float] | None) -> str:
    if variant_bbox is None:
        return "UNAVAILABLE"
    overlap = bbox_iou(full_bbox, variant_bbox)
    if overlap >= 0.90:
        return "CLONE"
    if overlap >= 0.50:
        return "NEARBY"
    return "ALTERNATIVE"


def outcome_role(full_iou: float, variant_iou: float | None) -> str:
    if variant_iou is None:
        return "UNAVAILABLE"
    if full_iou <= THRESHOLD < variant_iou:
        return "RESCUE"
    if full_iou > THRESHOLD and variant_iou > THRESHOLD:
        return "SAFE_HIT"
    if full_iou > THRESHOLD >= variant_iou:
        return "DANGER"
    return "SHARED_FAIL"




def evaluate(qwen, get_sam3, image, sam_confidence: float, row: dict,
             sl_version: str = "v1") -> dict:
    source_row = row
    forbidden_fields = {"gt_bbox", "iou"}
    row = {
        key: value for key, value in source_row.items()
        if key not in forbidden_fields
    }
    prediction_forbidden_fields_absent = forbidden_fields.isdisjoint(row)
    assert prediction_forbidden_fields_absent
    started = time.time()
    full = row["descriptions"]["v4_full"]
    width, height = row["image_size"]
    if sl_version == "v1":
        short_raw = deterministic_short_raw(row)
        short, short_gate = gate_short(row, short_raw)
    else:
        short_raw, short, short_gate = choose_short_v2(row)
    short_response, short_candidate_bbox = (
        detect_locked_bbox(qwen, row, short, width, height, "short")
        if short else (None, None)
    )
    short_bbox = short_candidate_bbox
    short_blocked_bbox = None
    short_reference_shift_blocked = False
    short_reference_overlap_ious = {}
    short_verification = {
        "accepted": short is not None and short_bbox is not None,
        "status": "not_required" if sl_version == "v1" else "pending",
        "problems": [] if short_bbox is not None else ["invalid_target_bbox"],
    }
    with tempfile.TemporaryDirectory(prefix=f"full_sl_{sl_version}_") as tmp:
        reference, evidence_path = build_reference_assist(
            qwen, get_sam3, image, row, sam_confidence, Path(tmp)
        )
        if sl_version == "v2" and reference.get("status") != "confirmed":
            frozen_reference_attempt = reference
            reference, evidence_path = build_query_reference_assist(
                qwen, get_sam3, image, row, sam_confidence, Path(tmp)
            )
            reference["frozen_reference_attempt"] = frozen_reference_attempt
        visual_cue = {
            "status": "not_requested", "accepted": False, "problems": [],
            "raw": None, "target_candidate_evidence": {
                "status": "not_requested", "accepted": False, "candidates": [],
            },
        }
        visual_attempt = {
            "attempted": False, "status": "not_requested",
            "eligible_for_oracle": False, "attempts": [],
        }
        fallback = {"used": False, "reasons": [], "accepted": False}
        mask_retry = {"used": False, "accepted": False, "reasons": []}
        target_candidate_blocked = False
        target_assets = {}
        if sl_version == "v1":
            long_raw = deterministic_long_raw(row, reference)
            long, long_gate = gate_long_role(row, long_raw, reference)
        else:
            visual_cue, _masks_by_id, target_assets = build_target_visual_cue(
                qwen, get_sam3, image, row, reference, evidence_path,
                sam_confidence, Path(tmp),
            )
            if reference.get("status") != "confirmed":
                evidence_path = None
            short_gate["text_accepted"] = short_gate["accepted"]
            if short is None:
                short_verification = {
                    "accepted": False, "status": "text_rejected",
                    "problems": list(short_gate["problems"]),
                }
            else:
                mask_gate = gate_short_target_mask(
                    short_candidate_bbox,
                    visual_cue.get("target_candidate_evidence", {}),
                )
                short_reference_shift_blocked, short_reference_overlap_ious = (
                    reference_shift_check(short_candidate_bbox, reference)
                )
                short_problems = list(mask_gate["problems"])
                if short_reference_shift_blocked:
                    short_problems.append("confirmed_reference_instance_shift")
                short_problems = sorted(set(short_problems))
                short_verification = {
                    **mask_gate,
                    "accepted": not short_problems,
                    "problems": short_problems,
                    "warnings": mask_gate.get("warnings", []),
                    "reference_shift_blocked": short_reference_shift_blocked,
                    "reference_overlap_ious": short_reference_overlap_ious,
                }
                short_gate["warnings"] = sorted(set(
                    short_gate.get("warnings", []) + mask_gate.get("warnings", [])
                ))
                if short_problems:
                    short_blocked_bbox = short_candidate_bbox
                    short = None
                    short_bbox = None
                    short_gate["accepted"] = False
                    short_gate["problems"] = sorted(set(
                        short_gate["problems"] + short_problems
                    ))
            if visual_cue.get("selection", {}).get(
                "accepted", visual_cue.get("accepted", False)
            ):
                long_raw = json.dumps(long_v2_payload(row, visual_cue), ensure_ascii=False)
                long, long_gate = gate_long_v2(row, long_raw, visual_cue)
            else:
                fallback = {
                    "used": True, "trigger": "visual_cue_gate",
                    "reasons": sorted(set(visual_cue["problems"])),
                    "accepted": False,
                }
                long_raw, long, long_gate = role_long_fallback(
                    row, reference, fallback["reasons"]
                )
        grounded_target_visual = (
            long_gate.get("mode") == "grounded_target_visual"
        )
        long_response, long_candidate_bbox = (
            detect_locked_bbox(
                qwen, row, long, width, height, "long", reference, evidence_path,
                target_evidence_path=(
                    target_assets.get("context")
                    if grounded_target_visual else None
                ),
                selected_target_id=(
                    visual_cue.get("selected_id") if grounded_target_visual else None
                ),
                selected_target_bbox=(
                    visual_cue.get("selected_mask_bbox")
                    if grounded_target_visual else None
                ),
            ) if long else (None, None)
        )
        if (
            sl_version == "v2"
            and long_gate.get("mode") == "grounded_target_visual"
        ):
            visual_check = verify_visual_target_bbox(long_candidate_bbox, visual_cue)
            visual_reference_shift, visual_reference_overlaps = reference_shift_check(
                long_candidate_bbox, reference
            )
            visual_problems = list(visual_check["problems"])
            if visual_reference_shift:
                visual_problems.append("confirmed_reference_instance_shift")
            visual_problems = sorted(set(visual_problems))
            attempts = [{
                "kind": "initial_mask_guided",
                "bbox_response": long_response, "bbox": long_candidate_bbox,
                "verification": {**visual_check, "problems": visual_problems},
                "reference_shift_blocked": visual_reference_shift,
                "reference_overlap_ious": visual_reference_overlaps,
                "eligible_for_oracle": not visual_problems,
            }]
            if visual_problems:
                retry_response, retry_bbox = detect_locked_bbox(
                    qwen, row, long, width, height, "long", reference,
                    evidence_path,
                    target_evidence_path=target_assets.get("focus"),
                    selected_target_id=visual_cue.get("selected_id"),
                    selected_target_bbox=visual_cue.get("selected_mask_bbox"),
                    mask_retry=True,
                )
                retry_check = verify_visual_target_bbox(retry_bbox, visual_cue)
                retry_reference_shift, retry_reference_overlaps = reference_shift_check(
                    retry_bbox, reference
                )
                retry_problems = list(retry_check["problems"])
                if retry_reference_shift:
                    retry_problems.append("confirmed_reference_instance_shift")
                retry_problems = sorted(set(retry_problems))
                attempts.append({
                    "kind": "mask_directed_retry",
                    "bbox_response": retry_response, "bbox": retry_bbox,
                    "verification": {**retry_check, "problems": retry_problems},
                    "reference_shift_blocked": retry_reference_shift,
                    "reference_overlap_ious": retry_reference_overlaps,
                    "eligible_for_oracle": not retry_problems,
                })
                mask_retry = {
                    "used": True, "accepted": not retry_problems,
                    "reasons": retry_problems,
                }
                long_response, long_candidate_bbox = retry_response, retry_bbox
                visual_check = retry_check
                visual_problems = retry_problems
                visual_reference_shift = retry_reference_shift
                visual_reference_overlaps = retry_reference_overlaps
            target_candidate_blocked = (
                "grounded_target_candidate_mismatch" in visual_problems
                or "other_target_candidate_dominates" in visual_problems
            )
            visual_attempt = {
                "attempted": True,
                "status": (
                    "accepted_after_retry" if mask_retry["accepted"] else
                    "accepted" if not visual_problems else "discarded_postcheck"
                ),
                "eligible_for_oracle": not visual_problems,
                "target_candidate_mismatch": target_candidate_blocked,
                "raw": long_raw, "description": long,
                "bbox_response": long_response, "bbox": long_candidate_bbox,
                "verification": {**visual_check, "problems": visual_problems},
                "reference_shift_blocked": visual_reference_shift,
                "reference_overlap_ious": visual_reference_overlaps,
                "attempts": attempts,
                "final_attempt": attempts[-1]["kind"],
            }
            if visual_problems:
                visual_cue["pre_bbox_accepted"] = visual_cue.get("accepted", False)
                visual_cue["bbox_status"] = "rejected_after_mask_retry"
                visual_cue["bbox_problems"] = visual_problems
                fallback = {
                    "used": True, "trigger": "visual_bbox_postcheck_after_retry",
                    "reasons": visual_problems, "accepted": False,
                }
                long_raw, long, long_gate = role_long_fallback(
                    row, reference, fallback["reasons"]
                )
                long_response, long_candidate_bbox = (
                    detect_locked_bbox(
                        qwen, row, long, width, height, "long", reference,
                        evidence_path,
                    ) if long else (None, None)
                )

        reference_shift_blocked, reference_overlap_ious = reference_shift_check(
            long_candidate_bbox, reference
        )
        reference_overlap_iou = reference_overlap_ious.get("sam_reference")
        verification_problems = []
        if long_candidate_bbox is None:
            verification_problems.append("invalid_target_bbox")
        if reference_shift_blocked:
            verification_problems.append("confirmed_reference_instance_shift")
        verification_problems = sorted(set(verification_problems))
        verification = {
            "accepted": long is not None and not verification_problems,
            "problems": verification_problems,
            "mode": long_gate["mode"],
            "support_level": long_gate["support_level"],
            "reference_lock": confirmed_reference_note(reference),
        }
        if long_gate.get("mode") == "grounded_target_visual":
            verification.update(visual_attempt.get("verification", {}))
            verification["accepted"] = long is not None and not verification_problems
            verification["problems"] = verification_problems
        if verification_problems:
            long = None
            long_gate["accepted"] = False
            long_gate["problems"] = sorted(set(
                long_gate["problems"] + verification_problems
            ))
        long_bbox = long_candidate_bbox if long is not None else None
        if fallback["used"]:
            fallback["accepted"] = long_bbox is not None

    # GT is used only after S/L descriptions and boxes are fixed.
    gt_bbox = source_row["gt_bbox"]
    iou_full = bbox_iou(row["bbox"]["v4_full"], gt_bbox)
    iou_short = bbox_iou(short_bbox, gt_bbox) if short_bbox is not None else None
    iou_long = bbox_iou(long_bbox, gt_bbox) if long_bbox is not None else None
    available_ious = [value for value in (iou_full, iou_short, iou_long) if value is not None]
    candidate_evidence = visual_cue.get("target_candidate_evidence", {})
    candidate_gt_ious = {
        item["id"]: bbox_iou(item["bbox"], gt_bbox)
        for item in candidate_evidence.get("candidates") or []
    }
    proposed_candidate_iou = candidate_gt_ious.get(visual_cue.get("selected_id"))
    selected_candidate_iou = (
        proposed_candidate_iou
        if visual_cue.get("selection", {}).get("accepted", False) else None
    )
    return {
        "schema_version": f"{SCHEMA_VERSION}-{sl_version}",
        "sl_version": sl_version,
        "dataset": row["dataset"],
        "split_label": row["split_label"],
        "sample_key": row["sample_key"],
        "index": row["index"],
        "image_id": row["image_id"],
        "image_size": row["image_size"],
        "query": row["query"],
        "target": row["target"],
        "mandatory_cues": mandatory_cues(row),
        "full": {"description": full, "bbox": row["bbox"]["v4_full"]},
        "short": {
            "raw": short_raw, "description": short, "gate": short_gate,
            "bbox_response": short_response, "bbox": short_bbox,
            "verification": short_verification,
            "blocked_bbox": short_blocked_bbox,
            "reference_shift_blocked": short_reference_shift_blocked,
            "reference_overlap_ious": short_reference_overlap_ious,
        },
        "long": {
            "raw": long_raw, "description": long, "gate": long_gate,
            "bbox_response": long_response, "bbox": long_bbox,
            "verification": verification,
            "blocked_bbox": long_candidate_bbox if reference_shift_blocked else None,
            "reference_assist": reference,
            "visual_cue": visual_cue,
            "visual_attempt": visual_attempt,
            "mask_retry": mask_retry,
            "fallback": fallback,
            "reference_overlap_iou": reference_overlap_iou,
            "reference_overlap_ious": reference_overlap_ious,
            "reference_shift_blocked": reference_shift_blocked,
        },
        "gt_bbox": gt_bbox,
        "iou": {
            "full": iou_full,
            "short": iou_short,
            "long": iou_long,
            "oracle": max(available_ious),
        },
        "candidate_analysis": {
            "short_bbox_behavior": bbox_behavior(row["bbox"]["v4_full"], short_bbox),
            "long_bbox_behavior": bbox_behavior(row["bbox"]["v4_full"], long_bbox),
            "short_outcome": outcome_role(iou_full, iou_short),
            "long_outcome": outcome_role(iou_full, iou_long),
            "post_prediction_only": True,
            "candidate_pool_best_gt_iou": (
                max(candidate_gt_ious.values()) if candidate_gt_ious else None
            ),
            "proposed_candidate_gt_iou": proposed_candidate_iou,
            "selected_candidate_gt_iou": selected_candidate_iou,
            "candidate_pool_hit_at_50": any(
                value > THRESHOLD for value in candidate_gt_ious.values()
            ),
            "candidate_pool_hit_at_75": any(
                value >= STRICT_THRESHOLD for value in candidate_gt_ious.values()
            ),
            "selected_candidate_hit_at_50": (
                selected_candidate_iou is not None
                and selected_candidate_iou > THRESHOLD
            ),
            "selected_candidate_hit_at_75": (
                selected_candidate_iou is not None
                and selected_candidate_iou >= STRICT_THRESHOLD
            ),
            "selected_candidate_preserved_by_final_long": (
                selected_candidate_iou is not None
                and long_gate.get("mode") == "grounded_target_visual"
                and long_bbox is not None
                and verify_visual_target_bbox(long_bbox, visual_cue)["accepted"]
            ),
        },
        "prediction_audit": {
            "forbidden_fields": sorted(forbidden_fields),
            "forbidden_fields_absent": prediction_forbidden_fields_absent,
            "gt_access_stage": "after_full_short_long_fixed",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
