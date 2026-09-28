#!/usr/bin/env python3
"""Generate one downstream Full Description and one selected SAM3 mask."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import numpy as np
from PIL import Image, ImageDraw


REQUIRED_QWEN = "Qwen2.5-VL-7B-Instruct"
SCHEMA_VERSION = "full_des_generation_fused_v1"
ROOT = Path(__file__).resolve().parent
FINAL_OUTPUT_ROOT = ROOT / "result" / "FullDes_Generation" / "FULL_FINAL"

CONTRACT_KEYS = frozenset({
    "answer_role_phrase", "required_kind", "required_number",
    "required_extent", "required_selector", "query_evidence",
})
PLAN_KEYS = frozenset({
    "full_description",
    "target_entity",
    "target_number",
    "target_scope",
    "selector",
    "visible_discriminator",
    "evidence",
    "confidence",
    "primary_prompt",
    "alternate_prompt",
    "bbox_xyxy_1000",
})
VISUAL_BINDING_KEYS = frozenset({
    "target_core", "target_kind", "number", "extent", "owner", "selector",
    "grounding_status", "evidence",
})
OBSERVATION_KEYS = frozenset({
    "label",
    "covered_entity",
    "covered_scope",
    "covered_number",
    "coverage",
    "uncovered_equivalents",
    "parent_entity",
    "evidence",
})
SELECTION_KEYS = frozenset({"choice", "reason"})
NUMBERS = frozenset({"singular", "plural", "unknown"})
SCOPES = frozenset({
    "whole_object",
    "physical_part",
    "visible_region",
    "place_support",
    "material_substance",
    "group_set",
    "unknown",
})
KINDS = SCOPES | frozenset({"person_or_animal"})
CONFIDENCES = frozenset({"high", "medium", "low"})
GROUNDING_STATUSES = frozenset({"clear", "ambiguous", "not_visible"})
COVERAGES = frozenset({
    "complete", "partial", "overinclusive", "empty", "unknown"
})
YES_NO_UNKNOWN = frozenset({"yes", "no", "unknown"})
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


@dataclass
class Candidate:
    """One actual SAM3 output considered for final selection."""

    prompt: str | None
    mask: np.ndarray
    source: str
    variant: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Ordered record of model calls and deterministic decisions."""

    qwen_call_count: int = 0
    sam3_call_count: int = 0
    stages: list[dict[str, Any]] = field(default_factory=list)
    work_dir: str = ""

    def record(self, role: str, model: str, **details: Any) -> None:
        self.stages.append({"role": role, "model": model, **details})


def _model_name(qwen: object) -> str:
    cfg = getattr(qwen, "qwen_cfg", None)
    return str(
        getattr(cfg, "model", "")
        or getattr(cfg, "model_dir", "")
        or ""
    )


def parse_json_object(
    text: str,
    *,
    null_conflicting_duplicate_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    raw = str(text).strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        raise ValueError("response is not one bare JSON object")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                if key in null_conflicting_duplicate_keys:
                    if value[key] != item:
                        value[key] = None
                    continue
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("JSON response is not an object")
    return value


def _parse_target_plan_object(text: str) -> dict[str, Any]:
    return parse_json_object(
        text,
        null_conflicting_duplicate_keys=frozenset({"bbox_xyxy_1000"}),
    )


def _repaired_json_payload(text: str) -> str:
    raw = str(text).strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    wrapped = re.fullmatch(
        r"(?:(?:here is|corrected json)[^{}\[\]()`\n]{0,80}:\s*)?"
        r"```(?:json)?\s*(\{.*\})\s*```",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if wrapped is None:
        raise ValueError("response is not one bare JSON object")
    return wrapped.group(1).strip()


def _target_plan_without_malformed_bbox_tail(
    text: str,
) -> dict[str, Any] | None:
    raw = str(text).strip()
    marker = '"bbox_xyxy_1000"'
    if not raw.startswith("{") or raw.count(marker) != 1:
        return None
    prefix = raw.split(marker, 1)[0]
    try:
        value = _parse_target_plan_object(f"{prefix}{marker}: null}}")
    except ValueError:
        return None
    return value if set(value) == set(PLAN_KEYS) else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_sha256(mask: np.ndarray) -> str:
    packed = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _qwen_raw(
    qwen: object,
    role: str,
    prompt: str,
    trace: Trace,
    *,
    use_image: bool,
    max_tokens: int,
) -> str:
    if hasattr(qwen, "generate_for_role"):
        raw = qwen.generate_for_role(role, prompt, use_image=use_image)
    elif all(hasattr(qwen, name) for name in ("client", "qwen_cfg")):
        content: list[dict[str, Any]] = []
        if use_image:
            image_url = getattr(qwen, "ori_image", None)
            if not image_url:
                raise RuntimeError("Qwen image was not loaded")
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })
        content.append({"type": "text", "text": prompt})
        kwargs: dict[str, Any] = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object and nothing else. "
                        "Use the requested keys, types, and enums exactly."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        configured_model = getattr(qwen.qwen_cfg, "model", None)
        if configured_model:
            kwargs["model"] = configured_model
        response = qwen.client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
    else:
        try:
            raw = qwen.generate(prompt, use_ori_image=use_image)
        except TypeError:
            raw = qwen.generate(prompt)

    raw = str(raw).strip()
    trace.qwen_call_count += 1
    trace.record(
        role,
        "qwen",
        use_image=bool(use_image),
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        response=raw,
    )
    return raw

def _qwen_text(
    qwen: object,
    role: str,
    prompt: str,
    trace: Trace,
    *,
    use_image: bool,
) -> str:
    if hasattr(qwen, "generate_for_role"):
        raw = qwen.generate_for_role(role, prompt, use_image=use_image)
    else:
        try:
            raw = qwen.generate(prompt, use_ori_image=use_image)
        except TypeError:
            raw = qwen.generate(prompt)

    raw = str(raw).strip()
    trace.qwen_call_count += 1
    trace.record(
        role,
        "qwen",
        use_image=bool(use_image),
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        response=raw,
    )
    return raw

QUERY_CONTRACT_INSTRUCTIONS = """ROLE: query_contract_extractor

Analyze only the QUERY; the image is unavailable. Locate the final
interrogative clause or requested noun phrase containing the open answer slot.
answer_role_phrase must describe the missing participant that the answer itself
fills. A noun mentioned only in background, an example, or a selector cannot
replace that open role.

Treat something/someone/object modified by a relative clause as the predicate
bearer: the entity that performs the action or bears the state. Do not return
the action, effect, emitted material, owner, actor, user, carrier, instrument,
container, content, support, neighbor, parent, child, whole, or part unless that
entity itself fills the open slot. Preserve relation direction: what is jumped
over, worn, held, contained, supported, or indicated is distinct from the
jumper, wearer, holder, container, supporter, indicated entity, or owner.
When the open slot asks for a visible bearer that conveys, indicates, or
describes information about a reference, that visible bearer is the target;
the reference and the information value are not.

Preserve every answer-modifying conjunction, relation, location, state,
appearance, ordinal, and ownership constraint in required_selector. Do not
guess a visually present category. Generic words such as thing, object, part,
area, place, or feature are not useful answer roles; express the full semantic
role they must satisfy.

Set number only from explicit quantification of the open answer target.
one/single means singular; both/all/every/each or a target-count numeral means
plural. Grammatical number on an unquantified placeholder or role noun does not
lock instance count. Ignore count words modifying only an attribute or reference.
Use unknown when the query leaves target count open; image binding may then
determine how many visible target instances jointly fill the role.
Use physical_part only for an explicitly requested attached component,
visible_region only for an explicitly requested bounded surface/portion, and
whole_object only when the open phrase explicitly asks for an independently
bounded object/person/animal. Otherwise use unknown. required_kind and
required_extent must agree except person_or_animal uses whole_object extent.

Return exactly one object with these string fields and no others:
{
  "answer_role_phrase": "<short direct open-variable role>",
  "required_kind": "<whole_object|person_or_animal|physical_part|visible_region|place_support|material_substance|group_set|unknown>",
  "required_number": "<singular|plural|unknown>",
  "required_extent": "<whole_object|physical_part|visible_region|place_support|material_substance|group_set|unknown>",
  "required_selector": "<explicit answer-modifying constraints or empty string>",
  "query_evidence": "<short exact query evidence>"
}"""


VISUAL_BINDING_INSTRUCTIONS = """Bind only the locked answer slot to a concrete visible common-category noun. Never return the owner, beneficiary,
holder, wearer, user, carrier, support, container, containing whole, or acted-on
object when it is only relational. For a requested physical part or visible
region, target_core must name that part/region and must not equal owner. For a
functional object, name the visible object, not an abstract purpose or action.
selector must be a short visible attribute or relation of at most three words;
do not write an explanation.
number counts visible target_core instances to segment, never owners or
references. Use plural when two or more interchangeable visible instances each
fill the answer role and should be covered together.

Return exactly one object with the supplied keys and no others. target_kind uses
the requested_kind enum; number=singular|plural|unknown; extent=whole_object|
physical_part|visible_region|place_support|material_substance|group_set|unknown;
grounding_status=clear|ambiguous|not_visible. All values are strings."""


TARGET_PLAN_INSTRUCTIONS = """ROLE: locked_target_description_planner

The original unmodified image is loaded. QUERY is authoritative and
QUERY_CONTRACT is only a structured aid. CANONICAL_TARGET_PROMPT and
BOUND_TARGET_PROMPT are independently generated hypotheses, not hard locks.
Re-solve the open answer slot against visible evidence and choose the physical
target whose own pixels fill it. Its noun must itself complete the final query;
an actor, owner, host, carrier, container, content, background, action, effect,
whole, part, or nearby object cannot replace the missing participant. Keep an
independent hypothesis when it is correct, but challenge it when the image and
query jointly support a different directly answering target.
When the open slot asks for a visible bearer that conveys, indicates, or
describes information about a reference, that visible bearer is the target;
the reference and the information value are not.

full_description is a factual target-centered downstream description and is
never a SAM3 prompt. primary_prompt is a separate SAM3-localizable noun phrase
of at most 12 words. Put the target noun in its first three words, then add only
visible appearance, location, attachment, or directed-relation cues that help
localize it. alternate_prompt is a genuinely useful short alias for that same
target or empty. bbox_xyxy_1000 is one tight target box [x1,y1,x2,y2] in
normalized 0..1000 coordinates, or null; never copy a schema example or return
a broad scene box.
Write target_entity as the grammatical subject at the start of full_description
and evidence. If no target-specific visual evidence is available, use only the
target_entity instead of describing another entity.

Use target_scope=whole_object for a complete independently bounded entity,
physical_part for an attached component, visible_region for a bounded surface
portion, place_support for a spatial support, material_substance for visible
material, and group_set for a requested set. Obey explicit one/all/both/numerals.
When count is not explicit, use plural if several visible interchangeable
instances independently fill the answer role and should be segmented together;

Return exactly one object with these keys and no others:
{
  "full_description": "<independent downstream description>",
  "target_entity": "<visible target noun>",
  "target_number": "<singular|plural|unknown>",
  "target_scope": "<whole_object|physical_part|visible_region|place_support|material_substance|group_set|unknown>",
  "selector": "<required visible selector or empty string>",
  "visible_discriminator": "<short visible cue or empty string>",
  "evidence": "<short visible evidence about the locked target>",
  "confidence": "<high|medium|low>",
  "primary_prompt": "<the short locked target phrase>",
  "alternate_prompt": "<different short alias for the same target or empty>",
  "bbox_xyxy_1000": null
}
Use a four-number array only for a reliable box; otherwise use JSON null."""


OBSERVE_INSTRUCTIONS = """ROLE: anonymous_candidate_observer

The loaded board contains only anonymously labelled candidate masks. Every
candidate tile uses the same layout: a full-image green overlay with magenta
boundary, a tight overlay zoom, and the covered real pixels isolated on a dark
background. Colors, boundaries, and labels are interface artifacts.
The three panels beneath one label are three views of the SAME mask, not three
candidates. Return exactly one observation for each supplied label, in supplied
label order. Never emit the same label twice and never describe the three views
as separate masks.

You do not know the query, intended target, prompt, candidate source, or model
confidence. Do not guess them. For every label, independently describe only the
real-world content actually covered. Distinguish a complete whole object from a
part, surface region, place/support, material/substance, or group. Count covered
instances of the same entity. Mark overinclusive when unrelated entities or
substantial background are included. Mark uncovered_equivalents=yes when other
visible instances of the same semantic category remain outside that candidate.
Use unknown rather than inventing identity from unclear pixels.

Return exactly:
{"observations":[
  {
    "label":"<one supplied label>",
    "covered_entity":"<literal real-world entity or unknown>",
    "covered_scope":"<whole_object|physical_part|visible_region|place_support|material_substance|group_set|unknown>",
    "covered_number":"<singular|plural|unknown>",
    "coverage":"<complete|partial|overinclusive|empty|unknown>",
    "uncovered_equivalents":"<yes|no|unknown>",
    "parent_entity":"<attached/containing entity or empty string>",
    "evidence":"<at most 12 words of literal visual evidence>"
  }
]}
Include every supplied label exactly once and no other keys."""


SELECT_INSTRUCTIONS = """ROLE: independent_candidate_selector

Use the QUERY as authoritative. The loaded image is the same anonymously
labelled candidate board used by the query-blind observer. Inspect every
label's full overlay, tight zoom, isolated pixels, and boundary directly.
The observations are an independent first pass made without the query. Re-read
the complete QUERY yourself: its final question or request is authoritative over
background sentences. Each anonymous TARGET_CLAIM records the entity a candidate
intended to segment; it is a hypothesis, never visual proof and never a
source-quality signal. The chosen TARGET_CLAIM.target_entity is published verbatim
as the final target name. If that name itself cannot fill the QUERY answer slot,
related pixels cannot rescue it. Then use the board and locked observation to
verify that the mask actually covers the claimed entity; a claim that answers the
query cannot license pixels covering a different entity. Correct an obvious
observation mistake only when the board visibly shows a different boundary.
When the open slot asks for a visible bearer that conveys, indicates, or describes
information about a reference, that visible bearer is the target; the reference
and the information value are not.

A related owner, actor, carrier, instrument, container, content, support,
parent, child, part, whole, neighbor, effect, or material is not the same
target. When the query joins multiple target predicates, the same claimed entity
must satisfy every predicate directly; satisfying only one is insufficient.
Check requested number, scope, explicit selector, and complete coverage
separately. Labels may share a claim when they represent different mask extents
for the same intended target.

For plural or all/both requests, prefer a candidate covering every requested
instance without unrelated objects; a single instance or uncovered equivalent
is insufficient. Apply a number word to the target only when it quantifies the
answer target rather than an attribute or reference. For a singular selected
instance, other unselected peers may remain visible. Reject empty, partial, or
overinclusive candidates when a
complete precise candidate exists. Geometry is supporting evidence only and
cannot establish semantic identity.

Return exactly:
{"choice":"<one supplied label>","reason":"<short decisive reason>"}"""


def canonical_target_prompt(query: str) -> str:
    return f"""The image is loaded.
ORIGINAL_QUERY: {query}

Solve target identity only in this round. Before wording, silently lock:
- TARGET: the entity, part, visible region, place/support, material/substance,
  or set whose OWN visible pixels answer the query.
- SCOPE: exactly one of whole_object, physical_part, visible_region, place_support,
  material_substance, group_set, or unknown.
- REFERENCES: entities used only to identify TARGET; never promote a reference.
- LOCKS: explicit count, ordinal, owner/source, state, negation, physical medium,
  functional role, and directed relation.

When the open slot asks for a visible bearer that conveys, indicates, or
describes information about a reference, that visible bearer is the TARGET;
the reference and the information value are not.

Use one query-syntax procedure for every input. If the complete query is already an
explicit noun phrase, preserve its lexical target head and attach its owner, state,
count, ordinal, and relation locks to that head. Otherwise identify the matrix
answer-bearing WH operator of the main question; embedded or relative participants
are REFERENCES, not replacement targets. Narrative and reference-only words create
no locks. Retain count, ordinal, owner, state, negation, and directed-role locks only
when the matrix answer slot or its target predicate explicitly licenses them.

If the query or image creates genuine ambiguity, internally compare at most two
plausible visible candidates: the first supported guess and one alternative from a
different category, state, scope, or role. Never manufacture an alternative for an
unambiguous query-explicit target. Apply the same hard checks to both. Reject a
candidate if its own pixels fail WH focus or scope, contradict a mutually exclusive
state, mismatch the physical medium or event stage, fill the wrong actor,
beneficiary/tool/carrier/owner/payload/source/support/marker role, or face positive
visible counterevidence. Salience and proximity never override a failed check. For
every required relation preserve the directed tuple
(TARGET, DIRECTED_RELATION, REFERENCE); keeping only a relation word is not enough
if a participant or direction changes. Do not reveal candidates or reasoning.

If the query explicitly names a concrete visible target category, the image may
select the matching instance but may not rename that head. A generic answer-slot
type such as entity, object, part, region, or place does not by itself name a target
category. Infer one broad visible head only for such a generic slot or a truly
headless or functional query. Return the shortest phrase that preserves all locks.

Apply the same rules across every category and scene. The grammatical answer
slot fixes target identity; relations constrain that target but never promote a
reference into it. Choose exactly the visible extent whose own pixels answer the
query: whole object, physical part, visible region, place/support, material, or
requested group/set.

Use an ordinary, visually groundable noun phrase. Express scope only through the
grammatical answer head and the fixed scope enum above; never infer scope from a
reference noun. When a bounded visible region is the answer, that region remains
TARGET and its carrier, source, owner, or observer remains context. If intended
pixel extent is ambiguous, do not change target scope. A requested physical part
remains TARGET; its whole owner is only a selector and never replaces the part.

Avoid adjective stuffing. Every query-explicit LOCK is mandatory and does not count
toward the optional selector budget. Add at most one image-only same-head
disambiguator, and add none when unnecessary.

Output exactly one 2-to-9-word English noun phrase beginning with The, A, or An.
No sentence, explanation, label, alternative, JSON, code fence, colon, final period,
or negative observation."""
def query_contract_prompt(query: str) -> str:
    return f"{QUERY_CONTRACT_INSTRUCTIONS}\n\nQUERY: {query}"


def visual_binding_prompt(
    query: str,
    contract: dict[str, str],
) -> str:
    slot = {
        "answer_head_phrase": contract["answer_role_phrase"],
        "requested_kind": contract["required_kind"],
        "requested_number": contract["required_number"],
        "explicit_owner": "",
        "relation_participant": contract["answer_role_phrase"],
        "selector": contract["required_selector"],
        "semantic_ambiguity": "ambiguous",
        "rationale": (
            f"The final question asks for {contract['answer_role_phrase']}."
        ),
    }
    schema = {
        key: "<value>" for key in sorted(VISUAL_BINDING_KEYS)
    }
    return (
        "ROLE: locked_answer_slot_image_binding\n"
        "The original unmodified image is loaded.\n"
        f"QUERY: {query}\n"
        f"LOCKED_ANSWER_SLOT: "
        f"{json.dumps(slot, ensure_ascii=False, sort_keys=True)}\n\n"
        f"{VISUAL_BINDING_INSTRUCTIONS}\n\n"
        "Return exactly one bare JSON object with these keys and no others:\n"
        f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
    )


def target_plan_prompt(
    query: str,
    contract: dict[str, str],
    bound_prompt: str,
    canonical_prompt: str,
) -> str:
    packed = json.dumps(contract, ensure_ascii=False, sort_keys=True)
    return (
        f"{TARGET_PLAN_INSTRUCTIONS}\n\n"
        f"QUERY_CONTRACT: {packed}\n"
        f"CANONICAL_TARGET_PROMPT: {canonical_prompt}\n"
        f"BOUND_TARGET_PROMPT: {bound_prompt}\n"
        f"QUERY: {query}"
    )


def observation_prompt(labels: list[str]) -> str:
    packed = json.dumps(labels, ensure_ascii=False)
    return f"{OBSERVE_INSTRUCTIONS}\n\nSUPPLIED_LABELS: {packed}"


def selection_prompt(
    query: str,
    observations: list[dict[str, str]],
    target_claims: list[dict[str, str]],
    geometry: dict[str, Any],
) -> str:
    return (
        f"{SELECT_INSTRUCTIONS}\n\n"
        f"QUERY: {query}\n"
        f"CANDIDATE_TARGET_CLAIMS: "
        f"{json.dumps(target_claims, ensure_ascii=False, sort_keys=True)}\n"
        f"LOCKED_OBSERVATIONS: "
        f"{json.dumps(observations, ensure_ascii=False, sort_keys=True)}\n"
        f"MASK_GEOMETRY: "
        f"{json.dumps(geometry, ensure_ascii=False, sort_keys=True)}"
    )


def _clean_string(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
    max_words: int | None = None,
    max_chars: int | None = None,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = " ".join(value.replace("\n", " ").split()).strip()
    if not cleaned and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    # if max_words is not None and len(_WORD_RE.findall(cleaned)) > max_words:
    #     raise ValueError(f"{field_name} exceeds {max_words} words")
    if max_chars is not None and len(cleaned) > max_chars:
        raise ValueError(f"{field_name} exceeds {max_chars} characters")
    return cleaned


def _clean_truncated_words(
    value: Any,
    field_name: str,
    max_words: int,
) -> str:
    cleaned = _clean_string(value, field_name)
    words = list(_WORD_RE.finditer(cleaned))
    if len(words) <= max_words:
        return cleaned
    return cleaned[:words[max_words - 1].end()].rstrip()


def _normalize_bbox(
    value: Any,
    image_size: tuple[int, int] | None = None,
) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        packed = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(packed) != 4:
            return None
        value = packed
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    coordinates: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        coordinates.append(number)

    if image_size and any(item > 1000.0 for item in coordinates):
        width, height = image_size
        if width <= 0 or height <= 0:
            return None
        coordinates = [
            coordinates[0] * 1000.0 / width,
            coordinates[1] * 1000.0 / height,
            coordinates[2] * 1000.0 / width,
            coordinates[3] * 1000.0 / height,
        ]
    coordinates = [min(1000.0, max(0.0, item)) for item in coordinates]
    x1, y1, x2, y2 = coordinates
    if not (x1 < x2 and y1 < y2):
        return None
    return coordinates


def _drop_coordinate_key_noise(
    value: dict[str, Any],
) -> dict[str, Any] | None:
    if set(PLAN_KEYS) - set(value):
        return None
    extras = set(value) - set(PLAN_KEYS)
    if len(extras) != 1:
        return None
    key = next(iter(extras))
    match = re.fullmatch(
        r"\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,"
        r"\s*(\d{1,4})\s*,\s*(\d{1,4})\s*",
        key,
    )
    if match is None:
        return None

    def strict_bbox(item: Any) -> list[float] | None:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            return None
        if any(isinstance(x, bool) or not isinstance(x, int) for x in item):
            return None
        if any(x < 0 or x > 1000 for x in item):
            return None
        x1, y1, x2, y2 = item
        if not (x1 < x2 and y1 < y2):
            return None
        return [float(x) for x in item]

    coordinate_bbox = strict_bbox([int(x) for x in match.groups()])
    official_bbox = strict_bbox(value["bbox_xyxy_1000"])
    extra_bbox = strict_bbox(value[key])
    if any(item is None for item in (
        coordinate_bbox, official_bbox, extra_bbox
    )):
        return None
    if official_bbox == coordinate_bbox == extra_bbox:
        return None
    cleaned = dict(value)
    cleaned.pop(key)
    cleaned["bbox_xyxy_1000"] = None
    return cleaned


def _normalize_number(value: str) -> str:
    lowered = value.casefold().strip()
    if lowered in {"singular", "one", "single", "1"}:
        return "singular"
    if lowered in {
        "plural", "multiple", "many", "several", "both", "all",
        "two", "three", "four", "five", "2", "3", "4", "5",
    }:
        return "plural"
    if lowered in {"unknown", "unspecified", "unclear", "not specified"}:
        return "unknown"
    return lowered


def validate_query_contract(
    value: dict[str, Any],
    query: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(CONTRACT_KEYS):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"query contract keys must be {sorted(CONTRACT_KEYS)}, got {actual}"
        )
    contract = {
        "answer_role_phrase": _clean_string(
            value["answer_role_phrase"], "answer_role_phrase", max_words=24
        ),
        "required_kind": _clean_string(
            value["required_kind"], "required_kind"
        ).casefold(),
        "required_number": _normalize_number(_clean_string(
            value["required_number"], "required_number"
        )),
        "required_extent": _clean_string(
            value["required_extent"], "required_extent"
        ).casefold(),
        "required_selector": _clean_string(
            value["required_selector"],
            "required_selector",
            allow_empty=True,
            max_words=30,
        ),
        "query_evidence": _clean_truncated_words(
            value["query_evidence"], "query_evidence", 30
        ),
    }
    if contract["required_kind"] not in KINDS:
        contract["required_kind"] = (
            contract["required_extent"]
            if contract["required_extent"] in SCOPES else "unknown"
        )
    if contract["required_number"] not in NUMBERS:
        raise ValueError("invalid required_number")
    if contract["required_extent"] not in SCOPES:
        kind = contract["required_kind"]
        contract["required_extent"] = (
            "whole_object" if kind == "person_or_animal"
            else kind if kind in SCOPES else "unknown"
        )

    return contract


def build_query_contract(
    query: str,
    qwen: object,
    trace: Trace,
) -> dict[str, str]:
    prompt = query_contract_prompt(query)
    raw = _qwen_raw(
        qwen,
        "query_contract",
        prompt,
        trace,
        use_image=False,
        max_tokens=500,
    )
    try:
        return validate_query_contract(parse_json_object(raw), query)
    except ValueError as error:
        repair_prompt = (
            f"{prompt}\n\n"
            "The previous response failed structural validation. Return a fresh "
            "complete object satisfying the exact same query-only task and schema.\n"
            f"VALIDATION_ERROR: {error}\n"
            f"REJECTED_RESPONSE: {raw}"
        )
        repaired = _qwen_raw(
            qwen,
            "query_contract_schema_repair",
            repair_prompt,
            trace,
            use_image=False,
            max_tokens=500,
        )
        repaired_payload = _repaired_json_payload(repaired)
        validated = validate_query_contract(
            parse_json_object(repaired_payload), query
        )
        if repaired_payload != str(repaired).strip():
            trace.record("query_contract_wrapper_fallback", "pipeline")
        return validated

def _clean_canonical_target(value: Any) -> str:
    cleaned = str(value).strip()
    cleaned = re.sub(
        r"^\x60\x60\x60(?:text)?\s*|\s*\x60\x60\x60$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = cleaned.rstrip(".").strip()
    return _clean_string(
        cleaned, "canonical_target", max_words=12, max_chars=100
    )


def build_canonical_target(
    image_path: str,
    query: str,
    qwen: object,
    trace: Trace,
) -> str:
    resolved_image = Path(image_path).resolve()
    qwen.load_image(str(resolved_image))
    raw = _qwen_text(
        qwen,
        "canonical_target",
        canonical_target_prompt(query),
        trace,
        use_image=True,
    )
    try:
        return _clean_canonical_target(raw)
    except ValueError as error:
        trace.record(
            "canonical_target_invalid",
            "pipeline",
            error=str(error),
        )
        return ""


def _phrase_target_core(phrase: str) -> str:
    value = _clean_canonical_target(phrase)
    value = re.sub(
        r"^(?:the|a|an|all|both)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    relation = re.search(
        r"\s+(?:of|on|in|at|under|above|below|beside|near|with|"
        r"attached|that|which|who|wearing|used|supporting|holding|held|"
        r"carrying|carried|containing|behind|between|over|assisting)\b",
        value,
        flags=re.IGNORECASE,
    )
    if relation:
        value = value[:relation.start()].strip()
    value = re.sub(
        r"\s+(?:far\s+)?(?:left|right|top|bottom|center|middle|front|back)"
        r"(?:most)?$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    return _clean_string(value, "canonical_target_core", max_words=12)


def _target_head(phrase: str) -> str:
    words = _WORD_RE.findall(phrase)
    return words[-1].casefold() if words else ""

def _target_number_hint(phrase: str) -> str | None:
    words = [word.casefold() for word in _WORD_RE.findall(str(phrase))]
    if not words:
        return None
    if words[0] in {"all", "both"}:
        return "plural"
    if words[-1] in {
        "children", "feet", "geese", "men", "mice", "people",
        "teeth", "women",
    }:
        return "plural"
    return None


def _word_forms(word: str) -> set[str]:
    token = str(word).casefold()
    forms = {token}
    if len(token) > 4 and token.endswith("ies"):
        forms.add(token[:-3] + "y")
    if len(token) > 3 and token.endswith("es"):
        forms.update((token[:-2], token[:-1]))
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        forms.add(token[:-1])
    return forms


def _content_words(value: str) -> list[str]:
    return [
        word.casefold() for word in _WORD_RE.findall(str(value))
        if word.casefold() not in {"the", "a", "an", "all", "both"}
    ]


def _token_stems(value: str) -> set[str]:
    stems: set[str] = set()
    for token in _content_words(value):
        stems.update(_word_forms(token))
    return stems


def _target_evidence_is_plural(target: str, evidence: str) -> bool:
    head = _target_head(str(target))
    if not head:
        return False
    head_forms = _word_forms(head)
    words = [word.casefold() for word in _WORD_RE.findall(str(evidence))]
    return any(
        bool(head_forms & _word_forms(word))
        and words[index + 1] in {"are", "were", "have"}
        for index, word in enumerate(words[:-1])
    )


def _same_target_name(left: str, right: str) -> bool:
    left_words = _content_words(left)
    right_words = _content_words(right)
    if not left_words or not right_words:
        return False
    if len(left_words) == len(right_words) and all(
        _word_forms(a) & _word_forms(b)
        for a, b in zip(left_words, right_words)
    ):
        return True
    return (
        (len(left_words) == 1 or len(right_words) == 1)
        and bool(_word_forms(left_words[-1]) & _word_forms(right_words[-1]))
    )


def _quantified_target_is_plural(target: str, evidence: str) -> bool:
    head_stems = _token_stems(_target_head(target))
    words = _WORD_RE.findall(str(evidence).casefold())
    for index, word in enumerate(words):
        if word in {"multiple", "several", "two", "three", "four", "five"}:
            if head_stems & _token_stems(" ".join(words[index + 1:index + 4])):
                return True
    return False


def _canonical_binding(
    contract: dict[str, str],
    canonical_prompt: str,
) -> dict[str, str]:
    target = _phrase_target_core(canonical_prompt)
    kind = str(contract.get("required_kind") or "unknown")
    extent = str(contract.get("required_extent") or "unknown")
    return {
        "target_core": target,
        "target_kind": kind if kind in KINDS else "unknown",
        "number": (
            str(contract["required_number"])
            if contract.get("required_number") in NUMBERS
            else "unknown"
        ),
        "extent": extent if extent in SCOPES else "unknown",
        "owner": "",
        "selector": str(contract.get("required_selector") or ""),
        "grounding_status": "clear",
        "evidence": f"canonical target: {canonical_prompt}",
    }


def validate_visual_binding(
    value: dict[str, Any],
    contract: dict[str, str],
    canonical_prompt: str,
    query: str = "",
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(VISUAL_BINDING_KEYS):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"visual binding keys must be {sorted(VISUAL_BINDING_KEYS)}, "
            f"got {actual}"
        )
    binding = {
        "target_core": _clean_string(
            value["target_core"], "target_core", max_words=12
        ),
        "target_kind": _clean_string(
            value["target_kind"], "target_kind"
        ).casefold(),
        "number": _normalize_number(_clean_string(
            value["number"], "number"
        )),
        "extent": _clean_string(value["extent"], "extent").casefold(),
        "owner": _clean_string(
            value["owner"], "owner", allow_empty=True, max_words=12
        ),
        "selector": _clean_string(
            value["selector"], "selector", allow_empty=True, max_words=12
        ),
        "grounding_status": _clean_string(
            value["grounding_status"], "grounding_status"
        ).casefold(),
        "evidence": _clean_string(
            value["evidence"], "binding evidence", max_words=40
        ),
    }
    if binding["target_kind"] not in KINDS:
        binding["target_kind"] = "unknown"
    if binding["number"] not in NUMBERS:
        binding["number"] = "unknown"
    if binding["extent"] not in SCOPES:
        binding["extent"] = "unknown"
    if binding["grounding_status"] not in GROUNDING_STATUSES:
        binding["grounding_status"] = "ambiguous"

    required_number = contract.get("required_number")
    if required_number in {"singular", "plural"}:
        binding["number"] = str(required_number)
    elif _target_evidence_is_plural(
        binding["target_core"], binding["evidence"]
    ):
        binding["number"] = "plural"
    if (
        binding["extent"] == "unknown"
        and contract.get("required_extent") in SCOPES - {"unknown"}
    ):
        binding["extent"] = str(contract["required_extent"])

    canonical_stems = _token_stems(canonical_prompt)
    target_stems = _token_stems(binding["target_core"])
    selector_stems = _token_stems(binding["selector"])
    role_stems = _token_stems(contract.get("answer_role_phrase", ""))
    evidence_stems = _token_stems(
        " ".join(_content_words(binding["evidence"])[:3])
    )
    clear_swap = (
        binding["grounding_status"] == "clear"
        and bool(canonical_stems & (selector_stems | evidence_stems))
        and not bool(canonical_stems & target_stems)
        and not bool(role_stems & target_stems)
    )
    if clear_swap:
        return _canonical_binding(contract, canonical_prompt)
    return binding


def build_visual_binding(
    image_path: str,
    query: str,
    contract: dict[str, str],
    canonical_prompt: str,
    qwen: object,
    trace: Trace,
) -> dict[str, str]:
    resolved_image = Path(image_path).resolve()
    qwen.load_image(str(resolved_image))
    prompt = visual_binding_prompt(query, contract)
    raw = _qwen_raw(
        qwen,
        "visual_binding",
        prompt,
        trace,
        use_image=True,
        max_tokens=650,
    )
    try:
        return validate_visual_binding(
            parse_json_object(raw), contract, canonical_prompt, query
        )
    except ValueError as error:
        repair_prompt = (
            f"{prompt}\n\n"
            "Return a fresh complete object for the same locked slot. "
            f"VALIDATION_ERROR: {error}\nREJECTED_RESPONSE: {raw}"
        )
        repaired = _qwen_raw(
            qwen,
            "visual_binding_schema_repair",
            repair_prompt,
            trace,
            use_image=True,
            max_tokens=650,
        )
        # print(repair_prompt)
        # print(resolved_image)
        # sys.exit(0)
        try:
            return validate_visual_binding(
                parse_json_object(repaired), contract, canonical_prompt, query
            )
        except ValueError as repaired_error:
            if not canonical_prompt:
                raise
            trace.record(
                "visual_binding_fallback",
                "pipeline",
                error=str(repaired_error),
            )
            return _canonical_binding(contract, canonical_prompt)


def _binding_sam_prompt(binding: dict[str, str]) -> str:
    target = _clean_string(
        binding["target_core"], "binding target", max_words=12, max_chars=100
    )
    if binding.get("number") == "plural" and not re.match(
        r"^(?:all|both)\b", target, flags=re.IGNORECASE
    ):
        target = f"all {target}"

    owner = str(binding.get("owner") or "").strip()
    if binding.get("extent") in {"physical_part", "visible_region"} and owner:
        if not (_token_stems(owner) & _token_stems(target)):
            target = f"{owner} {target}"

    selector = str(binding.get("selector") or "").strip()
    selector_words = _WORD_RE.findall(selector)
    adjective_selectors = {
        "left", "right", "widest", "largest", "smallest", "outermost",
        "innermost", "central", "center", "upper", "lower",
    }
    prepositions = {"on", "in", "at", "under", "above", "near", "beside", "with"}
    if len(selector_words) == 1 and selector.casefold() in adjective_selectors:
        target = f"{selector} {target}"
    elif selector_words and selector_words[0].casefold() in prepositions:
        target = f"{target} {selector}"
    elif _token_stems(selector) & _token_stems(target):
        remaining = [
            word for word in selector_words
            if not (_token_stems(word) & _token_stems(target))
        ]
        if remaining and remaining[0].casefold() not in prepositions:
            target = f"{' '.join(remaining)} {target}"
    return " ".join(target.split()[:6])


def _prompt_key(value: str) -> str:
    normalized = " ".join(str(value).split()).casefold()
    return re.sub(r"^(?:the|a|an)\s+", "", normalized)

def _sam_prompt_key(value: str) -> str:
    return " ".join(str(value).split()).casefold()





def validate_target_plan(
    value: dict[str, Any],
    query: str,
    image_size: tuple[int, int] | None = None,
    contract: dict[str, str] | None = None,
    canonical_prompt: str = "",
    binding: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(PLAN_KEYS):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"target plan keys must be {sorted(PLAN_KEYS)}, got {actual}"
        )

    plan = {
        "full_description": _clean_string(
            value["full_description"], "full_description", max_words=60
        ),
        "target_entity": _clean_string(
            value["target_entity"], "target_entity", max_words=12
        ),
        "target_number": _normalize_number(_clean_string(
            value["target_number"], "target_number"
        )),
        "target_scope": _clean_string(
            value["target_scope"], "target_scope"
        ).casefold(),
        "selector": _clean_string(
            value["selector"], "selector", allow_empty=True, max_words=20
        ),
        "visible_discriminator": _clean_string(
            value["visible_discriminator"],
            "visible_discriminator",
            allow_empty=True,
            max_words=20,
        ),
        "evidence": _clean_truncated_words(
            value["evidence"], "evidence", 40
        ),
        "confidence": _clean_string(
            value["confidence"], "confidence"
        ).casefold(),
        "primary_prompt": _clean_string(
            value["primary_prompt"],
            "primary_prompt",
            max_words=12,
            max_chars=100,
        ),
        "alternate_prompt": _clean_string(
            value["alternate_prompt"],
            "alternate_prompt",
            allow_empty=True,
            max_words=12,
            max_chars=100,
        ),
        "bbox_xyxy_1000": _normalize_bbox(
            value["bbox_xyxy_1000"], image_size
        ),
    }
    if plan["target_number"] not in NUMBERS:
        raise ValueError("invalid target_number")
    plan["planner_target_number"] = plan["target_number"]
    if plan["target_scope"] not in SCOPES:
        plan["target_scope"] = "unknown"
    if plan["confidence"] not in CONFIDENCES:
        raise ValueError("invalid confidence")

    locked = contract or {}
    binding_value = binding or {
        "target_core": plan["target_entity"],
        "number": plan["target_number"],
        "extent": plan["target_scope"],
        "selector": plan["selector"],
        "evidence": plan["evidence"],
    }
    bound_prompt = _binding_sam_prompt(binding_value)
    planner_description = plan["full_description"]
    planner_alternate = plan["alternate_prompt"]
    planner_primary = plan["primary_prompt"]

    if (
        plan["target_scope"] == "unknown"
        and locked.get("required_extent") in SCOPES - {"unknown"}
    ):
        plan["target_scope"] = str(locked["required_extent"])
    if (
        not plan["selector"]
        and str(locked.get("required_selector") or "").strip()
    ):
        plan["selector"] = str(locked["required_selector"])

    target_head = _target_head(plan["target_entity"])
    description_words = {
        word.casefold() for word in _WORD_RE.findall(plan["full_description"])
    }
    binding_evidence = str(plan["evidence"])
    if _same_target_name(
        plan["target_entity"], str(binding_value.get("target_core") or "")
    ):
        binding_evidence = str(binding_value.get("evidence") or plan["evidence"])
    if target_head and target_head not in description_words:
        evidence_words = {
            word.casefold() for word in _WORD_RE.findall(binding_evidence)
        }
        plan["full_description"] = (
            binding_evidence
            if target_head in evidence_words
            else f"{plan['target_entity']}; {binding_evidence}"
        )
    if target_head not in {
        word.casefold() for word in _WORD_RE.findall(plan["evidence"])
    }:
        plan["evidence"] = binding_evidence

    canonical = (
        _clean_canonical_target(canonical_prompt)
        if canonical_prompt else bound_prompt
    )
    if locked.get("required_number") in {"singular", "plural"}:
        plan["target_number"] = str(locked["required_number"])
    elif (
        _target_number_hint(_phrase_target_core(canonical)) == "plural"
        or _quantified_target_is_plural(
            plan["target_entity"], plan["evidence"]
        )
        or _target_evidence_is_plural(
            plan["target_entity"], plan["evidence"]
        )
        or (
            binding_value.get("number") == "plural"
            and binding_value.get("grounding_status") == "clear"
            and _same_target_name(
                plan["target_entity"], binding_value["target_core"]
            )
        )
    ):
        plan["target_number"] = "plural"
    canonical_core = _phrase_target_core(canonical)
    canonical_head = _target_head(canonical_core)
    if _same_target_name(canonical_core, plan["target_entity"]):
        canonical_description = planner_description
    elif _same_target_name(canonical_core, binding_value["target_core"]):
        evidence_words = {
            word.casefold() for word in _WORD_RE.findall(binding_evidence)
        }
        canonical_description = (
            binding_evidence
            if canonical_head in evidence_words
            else canonical
        )
    else:
        canonical_description = canonical

    plan["canonical_target_entity"] = canonical_core
    plan["canonical_full_description"] = canonical_description
    plan["bound_prompt"] = bound_prompt
    plan["visual_binding"] = dict(binding_value)
    plan["primary_prompt"] = canonical
    plan["literal_prompt"] = canonical_core
    planner_words = _WORD_RE.findall(planner_primary)
    planner_first_three = " ".join(planner_words[:3])
    planner_target_first = bool(
        _token_stems(plan["target_entity"])
        & _token_stems(planner_first_three)
    )
    direct_match = re.search(
        r"\b(?:what|which)\s+([a-z][a-z-]*)\b",
        query.casefold(),
    )
    explicit_head = direct_match.group(1) if direct_match else ""
    generic_heads = {
        "animal", "area", "body", "entity", "feature", "item",
        "object", "part", "parts", "person", "place", "region", "thing",
    }
    planner_explicit_ok = (
        not explicit_head or explicit_head in generic_heads
        or explicit_head in _token_stems(plan["target_entity"])
    )
    plan["planner_prompt"] = (
        planner_primary if planner_target_first and planner_explicit_ok else ""
    )
    if _prompt_key(canonical) != _prompt_key(bound_prompt):
        plan["alternate_prompt"] = bound_prompt
    else:
        plan["alternate_prompt"] = planner_alternate

    if _prompt_key(plan["alternate_prompt"]) in {
        "",
        _prompt_key(plan["primary_prompt"]),
    }:
        plan["alternate_prompt"] = ""
    return plan


def build_target_plan(
    image_path: str,
    query: str,
    contract: dict[str, str],
    binding: dict[str, str],
    canonical_prompt: str,
    qwen: object,
    trace: Trace,
) -> dict[str, Any]:
    resolved_image = Path(image_path).resolve()
    with Image.open(resolved_image) as loaded:
        image_size = loaded.size
    qwen.load_image(str(resolved_image))
    bound_prompt = _binding_sam_prompt(binding)
    prompt = target_plan_prompt(query, contract, bound_prompt, canonical_prompt)
    # print(prompt)
    raw = _qwen_raw(
        qwen,
        "target_plan",
        prompt,
        trace,
        use_image=True,
        max_tokens=900,
    ) # 900
    # print(f"end: {str(resolved_image)}")
    # print(raw)
    # sys.exit(0)
    try:
        return validate_target_plan(
            _parse_target_plan_object(raw), query, image_size=image_size,
            contract=contract, canonical_prompt=canonical_prompt,
            binding=binding,
        )
    except ValueError as error:
        repair_prompt = (
            f"{prompt}\n\n"
            "The previous response failed structural validation. Return a fresh "
            "complete object satisfying the same task and exact schema.\n"
            f"VALIDATION_ERROR: {error}\n"
            f"REJECTED_RESPONSE: {raw}"
        )
        repaired = _qwen_raw(
            qwen,
            "target_plan_schema_repair",
            repair_prompt,
            trace,
            use_image=True,
            max_tokens=900,
        )
        used_wrapper = False
        used_bbox_tail_fallback = False
        try:
            repaired_payload = _repaired_json_payload(repaired)
            used_wrapper = repaired_payload != str(repaired).strip()
            repaired_value = _parse_target_plan_object(repaired_payload)
        except ValueError:
            fallback_value = _target_plan_without_malformed_bbox_tail(
                repaired
            )
            if fallback_value is None:
                raise
            repaired_value = fallback_value
            used_bbox_tail_fallback = True
        try:
            validated = validate_target_plan(
                repaired_value, query, image_size=image_size,
                contract=contract, canonical_prompt=canonical_prompt,
                binding=binding,
            )
        except ValueError as repaired_error:
            cleaned_value = _drop_coordinate_key_noise(repaired_value)
            if cleaned_value is None:
                raise repaired_error
            validated = validate_target_plan(
                cleaned_value, query, image_size=image_size,
                contract=contract, canonical_prompt=canonical_prompt,
                binding=binding,
            )
            trace.record(
                "target_plan_coordinate_key_fallback",
                "pipeline",
                removed_keys=sorted(set(repaired_value) - set(PLAN_KEYS)),
            )
        if used_wrapper:
            trace.record("target_plan_wrapper_fallback", "pipeline")
        if used_bbox_tail_fallback:
            trace.record("target_plan_bbox_tail_fallback", "pipeline")
        return validated




def _binary_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.shape != shape:
        raise ValueError(
            f"mask shape mismatch: expected {shape}, got {value.shape}"
        )
    return value.astype(bool, copy=False)


def _ranked_masks(
    results: list[dict[str, Any]],
    shape: tuple[int, int],
    trace: Trace,
    role: str,
) -> list[tuple[np.ndarray, float]]:
    ranked: list[tuple[float, int, np.ndarray]] = []
    for index, item in enumerate(results):
        try:
            mask = _binary_mask(item["mask"], shape)
            confidence = float(item.get("conf", 0.0))
        except (KeyError, TypeError, ValueError) as error:
            trace.record(
                f"{role}_candidate_rejected",
                "pipeline",
                index=index,
                error=str(error),
            )
            continue
        if not mask.any():
            continue
        ranked.append((-confidence, index, mask))
    ranked.sort(key=lambda packed: (packed[0], packed[1]))

    unique: list[tuple[np.ndarray, float]] = []
    seen: set[str] = set()
    for negative_confidence, _, mask in ranked:
        digest = _mask_sha256(mask)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append((mask, -negative_confidence))
        if len(unique) == 4:
            break
    return unique


def _text_variants(
    sam3: object,
    prompt: str,
    shape: tuple[int, int],
    source: str,
    trace: Trace,
) -> list[Candidate]:
    results = list(sam3.predict_text(prompt) or [])
    trace.sam3_call_count += 1
    trace.record(
        f"{source}_sam",
        "sam3",
        prompt=prompt,
        raw_candidate_count=len(results),
    )
    ranked = _ranked_masks(results, shape, trace, source)
    if not ranked:
        return []

    first_mask, first_confidence = ranked[0]
    candidates = [Candidate(
        prompt=prompt,
        mask=first_mask.copy(),
        source=source,
        variant="top1",
        confidence=first_confidence,
    )]
    if len(ranked) >= 2:
        union2 = ranked[0][0] | ranked[1][0]
        candidates.append(Candidate(
            prompt=prompt,
            mask=union2,
            source=source,
            variant="top2_union",
            confidence=min(ranked[0][1], ranked[1][1]),
        ))
    if len(ranked) >= 3:
        union_all = np.logical_or.reduce([item[0] for item in ranked])
        candidates.append(Candidate(
            prompt=prompt,
            mask=union_all,
            source=source,
            variant="all_union",
            confidence=min(item[1] for item in ranked),
        ))
    return candidates


def _bbox_pixels(
    bbox_1000: list[float] | None,
    width: int,
    height: int,
) -> list[int] | None:
    bbox = _normalize_bbox(bbox_1000)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    packed = [
        int(round(x1 * width / 1000.0)),
        int(round(y1 * height / 1000.0)),
        int(round(x2 * width / 1000.0)),
        int(round(y2 * height / 1000.0)),
    ]
    packed[0] = min(width - 1, max(0, packed[0]))
    packed[1] = min(height - 1, max(0, packed[1]))
    packed[2] = min(width, max(packed[0] + 1, packed[2]))
    packed[3] = min(height, max(packed[1] + 1, packed[3]))
    return packed


def _bbox_variants(
    sam3: object,
    bbox: list[int],
    shape: tuple[int, int],
    trace: Trace,
) -> list[Candidate]:
    results = list(sam3.predict_box(bbox) or [])
    trace.sam3_call_count += 1
    trace.record(
        "bbox_sam",
        "sam3",
        bbox_xyxy=bbox,
        raw_candidate_count=len(results),
    )
    ranked = _ranked_masks(results, shape, trace, "bbox")
    if not ranked:
        return []

    first_mask, first_confidence = ranked[0]
    metadata = {"bbox_xyxy": list(bbox)}
    candidates = [Candidate(
        prompt=None,
        mask=first_mask.copy(),
        source="bbox",
        variant="top1",
        confidence=first_confidence,
        metadata=dict(metadata),
    )]
    if len(ranked) >= 2:
        candidates.append(Candidate(
            prompt=None,
            mask=ranked[0][0] | ranked[1][0],
            source="bbox",
            variant="top2_union",
            confidence=min(ranked[0][1], ranked[1][1]),
            metadata=dict(metadata),
        ))
    return candidates


def build_candidates(
    image_path: str,
    plan: dict[str, Any],
    sam3: object,
    trace: Trace,
) -> list[Candidate]:
    image = Path(image_path).resolve()
    with Image.open(image) as loaded:
        width, height = loaded.size
    shape = (height, width)
    sam3.load_image(str(image))

    candidates: list[Candidate] = []
    seen_prompts: set[str] = set()
    for source, field_name in (
        ("primary", "primary_prompt"),
        ("literal", "literal_prompt"),
        ("alternate", "alternate_prompt"),
        ("planner", "planner_prompt"),
    ):
        if source == "literal" and candidates:
            continue
        prompt = " ".join(str(plan.get(field_name) or "").split()).strip()
        key = _sam_prompt_key(prompt)
        if not prompt or key in seen_prompts:
            continue
        seen_prompts.add(key)
        candidates.extend(
            _text_variants(sam3, prompt, shape, source, trace)
        )

    alternate_prompt = str(plan.get("alternate_prompt") or "").strip()
    binding_target = str(
        (plan.get("visual_binding") or {}).get("target_core") or ""
    ).strip()
    binding_key = _sam_prompt_key(binding_target)
    if (
        alternate_prompt
        and not any(item.source == "alternate" for item in candidates)
        and binding_target
        and binding_key not in seen_prompts
    ):
        seen_prompts.add(binding_key)
        candidates.extend(
            _text_variants(
                sam3, binding_target, shape, "alternate", trace
            )
        )

    bbox = _bbox_pixels(plan.get("bbox_xyxy_1000"), width, height)
    if bbox is not None:
        candidates.extend(_bbox_variants(sam3, bbox, shape, trace))

    unique: list[Candidate] = []
    seen_masks: set[tuple[str, tuple[Any, ...]]] = set()
    for candidate in candidates:
        digest = _mask_sha256(candidate.mask)
        identity = (
            digest,
            _target_claim_key(_candidate_target_claim(candidate, plan)),
        )
        if identity in seen_masks:
            trace.record(
                "candidate_duplicate",
                "pipeline",
                source=candidate.source,
                variant=candidate.variant,
                mask_sha256=digest,
            )
            continue
        seen_masks.add(identity)
        candidate.metadata["mask_sha256"] = digest
        unique.append(candidate)

    for index, candidate in enumerate(unique):
        candidate.metadata["label"] = chr(ord("A") + index)
    _candidate_target_claims(unique, plan)
    trace.record(
        "candidate_pool",
        "pipeline",
        count=len(unique),
        candidates=[
            {
                "label": item.metadata["label"],
                "source": item.source,
                "variant": item.variant,
                "mask_sha256": item.metadata["mask_sha256"],
            }
            for item in unique
        ],
    )
    return unique


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for row in range(3):
        for column in range(3):
            eroded &= padded[
                row:row + mask.shape[0],
                column:column + mask.shape[1],
            ]
    return mask & ~eroded


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    value = rgb.astype(np.float32).copy()
    value[mask] = (
        0.55 * value[mask]
        + 0.45 * np.array([20, 230, 80], dtype=np.float32)
    )
    value[_mask_boundary(mask)] = np.array(
        [255, 0, 220], dtype=np.float32
    )
    return np.clip(value, 0, 255).astype(np.uint8)


def _fit_panel(
    pixels: np.ndarray,
    size: tuple[int, int],
    background: tuple[int, int, int] = (28, 28, 28),
) -> Image.Image:
    source = Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB")
    source.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, background)
    left = (size[0] - source.width) // 2
    top = (size[1] - source.height) // 2
    panel.paste(source, (left, top))
    return panel


def _tight_crop(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows, columns = np.where(mask)
    if not len(rows):
        return pixels
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    x1, x2 = int(columns.min()), int(columns.max()) + 1
    pad = max(4, int(round(max(y2 - y1, x2 - x1) * 0.2)))
    y1, y2 = max(0, y1 - pad), min(mask.shape[0], y2 + pad)
    x1, x2 = max(0, x1 - pad), min(mask.shape[1], x2 + pad)
    return pixels[y1:y2, x1:x2]


def render_candidate_board(
    image_path: str,
    candidates: list[Candidate],
    output_path: Path,
) -> Path:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    panel = 360
    tile_width = panel * 2
    tile_height = panel
    columns = 2
    rows = max(1, math.ceil(len(candidates) / columns))
    board = Image.new(
        "RGB", (columns * tile_width, rows * tile_height), (235, 235, 235)
    )

    for index, candidate in enumerate(candidates):
        mask = _binary_mask(candidate.mask, rgb.shape[:2])
        overlay = _overlay(rgb, mask)
        isolated = np.full_like(rgb, 18)
        isolated[mask] = rgb[mask]
        isolated[_mask_boundary(mask)] = np.array([255, 0, 220], dtype=np.uint8)

        left = _fit_panel(overlay, (panel, panel))
        zoom = _fit_panel(
            _tight_crop(overlay, mask), (panel, panel // 2)
        )
        isolated_panel = _fit_panel(isolated, (panel, panel // 2))
        tile = Image.new("RGB", (tile_width, tile_height), (20, 20, 20))
        tile.paste(left, (0, 0))
        tile.paste(zoom, (panel, 0))
        tile.paste(isolated_panel, (panel, panel // 2))

        label = str(candidate.metadata["label"])
        draw = ImageDraw.Draw(tile)
        draw.rectangle((8, 8, 62, 48), fill=(0, 0, 0))
        draw.text((27, 17), label, fill=(255, 255, 255))

        column = index % columns
        row = index // columns
        board.paste(tile, (column * tile_width, row * tile_height))

    _save_image(board, output_path)
    return output_path.resolve()


def _parse_observation_response(text: str) -> dict[str, Any]:
    try:
        return parse_json_object(text)
    except ValueError as object_error:
        raw = str(text).strip()
        if not raw.startswith("[") or not raw.endswith("]"):
            raise object_error

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate key: {key}")
                value[key] = item
            return value

        try:
            items = json.loads(raw, object_pairs_hook=unique_object)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid observation JSON: {error}") from error
        if not isinstance(items, list):
            raise object_error
        return {"observations": items}


def _validate_observations(
    value: dict[str, Any],
    labels: list[str],
) -> list[dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {"observations"}:
        raise ValueError("observation response must contain only observations")
    items = value["observations"]
    if (
        not isinstance(items, list)
        or not items
        or len(items) > len(labels)
    ):
        raise ValueError("observation count does not match candidate labels")

    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != set(OBSERVATION_KEYS):
            raise ValueError("invalid observation schema")
        record: dict[str, str] = {}
        for key in OBSERVATION_KEYS:
            record[key] = _clean_string(
                item[key],
                f"observation.{key}",
                allow_empty=(key == "parent_entity"),
                max_words=20 if key == "evidence" else 12,
            )
        record["covered_scope"] = record["covered_scope"].casefold()
        record["covered_number"] = _normalize_number(record["covered_number"])
        record["coverage"] = record["coverage"].casefold()
        record["uncovered_equivalents"] = (
            record["uncovered_equivalents"].casefold()
        )
        if record["covered_scope"] not in SCOPES:
            raise ValueError("invalid observed scope")
        if record["covered_number"] not in NUMBERS:
            raise ValueError("invalid observed number")
        if record["coverage"] not in COVERAGES:
            raise ValueError("invalid observed coverage")
        if record["uncovered_equivalents"] not in YES_NO_UNKNOWN:
            raise ValueError("invalid uncovered_equivalents")
        normalized.append(record)

    returned_labels = [item["label"] for item in normalized]
    if len(set(returned_labels)) != len(returned_labels):
        raise ValueError("duplicate observation label")
    if not set(returned_labels).issubset(labels):
        raise ValueError("observation labels do not match candidate labels")
    by_label = {item["label"]: item for item in normalized}
    for label in labels:
        by_label.setdefault(label, {
            "label": label,
            "covered_entity": "unknown",
            "covered_scope": "unknown",
            "covered_number": "unknown",
            "coverage": "unknown",
            "uncovered_equivalents": "unknown",
            "parent_entity": "",
            "evidence": "observer omitted this label",
        })
    return [by_label[label] for label in labels]


def observe_candidates(
    image_path: str,
    candidates: list[Candidate],
    qwen: object,
    trace: Trace,
    board_path: Path,
) -> tuple[list[dict[str, str]] | None, Path]:
    labels = [str(item.metadata["label"]) for item in candidates]
    rendered = render_candidate_board(image_path, candidates, board_path)
    qwen.load_image(str(rendered))
    prompt = observation_prompt(labels)
    raw = _qwen_raw(
        qwen,
        "mask_observe_batch",
        prompt,
        trace,
        use_image=True,
        max_tokens=min(1800, 300 + 160 * len(labels)),
    )
    try:
        observations = _validate_observations(
            _parse_observation_response(raw), labels
        )
    except ValueError as error:
        trace.record(
            "mask_observe_batch_invalid",
            "pipeline",
            error=str(error),
        )
        return None, rendered
    return observations, rendered


def _fallback_candidate(
    candidates: list[Candidate],
    target_number: str,
    preferred_source: str = "",
) -> Candidate | None:
    if not candidates:
        return None

    def source_choice(source: str) -> Candidate | None:
        matches = [item for item in candidates if item.source == source]
        if not matches:
            return None
        order = (
            ("all_union", "top2_union", "top1")
            if target_number == "plural"
            else ("top1", "top2_union", "all_union")
        )
        for variant in order:
            for item in matches:
                if item.variant == variant:
                    return item
        return matches[0]

    if target_number == "plural":
        source_order = dict.fromkeys((
            preferred_source, "primary", "literal", "alternate", "planner",
            "bbox",
        ))
        for variant in ("all_union", "top2_union", "top1"):
            for source in source_order:
                match = next((
                    item for item in candidates
                    if item.source == source and item.variant == variant
                ), None)
                if match is not None:
                    return match
    preferred = source_choice(preferred_source) if preferred_source else None
    return (
        preferred
        or source_choice("primary")
        or source_choice("literal")
        or source_choice("alternate")
        or source_choice("planner")
        or source_choice("bbox")
        or candidates[0]
    )


def _mask_box_1000(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.where(mask)
    if not len(rows):
        return None
    height, width = mask.shape
    return [
        int(round(columns.min() * 1000 / width)),
        int(round(rows.min() * 1000 / height)),
        int(round((columns.max() + 1) * 1000 / width)),
        int(round((rows.max() + 1) * 1000 / height)),
    ]


def _geometry(candidates: list[Candidate]) -> dict[str, Any]:
    total = int(candidates[0].mask.size)
    packed: dict[str, Any] = {"candidates": {}}
    for item in candidates:
        label = str(item.metadata["label"])
        mask = item.mask
        area = int(mask.sum())
        packed["candidates"][label] = {
            "area_fraction": round(area / total, 6),
            "mask_bounds_1000": _mask_box_1000(mask),
        }
    return packed


def _target_centered_text(target: str, evidence: str) -> str:
    target = " ".join(str(target).split()).strip()
    evidence = " ".join(str(evidence).split()).strip()
    if not evidence:
        return target
    target_words = _content_words(target)
    evidence_words = _content_words(evidence)
    if not target_words or not any(
        _word_forms(target_words[-1]) & _word_forms(word)
        for word in evidence_words
    ):
        return target
    centered = bool(target_words) and len(evidence_words) >= len(target_words) and all(
        _word_forms(expected) & _word_forms(observed)
        for expected, observed in zip(target_words, evidence_words)
    )
    return evidence if centered else f"{target}; {evidence}"


def _candidate_target_claim(
    candidate: Candidate,
    plan: dict[str, Any],
) -> dict[str, Any]:
    binding = dict(plan.get("visual_binding") or {})
    canonical = str(
        plan.get("canonical_target_entity") or plan.get("target_entity") or ""
    )
    plan_target = str(plan.get("target_entity") or canonical)
    plan_stems = _token_stems(plan_target)
    binding_target = str(binding.get("target_core") or "")
    binding_stems = _token_stems(binding_target)
    bound_candidate = (
        candidate.source == "alternate"
        and (
            _prompt_key(candidate.prompt or "")
            == _prompt_key(plan.get("bound_prompt", ""))
            or _same_target_name(candidate.prompt or "", binding_target)
        )
    )

    if candidate.source in {"primary", "literal"}:
        target = canonical
        matches_binding = bool(binding_target) and _same_target_name(
            canonical, binding_target
        )
        matches_plan = _same_target_name(canonical, plan_target)
        number_hint = _target_number_hint(canonical)
        binding_number = str(binding.get("number") or "unknown")
        planner_number = str(
            plan["planner_target_number"]
            if "planner_target_number" in plan
            else plan.get("target_number") or "unknown"
        )
        if number_hint is not None:
            number = number_hint
        elif (
            matches_binding
            and binding_number in {"singular", "plural"}
        ):
            number = binding_number
        elif matches_plan and planner_number in NUMBERS:
            number = planner_number
        else:
            number = "unknown"
        if matches_binding:
            scope = str(binding.get("extent") or "unknown")
            owner = str(binding.get("owner") or "")
            selector = str(binding.get("selector") or "")
            discriminator = selector
        elif matches_plan:
            scope = str(plan.get("target_scope") or "unknown")
            owner = ""
            selector = str(plan.get("selector") or "")
            discriminator = str(plan.get("visible_discriminator") or "")
        else:
            scope = "unknown"
            owner = ""
            selector = ""
            discriminator = ""
        evidence = str(plan.get("canonical_full_description") or target)
        text = _target_centered_text(target, evidence)
    elif bound_candidate:
        target = binding_target
        number = str(binding.get("number") or "unknown")
        scope = str(binding.get("extent") or "unknown")
        owner = str(binding.get("owner") or "")
        selector = str(binding.get("selector") or "")
        discriminator = selector
        evidence = str(binding.get("evidence") or target)
        text = _target_centered_text(target, evidence)
    else:
        target = plan_target
        producer_number = str(
            plan.get("planner_target_number") or plan.get("target_number")
            or "unknown"
        )
        number = producer_number if producer_number in NUMBERS else str(
            plan.get("target_number") or "unknown"
        )
        scope = str(plan.get("target_scope") or "unknown")
        owner = str(
            binding.get("owner") or ""
            if plan_stems & binding_stems else ""
        )
        selector = str(plan.get("selector") or "")
        discriminator = str(plan.get("visible_discriminator") or "")
        evidence = str(plan.get("evidence") or target)
        text = _target_centered_text(
            target, str(plan.get("full_description") or evidence)
        )
    if _target_centered_text(target, evidence) == target:
        evidence = target

    public = {
        "target_entity": target,
        "target_number": number,
        "target_scope": scope,
        "owner": owner,
        "selector": selector,
    }
    return {
        **public,
        "_display": {
            "text": text,
            "target_entity": target,
            "number": number,
            "scope": scope,
            "selector": selector,
            "visible_discriminator": discriminator,
            "evidence": evidence,
        },
    }


def _candidate_target_claims(
    candidates: list[Candidate],
    plan: dict[str, Any],
) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    for candidate in candidates:
        claim = _candidate_target_claim(candidate, plan)
        candidate.metadata["target_claim"] = claim
        claims.append({
            "label": str(candidate.metadata["label"]),
            "target_entity": str(claim["target_entity"]),
            "target_number": str(claim["target_number"]),
            "target_scope": str(claim["target_scope"]),
            "owner": str(claim["owner"]),
            "selector": str(claim["selector"]),
        })
    return claims


def _target_claim_key(claim: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(sorted(_token_stems(str(claim.get("target_entity") or "")))),
        str(claim.get("target_number") or "unknown").casefold(),
        str(claim.get("target_scope") or "unknown").casefold(),
        _prompt_key(str(claim.get("owner") or "")),
        _prompt_key(str(claim.get("selector") or "")),
    )


def _binding_name_lock(
    query: str,
    contract: dict[str, str],
    plan: dict[str, Any],
    selected: Candidate,
    candidates: list[Candidate],
    observations: list[dict[str, str]],
) -> Candidate | None:
    binding = dict(plan.get("visual_binding") or {})
    required_extent = str(contract.get("required_extent") or "unknown")
    binding_extent = str(binding.get("extent") or "unknown")
    binding_name = str(binding.get("target_core") or "").strip()
    if not binding_name:
        return None
    binding_core = _phrase_target_core(binding_name)
    core_stems = _token_stems(binding_core)
    selected_name = _phrase_target_core(str(
        _candidate_target_claim(selected, plan).get("target_entity") or ""
    ))
    if (
        required_extent == "unknown"
        or binding_extent != required_extent
        or not core_stems
        or not selected_name
        or _same_target_name(selected_name, binding_core)
    ):
        return None

    status = str(binding.get("grounding_status") or "")
    if status == "ambiguous":
        owner_stems = _token_stems(str(binding.get("owner") or ""))
        if (
            contract.get("required_kind") != binding.get("target_kind")
            or not (
                core_stems
                & _token_stems(str(contract.get("answer_role_phrase") or ""))
            )
            or not (
                core_stems & _token_stems(str(plan.get("selector") or ""))
            )
            or not owner_stems
            or not (owner_stems & _token_stems(query))
        ):
            return None
    elif status != "clear":
        return None

    related = []
    for candidate in candidates:
        if candidate.prompt is None:
            continue
        claim_name = str(
            _candidate_target_claim(candidate, plan).get("target_entity") or ""
        )
        prompt_core = _phrase_target_core(candidate.prompt)
        if (
            (
                _same_target_name(claim_name, binding_core)
                and bool(core_stems & _token_stems(candidate.prompt))
            )
            or _same_target_name(prompt_core, binding_core)
        ):
            related.append(candidate)
    if len(related) != 1 or related[0].source != "alternate":
        return None

    candidate = related[0]
    candidate_observation = next((
        item for item in observations
        if item.get("label") == str(candidate.metadata["label"])
    ), {})
    if (
        candidate_observation.get("coverage") != "complete"
        or candidate_observation.get("uncovered_equivalents") != "no"
    ):
        return None
    if candidate is selected:
        return None
    union = int(np.logical_or(selected.mask, candidate.mask).sum())
    if not union:
        return None
    intersection = int(np.logical_and(selected.mask, candidate.mask).sum())
    return candidate if intersection / union < 0.75 else None



def _physical_part_union_completion(
    selected: Candidate,
    candidates: list[Candidate],
    observations: list[dict[str, str]],
    contract: dict[str, str],
    plan: dict[str, Any],
) -> Candidate | None:
    if (
        selected.source not in {"primary", "literal", "alternate", "planner"}
        or selected.variant != "top1"
        or contract.get("required_extent") != "physical_part"
        or contract.get("required_number") == "singular"
        or str(contract.get("required_selector") or "").strip()
        or selected.confidence <= 0
    ):
        return None

    by_label = {item["label"]: item for item in observations}
    selected_observation = by_label.get(
        str(selected.metadata["label"]), {}
    )
    if (
        selected_observation.get("coverage") != "complete"
        or selected_observation.get("uncovered_equivalents") != "no"
    ):
        return None

    match_keys = (
        "covered_entity", "covered_scope", "covered_number", "coverage",
        "uncovered_equivalents",
    )
    selected_area = int(selected.mask.sum())
    selected_name = str(
        _candidate_target_claim(selected, plan).get("target_entity") or ""
    )
    unions = sorted(
        (
            item for item in candidates
            if item.source == selected.source
            and item.variant in {"top2_union", "all_union"}
        ),
        key=lambda item: int(item.mask.sum()),
        reverse=True,
    )
    for candidate in unions:
        candidate_area = int(candidate.mask.sum())
        candidate_name = str(
            _candidate_target_claim(candidate, plan).get("target_entity") or ""
        )
        observation = by_label.get(str(candidate.metadata["label"]), {})
        if (
            not _same_target_name(selected_name, candidate_name)
            or candidate_area < 1.2 * selected_area
            or candidate.confidence < 0.97 * selected.confidence
            or any(
                str(observation.get(key, "")).strip().casefold()
                != str(selected_observation.get(key, "")).strip().casefold()
                for key in match_keys
            )
        ):
            continue
        return candidate
    return None



def _whole_object_bbox_superset(
    selected: Candidate,
    candidates: list[Candidate],
    observations: list[dict[str, str]],
    plan: dict[str, Any],
) -> Candidate | None:
    if (
        plan.get("target_scope") != "whole_object"
        or selected.source != "planner"
        or selected.variant != "top1"
    ):
        return None
    by_label = {item["label"]: item for item in observations}
    selected_label = str(selected.metadata["label"])
    selected_observation = by_label.get(selected_label, {})
    if (
        selected_observation.get("covered_scope") != "whole_object"
        or selected_observation.get("coverage") != "complete"
        or selected_observation.get("uncovered_equivalents") != "no"
    ):
        return None
    selected_area = int(selected.mask.sum())
    if not selected_area:
        return None
    match_keys = (
        "covered_entity", "covered_scope", "covered_number", "coverage",
        "uncovered_equivalents",
    )
    for candidate in candidates:
        if candidate.source != "bbox" or candidate.variant != "top1":
            continue
        observation = by_label.get(str(candidate.metadata["label"]), {})
        if any(
            observation.get(key) != selected_observation.get(key)
            for key in match_keys
        ):
            continue
        intersection = int(
            np.logical_and(selected.mask, candidate.mask).sum()
        )
        if (
            intersection / selected_area >= 0.995
            and int(candidate.mask.sum()) >= 1.2 * selected_area
        ):
            return candidate
    return None


def _whole_object_union_completion(
    selected: Candidate,
    candidates: list[Candidate],
    observations: list[dict[str, str]],
    contract: dict[str, str],
    plan: dict[str, Any],
) -> Candidate | None:
    if (
        plan.get("target_scope") != "whole_object"
        or selected.source == "bbox"
        or selected.variant != "top1"
    ):
        return None

    selected_area = int(selected.mask.sum())
    if not selected_area:
        return None
    binding_extent = str(
        (plan.get("visual_binding") or {}).get("extent") or ""
    )
    if (
        binding_extent != "whole_object"
        and selected_area / selected.mask.size >= 0.001
    ):
        return None

    by_label = {item["label"]: item for item in observations}
    selected_observation = by_label.get(
        str(selected.metadata["label"]), {}
    )
    if (
        plan.get("target_number") == "singular"
        and selected_observation.get("covered_number") == "plural"
    ):
        return None
    if (
        selected_observation.get("coverage") != "complete"
        or selected_observation.get("uncovered_equivalents") != "no"
    ):
        return None

    match_keys = (
        "covered_entity", "covered_scope", "covered_number", "coverage",
        "uncovered_equivalents",
    )
    binding = plan.get("visual_binding") or {}
    selected_claim = _target_claim_key(
        _candidate_target_claim(selected, plan)
    )
    selected_prompt = _prompt_key(selected.prompt or "")
    confidence_plateau = (
        contract.get("required_extent") == "whole_object"
        and not str(contract.get("required_selector") or "").strip()
        and binding.get("extent") == "whole_object"
        and binding.get("grounding_status") == "clear"
        and selected_observation.get("covered_scope") == "whole_object"
        and selected.confidence > 0
    )
    unions = sorted(
        (
            item for item in candidates
            if item.source == selected.source
            and item.variant in {"top2_union", "all_union"}
        ),
        key=lambda item: int(item.mask.sum()),
        reverse=True,
    )
    for candidate in unions:
        candidate_area = int(candidate.mask.sum())
        observation = by_label.get(
            str(candidate.metadata["label"]), {}
        )
        observation_matches = all(
            str(observation.get(key, "")).casefold()
            == str(selected_observation.get(key, "")).casefold()
            for key in match_keys
        )
        legacy_completion = (
            candidate_area >= 3 * selected_area and observation_matches
        )
        observation_omitted = (
            all(
                str(observation.get(key, "")).casefold() == "unknown"
                for key in match_keys
            )
            and str(observation.get("evidence", "")).casefold()
            == "observer omitted this label"
        )
        plateau_completion = (
            confidence_plateau
            and _prompt_key(candidate.prompt or "") == selected_prompt
            and _target_claim_key(_candidate_target_claim(candidate, plan))
            == selected_claim
            and candidate.confidence + 1 / 256
            >= 0.97 * selected.confidence
            and (
                observation_matches and candidate_area >= 2 * selected_area
                or observation_omitted
                and candidate_area >= 3 * selected_area
            )
        )
        if legacy_completion or plateau_completion:
            return candidate
    return None


def _whole_object_large_alternate(
    selected: Candidate,
    candidates: list[Candidate],
    observations: list[dict[str, str]],
    plan: dict[str, Any],
) -> Candidate | None:
    binding = plan.get("visual_binding") or {}
    if (
        plan.get("target_scope") != "whole_object"
        or binding.get("extent") != "whole_object"
        or binding.get("grounding_status") != "clear"
        or selected.source not in {"primary", "literal"}
        or selected.variant != "top1"
    ):
        return None

    selected_area = int(selected.mask.sum())
    if not selected_area:
        return None
    by_label = {item["label"]: item for item in observations}
    selected_observation = by_label.get(
        str(selected.metadata["label"]), {}
    )
    if (
        selected_observation.get("coverage") != "complete"
        or selected_observation.get("uncovered_equivalents") != "no"
    ):
        return None

    match_keys = (
        "covered_entity", "covered_scope", "covered_number", "coverage",
        "uncovered_equivalents",
    )
    alternates = sorted(
        (
            item for item in candidates
            if item.source == "alternate" and item.variant == "top1"
        ),
        key=lambda item: int(item.mask.sum()),
        reverse=True,
    )
    selected_claim = _target_claim_key(
        _candidate_target_claim(selected, plan)
    )
    for candidate in alternates:
        if (
            _target_claim_key(_candidate_target_claim(candidate, plan))
            != selected_claim
            or int(candidate.mask.sum()) < 2 * selected_area
        ):
            continue
        observation = by_label.get(
            str(candidate.metadata["label"]), {}
        )
        if all(
            str(observation.get(key, "")).casefold()
            == str(selected_observation.get(key, "")).casefold()
            for key in match_keys
        ):
            return candidate
    return None


def select_candidate(
    query: str,
    _query_contract: dict[str, str],
    plan: dict[str, Any],
    observations: list[dict[str, str]] | None,
    candidates: list[Candidate],
    qwen: object,
    trace: Trace,
    board_path: Path,
) -> tuple[Candidate, dict[str, Any]]:
    default = _fallback_candidate(candidates, str(plan["target_number"]))
    if default is None:
        raise ValueError("cannot select from an empty candidate list")
    default_label = str(default.metadata["label"])

    if observations is None:
        return default, {
            "label": default_label,
            "reason": "invalid anonymous observation; conservative fallback",
            "fallback": True,
        }

    target_claims = _candidate_target_claims(candidates, plan)
    prompt = selection_prompt(
        query,
        observations,
        target_claims,
        _geometry(candidates),
    )
    qwen.load_image(str(Path(board_path).resolve()))
    raw = _qwen_raw(
        qwen,
        "mask_select",
        prompt,
        trace,
        use_image=True,
        max_tokens=400,
    )
    try:
        value = parse_json_object(raw)
        if set(value) != set(SELECTION_KEYS):
            raise ValueError("selection schema mismatch")
        choice = _clean_string(value["choice"], "choice").upper()
        reason = _clean_string(value["reason"], "reason", max_words=40)
        by_label = {
            str(item.metadata["label"]): item for item in candidates
        }
        if choice == "NONE" or choice not in by_label:
            raise ValueError("selection did not name a valid candidate")
        selected = by_label[choice]
        selected_observation = next(
            (item for item in observations if item.get("label") == choice),
            {},
        )
        by_observation = {item["label"]: item for item in observations}
        default_observation = by_observation.get(default_label, {})
        default_claim = _candidate_target_claim(default, plan)
        selected_claim = _candidate_target_claim(selected, plan)
        locked_candidate = _binding_name_lock(
            query, _query_contract, plan, selected, candidates, observations
        )
        if locked_candidate is not None:
            label = str(locked_candidate.metadata["label"])
            trace.record(
                "binding_name_lock",
                "pipeline",
                requested_choice=choice,
                selected_label=label,
            )
            return locked_candidate, {
                "label": label,
                "reason": "unique grounded target name overrides unrelated claim",
                "fallback": True,
            }

        if (
            selected is not default
            and plan.get("target_number") == "plural"
            and default.variant in {"all_union", "top2_union"}
            and selected.variant == "top1"
            and _same_target_name(
                str(default_claim.get("target_entity") or ""),
                str(selected_claim.get("target_entity") or ""),
            )
            and not (
                default_observation.get("coverage") == "overinclusive"
                and selected_observation.get("coverage") == "complete"
                and selected_observation.get(
                    "uncovered_equivalents"
                ) == "no"
            )
        ):
            trace.record(
                "plural_single_switch_blocked", "pipeline",
                requested_choice=choice,
            )
            return default, {
                "label": default_label,
                "reason": "single-instance switch contradicted plural target",
                "fallback": True,
            }
        tied_fields = (
            "covered_entity", "covered_scope", "covered_number",
            "coverage", "uncovered_equivalents",
        )
        if (
            selected is not default
            and default.source in {"primary", "literal"}
            and selected.source != default.source
            and _target_claim_key(default_claim)
            == _target_claim_key(selected_claim)
            and default_observation.get("coverage") == "complete"
            and default_observation.get("uncovered_equivalents") == "no"
            and all(
                str(default_observation.get(field, "")).casefold()
                == str(selected_observation.get(field, "")).casefold()
                for field in tied_fields
            )
        ):
            trace.record(
                "primary_switch_blocked",
                "pipeline",
                requested_choice=choice,
            )
            return default, {
                "label": default_label,
                "reason": "non-primary switch lacked a decisive primary failure",
                "fallback": True,
            }
        part_union = _physical_part_union_completion(
            selected, candidates, observations, _query_contract, plan
        )
        if part_union is not None:
            label = str(part_union.metadata["label"])
            trace.record(
                "physical_part_union_completion_override",
                "pipeline",
                requested_choice=choice,
                selected_label=label,
            )
            return part_union, {
                "label": label,
                "reason": "equivalent union completes the physical-part set",
                "fallback": True,
            }

        superset = _whole_object_bbox_superset(
            selected, candidates, observations, plan
        )
        if superset is not None:
            label = str(superset.metadata["label"])
            trace.record(
                "whole_object_superset_override",
                "pipeline",
                requested_choice=choice,
                selected_label=label,
            )
            return superset, {
                "label": label,
                "reason": "equivalent bbox completes the whole-object mask",
                "fallback": True,
            }
        union = _whole_object_union_completion(
            selected, candidates, observations, _query_contract, plan
        )
        if union is not None:
            label = str(union.metadata["label"])
            trace.record(
                "whole_object_union_completion_override",
                "pipeline",
                requested_choice=choice,
                selected_label=label,
            )
            return union, {
                "label": label,
                "reason": "equivalent union completes the whole-object mask",
                "fallback": True,
            }
        alternate = _whole_object_large_alternate(
            selected, candidates, observations, plan
        )
        if alternate is not None:
            label = str(alternate.metadata["label"])
            trace.record(
                "whole_object_large_alternate_override",
                "pipeline",
                requested_choice=choice,
                selected_label=label,
            )
            return alternate, {
                "label": label,
                "reason": "equivalent alternate completes the whole-object mask",
                "fallback": True,
            }
        return selected, {
            "label": choice,
            "reason": reason,
            "fallback": False,
        }
    except ValueError as error:
        trace.record("mask_select_invalid", "pipeline", error=str(error))
        return default, {
            "label": default_label,
            "reason": f"invalid selection; conservative fallback: {error}",
            "fallback": True,
        }


def _require_output_path(path: Path) -> Path:
    return Path(path).resolve()


def _save_image(image: Image.Image, path: Path) -> None:
    path = _require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".tmp.png", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def save_binary_mask(mask: np.ndarray, output_path: Path) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("selected mask must be two-dimensional")
    _save_image(
        Image.fromarray(binary.astype(np.uint8) * 255, mode="L"),
        output_path,
    )
    output_path = output_path.resolve()
    return {
        "path": str(output_path),
        "shape": [int(binary.shape[0]), int(binary.shape[1])],
        "area": int(binary.sum()),
        "encoding": "png",
        "sha256": _file_sha256(output_path),
        "raw_sha256": _mask_sha256(binary),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = _require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise




def _prompt_protocol_sha256() -> str:
    payload = "\n".join((
        QUERY_CONTRACT_INSTRUCTIONS,
        canonical_target_prompt("<QUERY>"),
        VISUAL_BINDING_INSTRUCTIONS,
        TARGET_PLAN_INSTRUCTIONS,
        OBSERVE_INSTRUCTIONS,
        SELECT_INSTRUCTIONS,
        json.dumps(sorted(CONTRACT_KEYS)),
        json.dumps(sorted(VISUAL_BINDING_KEYS)),
        json.dumps(sorted(PLAN_KEYS)),
        json.dumps(sorted(OBSERVATION_KEYS)),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selection_action(candidate: Candidate) -> str:
    if candidate.source in {"primary", "literal"}:
        return "keep_primary"
    if candidate.source == "none":
        return "no_candidate"
    return "use_alternative"


def _mask_input(candidate: Candidate) -> dict[str, Any]:
    if candidate.source in {"primary", "literal", "alternate", "planner"}:
        return {"type": "text", "prompt": candidate.prompt}
    if candidate.source == "bbox":
        return {
            "type": "bbox",
            "bbox_xyxy": list(candidate.metadata["bbox_xyxy"]),
        }
    return {"type": "none"}


def _candidate_artifacts(
    candidates: list[Candidate],
    destination: Path,
) -> list[dict[str, Any]]:
    records = []
    for candidate in candidates:
        label = str(candidate.metadata["label"])
        mask_record = save_binary_mask(
            candidate.mask, destination / "candidates" / f"{label}.png"
        )
        records.append({
            "label": label,
            "source": candidate.source,
            "variant": candidate.variant,
            "prompt": candidate.prompt,
            "confidence": candidate.confidence,
            "mask": mask_record,
            "sam_input": _mask_input(candidate),
            "target_claim": {
                key: value for key, value in candidate.metadata[
                    "target_claim"
                ].items() if not key.startswith("_")
            },
        })
    return records




def generate(
    image_path: str,
    query: str,
    qwen: object,
    sam3: object,
    output_dir: str,
) -> dict[str, Any]:
    image = Path(image_path).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    normalized_query = " ".join(str(query).split()).strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    destination = _require_output_path(Path(output_dir))
    destination.mkdir(parents=True, exist_ok=True)

    trace = Trace(work_dir=str(destination / "boards"))
    query_contract = build_query_contract(normalized_query, qwen, trace)
    canonical_prompt = build_canonical_target(
        str(image), normalized_query, qwen, trace
    )
    visual_binding = build_visual_binding(
        str(image), normalized_query, query_contract, canonical_prompt,
        qwen, trace,
    )
    plan = build_target_plan(
        str(image), normalized_query, query_contract, visual_binding,
        canonical_prompt, qwen, trace,
    )
    candidates = build_candidates(str(image), plan, sam3, trace)
    board_path: Path | None = None

    if candidates:
        observations, board_path = observe_candidates(
            str(image),
            candidates,
            qwen,
            trace,
            destination / "boards" / "candidate_board.png",
        )
        selected, decision = select_candidate(
            normalized_query,
            query_contract,
            plan,
            observations,
            candidates,
            qwen,
            trace,
            board_path,
        )
    else:
        with Image.open(image) as loaded:
            shape = (loaded.height, loaded.width)
        selected = Candidate(
            prompt=None,
            mask=np.zeros(shape, dtype=bool),
            source="none",
            variant="none",
        )
        decision = {
            "label": None,
            "reason": "SAM3 returned no non-empty candidate",
            "fallback": True,
        }

    candidate_records = _candidate_artifacts(candidates, destination)
    mask_record = save_binary_mask(
        selected.mask, destination / "selected_mask.png"
    )
    selected_claim = _candidate_target_claim(selected, plan)
    display = dict(selected_claim["_display"])
    accepted = selected.source != "none"
    full_des = {
        **display,
        "confidence": plan["confidence"],
        "validation": "accepted" if accepted else "no_candidate",
    }
    legacy_full_description = {
        "sam_prompt": full_des["text"],
        "target_entity": full_des["target_entity"],
        "target_number": full_des["number"],
        "target_scope": full_des["scope"],
        "selector": full_des["selector"],
        "visible_discriminator": full_des["visible_discriminator"],
        "evidence": full_des["evidence"],
        "confidence": full_des["confidence"],
    }
    mask_prompt = (
        selected.prompt
        if selected.source in {"primary", "literal", "alternate", "planner"}
        else None
    )
    selection = {
        "action": _selection_action(selected),
        "label": decision["label"],
        "source": selected.source,
        "variant": selected.variant,
        "reason": decision["reason"],
        "fallback": bool(decision["fallback"]),
    }

    result_path = destination / "result.json"
    source_hash = _file_sha256(Path(__file__))
    result = {
        "schema_version": SCHEMA_VERSION,
        "image_path": str(image),
        "query": normalized_query,
        "input": {
            "image_sha256": _file_sha256(image),
            "model": REQUIRED_QWEN,
        },
        "ground_truth_loaded": False,
        "query_contract": query_contract,
        "visual_binding": visual_binding,
        "full_description": legacy_full_description,
        "full_description_validation": {
            "accepted": accepted,
            "violations": [] if accepted else ["no mask candidate"],
        },
        "full_des": full_des,
        "mask_prompt": mask_prompt,
        "mask_input": _mask_input(selected),
        "mask": mask_record,
        "selection": selection,
        "candidates": candidate_records,
        "candidate_board_path": str(board_path) if board_path else None,
        "trace": {
            "qwen_call_count": trace.qwen_call_count,
            "sam3_call_count": trace.sam3_call_count,
            "prompt_protocol_sha256": _prompt_protocol_sha256(),
            "source_sha256": source_hash,
            "stages": trace.stages,
        },
        "result_path": str(result_path.resolve()),
    }
    _atomic_json(result_path, result)
    return result



