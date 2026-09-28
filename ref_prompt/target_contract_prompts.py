"""Query-first target contracts for Full General V4.

The original query is authoritative.  Qwen parses its grammatical target once,
SAM3 may add visual evidence, and deterministic checks prevent target shift.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

reasoning_prompts = {}

MAX_REFERENCE_OBJECTS = 2
VALID_SCOPES = {"whole_object", "part", "region", "group"}

HUMAN_WORDS = {
    "person", "people", "man", "men", "woman", "women", "boy", "boys",
    "girl", "girls", "child", "children", "kid", "kids", "guy", "guys",
    "dude", "dudes", "dudue", "lady", "ladies",
    "referee", "referees", "umpire", "umpires", "soldier", "soldiers",
}
IMPLICIT_OWNER_WORDS = {
    "shirt", "shirts", "shorts", "pants", "trousers", "jeans", "jacket",
    "coat", "sleeve", "sleeves", "sock", "socks", "shoe", "shoes", "dress",
    "skirt", "uniform", "hat", "hats", "cap", "caps", "suit", "suits",
    "jersey", "jerseys", "tie", "ties", "arm", "arms", "hand",
    "hands", "leg", "legs", "foot", "feet", "elbow", "elbows", "knee",
    "knees", "shoulder", "shoulders", "hair", "face", "head", "heads",
    "neck", "necks", "body", "bodies",
    "glasses", "goggles", "sunglasses", "spectacles",
}
CLOTHING_OWNER_WORDS = {
    "shirt", "shirts", "shorts", "pants", "trousers", "jeans", "jacket",
    "coat", "sleeve", "sleeves", "sock", "socks", "shoe", "shoes", "dress",
    "skirt", "uniform", "hat", "hats", "cap", "caps", "suit", "suits",
    "jersey", "jerseys", "tie", "ties",
    "glasses", "goggles", "sunglasses", "spectacles",
}
ANATOMY_OWNER_WORDS = IMPLICIT_OWNER_WORDS - CLOTHING_OWNER_WORDS

APPEARANCE_WORDS = {
    "striped", "spotted", "plaid", "checkered", "patterned", "wooden",
    "metal", "plastic", "old", "young", "adult", "older", "younger", "big", "small",
    "large", "tiny", "tall", "short", "long-haired", "short-haired",
}

COLOR_WORDS = {
    "black", "white", "gray", "grey", "brown", "red", "orange", "yellow",
    "green", "blue", "purple", "pink", "beige", "tan", "gold", "golden",
    "silver", "cream", "maroon", "burgundy", "navy",
}
_COLOR_TAIL_STOP = {
    "and", "or", "of", "on", "in", "at", "to", "from", "with", "without",
    "behind", "beside", "between", "under", "over", "above", "below", "near",
    "left", "right", "top", "bottom", "front", "back", "standing", "sitting",
    "holding", "held", "wearing", "carrying", "looking", "touching", "riding",
}
ACTION_WORDS = {
    "standing", "sitting", "seated", "walking", "running", "kneeling", "lying",
    "jumping", "smiling", "leaning", "holding", "carrying", "wearing", "riding",
    "touching", "looking", "pulling", "pushing", "feeding", "eating", "catching",
    "facing", "drinking", "petting", "grasping", "dressed", "parked", "calling", "skiing",
    "talking", "nestled",
}
RELATION_PATTERNS = (
    r"\b(?:to\s+(?:the\s+)?)?left\s+of\b",
    r"\b(?:to\s+(?:the\s+)?)?right\s+of\b",
    r"\bin\s+front\s+of\b",
    r"\b(?:next|close)\s+to\b",
    r"\bbeside\b",
    r"\b(?:behind|behidn)\b",
    r"\bbetween\b",
    r"\b(?:under|below|above|over)\b",
    r"\bnear\b",
    r"\bwith\b",
    r"\bon\b(?!\s+(?:the\s+)?(?:(?:very|far|farthest|furthest)\s+)?"
    r"(?:left|right|top|bottom|edge|side|front|back|middle|foreground|background|"
    r"it|there|one|ones)\b)",
    r"\b(?:holding|holds|held\s+by|carrying|carried\s+by|used\s+by|"
    r"being\s+(?:held|carried|used)\s+by|riding|touching|looking\s+at|"
    r"approaching|following|followed\s+by|grasping)\b",
    r"\b(?:blocked|covered|occluded)\s+by\b",
)
_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
}


TARGET_NAME_PROMPT = r'''The image is loaded, but analyze sentence grammar first.
ORIGINAL_QUERY: {Q}

Identify the physical target of the whole referring expression. Lock the target
that the query asks to localize; never promote a noun that is only a relation or
reference object.

- Explicit target: copy its exact query span. The image may choose an instance,
  but may not rename that span.
- Headless cue (only color, position, clothing, action, comparison, age, or number):
  use the image to add exactly one broad target category.
- For "X approaching/near/behind Y", "X used/carried/held by Y", and
  "X that Y is sitting on", X is the target and Y is reference-only.
- For a nonhuman "body/back/rear/reflection of X", keep that requested scope in
  a compact name such as "elephant body" or "bird reflection".
- Clothing and human body-part phrases can identify a person bbox: keep the
  phrase as a selector while using person as the broad target category. When
  the phrase itself has no explicit owner, return target_name="person",
  source="visual_inference", and an empty target_span.
- Only for a headless action cue or an ownerless human body-part cue, if several
  people still match, return one short anchor (at most 8 words). Prefer a physical
  relation such as "on couch" or "holding hotdog"; otherwise use absolute image
  position. A clothing-only anchor is invalid. Never add a second anchor.
- Use target_scope="group" only when the query jointly refers to multiple target
  instances; use part/region only when it explicitly asks for a physical part.
- In "left/right side of X", do not assume that side is a physical part; grammar
  and the image may instead mean the X located on that side.
- If broken grammar leaves multiple plausible targets, set confidence="low" and
  list only exact query-span candidates. Do not invent candidates from the image.

Examples: "bus behind man" -> bus; "laptop carried by child" -> laptop;
"back of a jacket" -> person selected by the jacket back. For headless "blue 7" or
"leaning over", and for age-only "adult", infer one visible broad category and
leave target_span empty.

Return JSON only:
{{"target_name":"object","target_scope":"whole_object","source":"explicit_query","target_span":"object words copied from query or empty","reference_span":"reference words copied from query or empty","visual_anchor":"empty unless the query is headless","role":"grammatical target","confidence":"high","candidates":[]}}'''


TARGET_CANDIDATE_PROMPT = r'''The image is loaded.
ORIGINAL_QUERY: {Q}
CANDIDATES_COPIED_FROM_QUERY: {candidates}

Choose the candidate ID that is the grammatical target of the whole query.
Use image evidence only to break a real grammatical tie. Return an ID from the
list; never write a new object name.

Return JSON only: {{"candidate_id":"T1"}}'''


QUERY_PLAN_PROMPT = r'''The original image is loaded.
ORIGINAL_QUERY: {Q}
LOCKED_TARGET_NAME: {target_name}
REFERENCE_CANDIDATES_COPIED_FROM_QUERY: {reference_candidates}

Select distinct REFERENCE objects used to identify the locked target. A reference
object supplies context and must never become the target. The candidates were
extracted from the query before this call.

- Return candidate IDs only. Never write or invent an object name from the image.
- Do not treat target-owned clothing, body parts, colors, patterns, or an
  intransitive pose such as standing/sitting as a reference object.
- Select at most two IDs. Return [] when none is reliable.

Examples: "dog with frisbee" -> the frisbee candidate; "man holding a golf
club" -> the golf-club candidate; "person wearing a black shirt" -> [];
"man with one prosthetic leg" -> [].

Return JSON only:
{{"reference_ids":["R1"]}}'''


FULL_DESCRIPTION_PROMPT = r'''The original image is Image 1.
ORIGINAL_QUERY (authoritative): {Q}
LOCKED_TARGET_CONTRACT: {target_contract}
LITERAL_CUES_TO_COPY: {literal_cues}
REFERENCE_OBJECTS_NEVER_TARGET: {references}
VISUAL_EVIDENCE_IMAGES: {evidence_note}

Losslessly linearize the original query as one concise target-first description.
Begin exactly with "The {target_name}". Keep every query cue with the same owner,
scope, direction, negation, count, and relation role. Green masks are reference
context only. When the locked contract has a nonempty visual_anchor, copy that
single anchor; otherwise add no visual fact. Write a restrictive noun phrase
("The giraffe that is taller"), not a standalone claim ("The giraffe is taller").
For a physical support relation, "X on Y" may be written "X on top of Y". Do not
caption, explain, or reinterpret the image.

Output only the description.'''


FINAL_BBOX_PROMPT = r'''Context: User Referring Expression Q: '{description}'.
Image size: width={W}, height={H}.

Task: Detect the specific target instance described by '{description}' and output
a SINGLE, TIGHT bounding box. {scope_instruction}
Format: [x1, y1, x2, y2]
Output ONLY the list [x1, y1, x2, y2]:'''


CONTRACT_BBOX_PROMPT = r'''ORIGINAL_QUERY: {Q}
LOCKED_TARGET_CONTRACT: {target_contract}
VERIFIED_TARGET_DESCRIPTION: {description}
Image size: width={W}, height={H}.

Detect the locked target requested by ORIGINAL_QUERY. Use the verified description
only as clarification; never change target/reference roles. Output one tight box.
{scope_instruction}
Format: [x1, y1, x2, y2]
Output ONLY the list [x1, y1, x2, y2]:'''


ORDINAL_BBOX_PROMPT = r'''The original image is Image 1. Additional images show up
to three SAM3 masks, each labeled TARGET CANDIDATE A/B/C.
ORIGINAL_QUERY (authoritative): {Q}
VERIFIED_TARGET_NAME: {target_name}
VERIFIED_FULL_DESCRIPTION: {description}
ORDINAL_RULE: {ordinal}
CANDIDATE_RANK_MAP: {candidate_rank_map}
CANDIDATE_SAM_BBOX_MAP: {candidate_bbox_map}
Image size: width={W}, height={H}.

Choose the candidate mask that satisfies the full original query, especially its
ordinal direction. Candidate ranks are sorted from the named direction. Then
generate a tight absolute-pixel bbox around that SAME chosen mask; candidate_id
and bbox must identify one instance. Use that candidate's SAM bbox as the geometric
prior and refine it only when the image clearly shows an incomplete mask. Do not
choose a reference object.

Return JSON only:
{{"candidate_id":"A|B|C","bbox":[x1,y1,x2,y2]}}'''


def _space(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _space(text).lower()).strip()


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", str(text).lower())


def canonical_name(name: str) -> str:
    words = _words(name)
    while words and words[0] in {"a", "an", "the"}:
        words.pop(0)
    if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    if words:
        words[-1] = {
            "women": "woman", "men": "man", "people": "person",
            "children": "child", "feet": "foot", "teeth": "tooth",
            "dudue": "dude",
        }.get(words[-1], words[-1])
    return " ".join(words)


def names_equivalent(left: str, right: str) -> bool:
    return bool(canonical_name(left)) and canonical_name(left) == canonical_name(right)


def _contains_token_sequence(text: str, phrase: str) -> bool:
    haystack, needle = _norm(text).split(), _norm(phrase).split()
    return bool(needle) and any(
        haystack[index:index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def _contains_reference_content(text: str, phrase: str) -> bool:
    grammar = {
        "a", "an", "the", "of", "to", "for", "that", "which", "who",
        "is", "are", "was", "were", "it", "on", "in", "at", "by",
    }
    haystack = set(_words(text))
    wanted = [word for word in _words(phrase) if word not in grammar]
    return bool(wanted) and all(word in haystack for word in wanted)


def _contains_selector_content(text: str, phrase: str, target_name: str) -> bool:
    ignored = {
        "a", "an", "the", "of", "to", "for", "that", "which", "who",
        "is", "are", "was", "were", "it", "on", "in", "at", "by", "with",
        "his", "her", "their", "its",
    } | HUMAN_WORDS | set(_words(target_name))
    haystack = set(_words(text))
    wanted = [word for word in _words(phrase) if word not in ignored]
    return bool(wanted) and all(word in haystack for word in wanted)


def _exact_query_span(query: str, phrase: str) -> str:
    matches = list(re.finditer(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", query))
    needle = _words(phrase)
    if not needle:
        return ""
    for index in range(len(matches) - len(needle) + 1):
        if [item.group(0).lower() for item in matches[index:index + len(needle)]] == needle:
            return _space(query[matches[index].start():matches[index + len(needle) - 1].end()])
    return ""


def _clean_target_name(name: str) -> str:
    words = _words(name)
    leading_modifiers = (
        {"a", "an", "the", "one", "first", "second", "third", "fourth", "fifth",
         "left", "right", "top", "bottom", "upper", "lower", "dark", "light",
         "front", "empty", "blurred", "blurry",
         "leftmost", "rightmost", "topmost", "bottommost", "closer", "closest",
         "nearest", "far", "farthest", "furthest", "bigger", "biggest", "larger", "largest", "smaller",
         "smallest", "taller", "tallest", "shorter", "shortest", "short", "haired",
         "big", "small", "large", "tiny", "blurred", "blurry", "whole", "older",
         "younger", "baby", "adult", "all", "both", "and", "of", "from", "to",
         "completely", "visible", "visibile", "visiable", "partially"}
        | COLOR_WORDS | APPEARANCE_WORDS | ACTION_WORDS
    )
    # A lone color word can be a COCO noun (notably "orange").
    lone_color_noun = words == ["orange"]
    while words and not lone_color_noun and words[0] in leading_modifiers:
        words.pop(0)
    cutters = ACTION_WORDS | HUMAN_WORDS | {
        "a", "an", "the", "this", "these", "those", "not",
        "of", "on", "in", "at", "to", "from", "with", "without", "no",
        "close", "behind", "beside",
        "between", "under", "over", "above", "below", "near", "left", "right",
        "top", "bottom", "upper", "lower", "front", "back", "partially",
        "covered", "blocked", "occluded", "lined", "that", "which", "who", "is", "are",
        "was", "were", "first", "second", "third", "fourth", "fifth", "last", "all",
        "both", "ahead", "farthest", "furthest", "closest", "nearest", "bigger",
        "biggest", "smaller", "smallest", "taller", "tallest", "shorter", "shortest",
    }
    cut = next((index for index, word in enumerate(words) if index > 0 and word in cutters), len(words))
    return " ".join(words[:cut])


def valid_locked_target_name(name: str) -> bool:
    words = _words(name)
    non_heads = {
        "a", "an", "the", "this", "that", "these", "those", "one", "all",
        "both", "something", "anything", "everything", "object", "item",
        "on", "in", "at", "to", "from", "of", "with", "by", "over", "under",
        "left", "right", "top", "bottom", "front", "back", "edge", "side",
        "near", "row", "number", "image", "picture", "photo", "scene", "frame",
    }
    return bool(
        words
        and words[0] not in non_heads
        and not (set(words) & IMPLICIT_OWNER_WORDS)
        and not all(word in non_heads or word in COLOR_WORDS for word in words)
        and not all(word.isdigit() for word in words)
    )


def extract_json(raw: str) -> Optional[dict]:
    start, end = str(raw or "").find("{"), str(raw or "").rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(str(raw)[start:end + 1])
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _color_phrases(query: str) -> List[str]:
    tokens = list(re.finditer(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", query))
    phrases, seen = [], set()
    i = 0
    while i < len(tokens):
        word = tokens[i].group(0).lower()
        color_index = i + 1 if word in {"dark", "light"} and i + 1 < len(tokens) else i
        if tokens[color_index].group(0).lower() not in COLOR_WORDS:
            i += 1
            continue
        start, end, j = tokens[i].start(), tokens[color_index].end(), color_index
        if j + 2 < len(tokens) and tokens[j + 1].group(0).lower() == "and" and tokens[j + 2].group(0).lower() in COLOR_WORDS:
            end, j = tokens[j + 2].end(), j + 2
        if j + 1 < len(tokens):
            tail = tokens[j + 1].group(0).lower()
            if tail not in _COLOR_TAIL_STOP and tail not in COLOR_WORDS:
                end = tokens[j + 1].end()
        phrase = _space(query[start:end])
        key = _norm(phrase)
        if key and key not in seen:
            seen.add(key)
            phrases.append(phrase)
        i = j + 1
    return phrases


def parse_ordinal(query: str) -> Optional[dict]:
    rank_words = "|".join(map(re.escape, _ORDINALS))
    patterns = (
        rf"\b(?P<rank>{rank_words})\b(?:\s+[A-Za-z-]+){{0,4}}?\s+from\s+(?:the\s+)?(?P<direction>left|right|top|bottom)\b",
        rf"\b(?P<direction>left|right|top|bottom)\s+(?P<rank>{rank_words})\b",
        rf"\b(?P<rank>{rank_words})\s+(?P<direction>left|right|top|bottom)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, query, re.I)
        if not match:
            continue
        direction = match.group("direction").lower()
        return {
            "rank": _ORDINALS[match.group("rank").lower()],
            "direction": direction,
            "axis": "x" if direction in {"left", "right"} else "y",
            "descending": direction in {"right", "bottom"},
            "evidence_span": _space(match.group(0)),
        }
    return None


def _absolute_positions(query: str, ordinal: Optional[dict]) -> List[str]:
    spans = []
    specific = re.compile(
        r"\b(?:left\s*-?\s*most|right\s*-?\s*most|top\s*-?\s*most|bottom\s*-?\s*most|"
        r"(?:far|farthest|furthest|very)\s+(?:to\s+the\s+)?(?:left|right)|"
        r"(?:at|on)\s+the\s+(?:top|bottom|left|right)|"
        r"from\s+(?:the\s+)?(?:front|back))\b",
        re.I,
    )
    spans.extend(_space(match.group(0)) for match in specific.finditer(query))
    spans.extend(
        _space(match.group(0))
        for match in re.finditer(r"\b(?:foreground|background|center|centre|middle|ahead)\b", query, re.I)
    )
    for match in re.finditer(r"\b(?:top|bottom|upper|lower|left|right|front|back)\b", query, re.I):
        tail = query[match.end():]
        if re.match(r"\s+(?:of|above|below|under|behind)\b", tail, re.I):
            continue
        span = _space(match.group(0))
        if ordinal and _norm(span) in _norm(ordinal["evidence_span"]).split():
            continue
        spans.append(span)
    output, seen = [], set()
    for span in spans:
        key = _norm(span)
        if key and key not in seen and not any(key != _norm(old) and key in _norm(old).split() for old in spans):
            seen.add(key)
            output.append(span)
    return output


def _implicit_person_owner_span(target_clause: str) -> str:
    """Map non-COCO clothing/body selectors to their person bbox owner."""
    words = [word for word in _words(target_clause) if word not in {"a", "an", "the"}]
    if not words or set(words) & HUMAN_WORDS:
        return ""
    clothing_prefix = COLOR_WORDS | APPEARANCE_WORDS | ACTION_WORDS | {
        "one", "two", "left", "right", "top", "bottom", "upper", "lower",
        "visible", "blurry", "blurred", "back", "front", "of", "only", "and",
        "far", "farthest", "furthest", "leftmost", "rightmost", "light", "dark",
        "pale", "turned", "number", "usa", "baseball",
    }
    if (
        words[-1] in CLOTHING_OWNER_WORDS
        and all(
            word in clothing_prefix or word in CLOTHING_OWNER_WORDS or word.isdigit()
            for word in words[:-1]
        )
    ):
        return " ".join(words[-5:])
    anatomy_prefix = COLOR_WORDS | APPEARANCE_WORDS | ACTION_WORDS | {
        "one", "two", "left", "right", "top", "bottom", "upper", "lower",
        "visible", "blurry", "blurred", "prosthetic",
    }
    if words[-1] in ANATOMY_OWNER_WORDS and all(
        word in anatomy_prefix for word in words[:-1]
    ):
        return " ".join(words)
    explicit_part = re.search(
        r"\b(?:back|top|side)\s+of\s+(?:the\s+)?(?:head|hair|face|neck)\b",
        target_clause,
        re.I,
    )
    return _space(explicit_part.group(0)) if explicit_part else ""


def _reference_candidates(query: str) -> List[dict]:
    """Copy relation-tail candidates from the query; never infer image nouns."""
    matches = sorted(
        (
            match.start(), match.end(), _space(match.group(0))
        )
        for pattern in RELATION_PATTERNS
        for match in re.finditer(pattern, query, re.I)
    )
    unique = []
    for item in matches:
        if not unique or item[:2] != unique[-1][:2]:
            unique.append(item)
    candidates = []
    for index, (_, end, relation) in enumerate(unique):
        stop = unique[index + 1][0] if index + 1 < len(unique) else len(query)
        punctuation = re.search(r"[,;.]", query[end:stop])
        if punctuation:
            stop = end + punctuation.start()
        phrase = _space(query[end:stop]).strip(" ,.;:[]{}()\"'")
        boundary = re.search(
            r"\s+\b(?:and|or|while|who|that|which|wearing|holding|carrying|standing|sitting|walking|running)\b",
            phrase,
            re.I,
        )
        if boundary:
            phrase = phrase[:boundary.start()].strip()
        words = phrase.split()
        if not words:
            continue
        phrase = " ".join(words[:8])
        candidates.append({
            "id": f"R{len(candidates) + 1}",
            "query_phrase": phrase,
            "relation": relation,
        })
    return candidates


def static_query_plan(query: str) -> Dict[str, Any]:
    ordinal = parse_ordinal(query)
    colors = _color_phrases(query)
    positions = _absolute_positions(query, ordinal)
    actions = []
    for match in re.finditer(r"\b[A-Za-z]+\b", query):
        if match.group(0).lower() in ACTION_WORDS:
            actions.append(match.group(0))
    actions = list(dict.fromkeys(actions))
    relation_triggers = [
        normalize_literal_spelling(_space(match.group(0)))
        for pattern in RELATION_PATTERNS
        for match in re.finditer(pattern, query, re.I)
    ]
    ordinal_cues = [ordinal["evidence_span"]] if ordinal else [
        match.group(0) for match in re.finditer(r"\b(?:first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th))\b", query, re.I)
    ]
    hard_cues = [
        _space(match.group(0))
        for pattern in (
            r"\b(?:no|not|without)\s+[A-Za-z0-9'-]+",
            r"\b(?:only|all|both|whole|half|empty|real|mirror\s+image|reflection|"
            r"blurry|blurred|partial(?:ly)?|visible|visibile|visiable)\b",
            r"\b\d+\b",
            r"\b(?:closest|nearest|farthest|furthest|bigger|biggest|smaller|smallest|"
            r"taller|tallest|shorter|shortest|longer|longest)\b",
        )
        for match in re.finditer(pattern, query, re.I)
    ]
    hard_cues = [
        "visible" if _norm(cue) in {"visibile", "visiable"} else cue
        for cue in hard_cues
    ]
    appearance_cues = [
        _space(match.group(0))
        for match in re.finditer(
            r"\b(?:short[- ]haired|long[- ]haired|striped|spotted|plaid|checkered|"
            r"patterned|wooden|metal|plastic|adult|older|younger|big|small|large|tiny|tall|short)\b",
            query,
            re.I,
        )
    ]
    relation_starts = [
        match.start()
        for pattern in RELATION_PATTERNS
        for match in re.finditer(pattern, query, re.I)
    ]
    role_starts = relation_starts + [
        match.start() for match in re.finditer(r"\b(?:that|which|who)\b", query, re.I)
    ] + [
        match.start() for match in re.finditer(r"\bwearing\b", query, re.I)
    ]
    target_clause = query[:min(role_starts)] if role_starts else query
    target_role_span = _space(target_clause).strip(" ,.;:")
    target_role_name = _clean_target_name(target_role_span)
    owner_clause = query if not target_role_span and re.match(r"^\s*wearing\b", query, re.I) else target_role_span
    owner_proxy_span = _implicit_person_owner_span(owner_clause)
    scope_words = [
        match.group(0)
        for match in re.finditer(
            r"\b(?:piece|part|portion|section|body|back|butt|rear|half|mirror\s+image|reflection|corner)\b",
            target_clause,
            re.I,
        )
    ]
    literal = list(dict.fromkeys(
        colors + positions + actions + ordinal_cues + relation_triggers
        + hard_cues + appearance_cues + scope_words
    ))
    human_match = next(
        (
            match
            for match in re.finditer(r"\b[A-Za-z]+\b", query)
            if match.group(0).lower() in HUMAN_WORDS
        ),
        None,
    )
    baby_match = next(iter(re.finditer(r"\b(?:baby|babies)\b", query, re.I)), None)
    baby_tail = _words(query[baby_match.end():]) if baby_match else []
    baby_is_human = bool(
        baby_match
        and not human_match
        and (
            not baby_tail
            or baby_tail[0] in ACTION_WORDS
            | {"is", "who", "that", "with", "without", "on", "in", "at", "near",
               "behind", "between", "left", "right", "wearing", "holding"}
        )
    )
    if baby_is_human:
        human_match = baby_match
    human_prefix = query[:human_match.start()] if human_match else ""
    human_prefix = re.sub(r"\bolf\b", "old", human_prefix, flags=re.I)
    human_prefix_words = _words(human_prefix)
    human_prefix_modifiers = (
        COLOR_WORDS | APPEARANCE_WORDS | ACTION_WORDS | IMPLICIT_OWNER_WORDS | {
            "a", "an", "the", "one", "and", "or", "they", "very", "middle",
            "row", "number", "half", "little", "fat", "hot", "dark", "light",
            "haired", "naked",
            "blond", "blonde", "asian", "angry", "smiling", "keep", "playing",
            "left", "right", "front", "back", "top", "bottom", "upper", "lower",
            "far", "farthest", "furthest", "closest", "nearest", "first", "second",
            "third", "fourth", "fifth", "last", "up", "down", "straight",
        }
    )
    human_prefix_head = (
        "" if human_prefix_words and all(
            word in human_prefix_modifiers or word.isdigit()
            for word in human_prefix_words
        ) else _clean_target_name(human_prefix)
    )
    comparative_of = re.search(
        r"^\s*(?:the\s+)?(?P<comparison>longer|shorter|taller|smaller|bigger)\s+"
        r"of\s+(?:the\s+)?(?:two|2)\s+(?P<parent>.+)$",
        query,
        re.I,
    )
    explicit_human = bool(
        human_match
        and len(_words(query[:human_match.start()])) <= 4
        and not valid_locked_target_name(human_prefix_head)
        and not query[human_match.end():].lower().startswith("'s")
        and not any(start < human_match.start() for start in relation_starts)
        and not re.search(
            r"\b(?:of|by|with|without|near|beside|behind|under|over|above|below|holding|carrying|riding|touching|looking|petting)\b",
            query[:human_match.start()],
            re.I,
        )
    )
    if explicit_human:
        target_hint = {
            "target_name": (
                "person" if _norm(human_match.group(0)) in {"baby", "babies"}
                else canonical_name(human_match.group(0))
            ),
            "reason": "explicit_human_head",
            "target_span": human_match.group(0),
        }
    elif comparative_of:
        comparison_parent = _clean_target_name(comparative_of.group("parent"))
        target_hint = {
            "target_name": canonical_name(comparison_parent),
            "reason": "comparative_head",
            "target_span": _exact_query_span(query, comparison_parent),
        }
    else:
        target_hint = None
    scope_modifier_pattern = "|".join(
        re.escape(word)
        for word in sorted(
            COLOR_WORDS | APPEARANCE_WORDS
            | {"right", "left", "top", "bottom", "upper", "lower", "open",
               "closed", "exposed", "blurry", "blurred", "visible", "small",
               "large", "tiny", "only", "whole"},
            key=len,
            reverse=True,
        )
    )
    scoped_of = re.search(
        rf"^\s*(?:(?:the|a|an)\s+)?(?:(?:{scope_modifier_pattern})\s+){{0,3}}"
        r"(?P<label>piece|part|portion|section|body|back|butt|rear|"
        r"(?:top|bottom)\s+half|half|mirror\s+image|reflection|corner)\s+of\b",
        query,
        re.I,
    )
    grouped_of = re.search(
        r"^\s*(?:(?:the\s+)?(?:all|both)\s+of\s+)(?P<parent>.+)$",
        query,
        re.I,
    )
    half_relation = re.search(
        r"^\s*(?:(?:the|a|an)\s+)?(?P<parent>[A-Za-z-]+(?:\s+[A-Za-z-]+){0,2})\s+"
        r"(?P<label>half)\s+(?:on|under|over|above|below)\b",
        query,
        re.I,
    )
    group_parent_words = [
        word for word in _words(grouped_of.group("parent"))
        if word not in {"a", "an", "the"}
    ] if grouped_of else []
    group_head = group_parent_words[0] if group_parent_words else ""
    group_is_multiple = bool(
        grouped_of
        and (
            re.match(r"^\s*(?:the\s+)?both\s+of\b", query, re.I)
            or group_head in {"people", "men", "women", "children", "kids"}
            or (len(group_head) > 2 and group_head.endswith("s") and not group_head.endswith("ss"))
        )
    )
    scope_hint = (
        "group" if group_is_multiple else
        "part" if (scoped_of or half_relation) else None
    )
    raw_scope_label = _space(
        (scoped_of or half_relation).group("label") if (scoped_of or half_relation) else ""
    ).lower()
    scope_label = {
        "mirror image": "reflection", "butt": "rear",
        "top half": "half", "bottom half": "half",
    }.get(raw_scope_label, raw_scope_label)
    scope_parent_hint = ""
    if group_is_multiple:
        scope_parent_hint = _clean_target_name(grouped_of.group("parent"))
    elif scoped_of:
        parent_text = query[scoped_of.end():]
        scope_parent_hint = _clean_target_name(parent_text)
    elif half_relation:
        scope_parent_hint = _clean_target_name(half_relation.group("parent"))
    if owner_proxy_span:
        scope_hint = None
        scope_label = ""
        scope_parent_hint = ""
    elif scope_hint and target_hint is None:
        target_hint = {
            "target_name": scope_parent_hint or None,
            "reason": "scoped_parent_hint",
            "target_span": _exact_query_span(query, scope_parent_hint),
        }
    compact_words = [
        word for word in _words(target_role_span) if word not in {"a", "an", "the"}
    ]
    action_only = bool(
        compact_words
        and compact_words[0] in ACTION_WORDS
        and all(
            word in ACTION_WORDS | {"up", "down", "over", "straight"}
            for word in compact_words
        )
    )
    implicit_person_owner = bool(
        not explicit_human and bool(owner_proxy_span or action_only)
    )
    anatomy_owner = bool(
        owner_proxy_span
        and set(_words(owner_proxy_span)) & ANATOMY_OWNER_WORDS
    )
    side_location = bool(re.match(
        r"^\s*(?:left|right)\s+side\s+of\b",
        query,
        re.I,
    ))
    role_words = set(_words(target_role_span))
    target_name_words = set(_words(target_role_name))
    preserve_raw_query = bool(
        role_starts
        and not owner_proxy_span
        and (role_words & ANATOMY_OWNER_WORDS) - target_name_words
    )
    support_reference = ""
    if half_relation:
        tail = query[half_relation.end():]
        support_reference = _space(tail).strip(" ,.;:")
        literal.append("on")
    return {
        "color_phrases": colors,
        "absolute_positions": positions,
        "actions": actions,
        "ordinal": ordinal,
        "ordinal_cues": ordinal_cues,
        "hard_cues": hard_cues,
        "appearance_cues": appearance_cues,
        "relation_triggers": list(dict.fromkeys(relation_triggers)),
        "reference_candidates": _reference_candidates(query),
        "target_role_locked": bool(role_starts and target_role_span),
        "target_role_span": target_role_span,
        "target_role_name": target_role_name,
        "implicit_person_owner": implicit_person_owner,
        "owner_proxy_span": owner_proxy_span,
        "allow_visual_anchor": bool(action_only or anatomy_owner),
        "anatomy_anchor": anatomy_owner,
        "headless_query": bool(
            not implicit_person_owner and not valid_locked_target_name(target_role_name)
        ),
        "side_location": side_location,
        "preserve_raw_query": preserve_raw_query,
        "support_reference": support_reference,
        "literal_cues": literal,
        "target_hint": target_hint,
        "scope_hint": scope_hint,
        "scope_label": scope_label,
        "scope_parent_hint": scope_parent_hint,
        "scope_cues": list(dict.fromkeys(scope_words)),
    }


def parse_target_response(raw: str, query: str, target_hint: Optional[dict] = None,
                          scope_hint: Optional[str] = None,
                          allow_visual_anchor: bool = False,
                          headless_query: bool = False,
                          anatomy_anchor: bool = False):
    value = extract_json(raw) or {}
    warnings = [] if value else ["target_json"]
    raw_name = _space(value.get("target_name")).strip(" ,.;:[]{}()\"'")
    if not raw_name and not value and 1 <= len(_words(raw)) <= 6:
        raw_name = _space(raw).strip(" ,.;:[]{}()\"'")
    name = _clean_target_name(raw_name)
    if _norm(name) in {"broad head", "broad head or scoped compound", "object words copied from query or empty"}:
        name = ""
        warnings.append("schema_placeholder_rejected")
    if name and all(word.isdigit() for word in _words(name)):
        name = ""
        warnings.append("numeric_target_rejected")
    headless_nonphysical_heads = {
        "image", "picture", "photo", "scene", "frame", "row", "number",
        "stripe", "stripes", "with", "near",
    }
    if headless_query and (
        _norm(name) in headless_nonphysical_heads
        or (_words(name) and _words(name)[0] in headless_nonphysical_heads)
    ):
        name = ""
        warnings.append("headless_nonphysical_target_rejected")
    if _norm(name) in {
        "from", "of", "on", "in", "at", "to", "left", "right", "top",
        "bottom", "far", "farthest", "closest", "nearest", "over", "under",
        "all", "both", "half", "part", "piece", "side",
    }:
        name = ""
        warnings.append("non_head_target_rejected")
    scope = _space(value.get("target_scope")).lower()
    declared_source = _space(value.get("source")).lower()
    source = declared_source
    target_span = _exact_query_span(
        query, _space(value.get("target_span") or value.get("evidence_span"))
    )
    reference_span = _exact_query_span(query, _space(value.get("reference_span")))
    if not name and target_hint and target_hint.get("target_name"):
        name = _space(target_hint["target_name"])
        warnings.append("advisory_target_hint_used")
    if not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 '\-]{0,80}", name):
        name = "object"
        warnings.append("target_name_fallback")
    if len(_words(name)) > 6:
        name = " ".join(_words(name)[-4:])
        warnings.append("target_name_shortened")
    if target_span and not set(_words(name)) & set(_words(target_span)):
        exact_current_name = _exact_query_span(query, name)
        span_name = _clean_target_name(target_span)
        if exact_current_name:
            target_span = exact_current_name
            warnings.append("target_span_repaired")
        elif headless_query:
            target_span = ""
            source = "visual_inference"
            warnings.append(
                "headless_visual_target_retained"
                if name != "object" else "headless_span_rejected"
            )
        elif span_name and _norm(span_name) not in {
            "of", "on", "in", "at", "to", "from", "left", "right", "top", "bottom"
        }:
            name = span_name
            warnings.append("explicit_name_restored_from_span")
        else:
            target_span = ""
            warnings.append("target_span_mismatch")
    if target_span and _norm(reference_span) == _norm(target_span):
        reference_span = ""
        warnings.append("self_reference_cleared")
    if scope_hint in VALID_SCOPES:
        scope = scope_hint
    if scope not in VALID_SCOPES:
        scope = "whole_object"
        warnings.append("target_scope_normalized")
    exact_name_span = _exact_query_span(query, name)
    if not target_span and exact_name_span:
        target_span = exact_name_span
    if source not in {"explicit_query", "visual_inference"}:
        source = "explicit_query" if target_span else "visual_inference"
        warnings.append("target_source_normalized")
    if source == "explicit_query" and not target_span:
        source = "visual_inference"
        warnings.append("explicit_span_missing")
    visual_anchor = _space(value.get("visual_anchor")).strip(" ,.;:[]{}()\"'")
    anchor_allowed = allow_visual_anchor
    anchor_words = set(_words(visual_anchor))
    anchor_selector_words = (
        COLOR_WORDS | APPEARANCE_WORDS | IMPLICIT_OWNER_WORDS | ACTION_WORDS
        | {"left", "right", "top", "bottom", "front", "back", "foreground",
           "background", "center", "middle", "near", "beside", "behind", "on", "with"}
    )
    spatial_anchor_words = {
        "left", "right", "top", "bottom", "front", "back", "foreground",
        "background", "center", "middle",
    }
    support_anchor = bool(re.search(
        r"\b(?:on|at|near|beside|behind|by|holding|carrying|grasping)\s+"
        r"(?:the\s+)?[a-z0-9-]+"
        r"(?:\s+[a-z0-9-]+){0,2}$",
        visual_anchor,
        re.I,
    ))
    support_action = bool(
        set(_words(query)) & {"sitting", "seated", "lying", "riding"}
    )
    non_object_anchor_words = (
        COLOR_WORDS | APPEARANCE_WORDS | IMPLICIT_OWNER_WORDS | ACTION_WORDS
        | HUMAN_WORDS | spatial_anchor_words
        | {"a", "an", "the", "of", "to", "from", "for", "in", "at", "by",
           "is", "are", "was", "were", "that", "who", "whose", "very"}
    )
    bare_support_anchor = bool(
        support_action
        and not support_anchor
        and 1 <= len(_words(visual_anchor)) <= 3
        and anchor_words - non_object_anchor_words
        and not (anchor_words & {"and", "or"})
    )
    valid_anchor_type = (
        support_anchor or bare_support_anchor
        if support_action else
        bool(anchor_words & spatial_anchor_words) or support_anchor
    )
    anchor_safe_words = anchor_selector_words | HUMAN_WORDS | {
        "a", "an", "the", "of", "to", "from", "for", "in", "at", "by",
        "is", "are", "was", "were", "that", "who", "whose", "very",
    }
    if (
        not anchor_allowed
        or not visual_anchor
        or len(_words(visual_anchor)) > 8
        or (not bare_support_anchor and not (anchor_words & anchor_selector_words))
        or not valid_anchor_type
        or (
            not (support_anchor or bare_support_anchor)
            and any(word not in anchor_safe_words and not word.isdigit() for word in anchor_words)
        )
        or bool(anchor_words & {"and", "or"})
        or _norm(visual_anchor) in {"empty", "none", "n a", "empty unless the query is headless"}
        or re.search(r"\b(?:target name|full description|therefore|answer)\b", visual_anchor, re.I)
    ):
        visual_anchor = ""
    elif bare_support_anchor:
        visual_anchor = f"on {visual_anchor}"
    if anatomy_anchor and visual_anchor:
        anatomy_words = _words(visual_anchor)
        anatomy_spatial = bool(
            1 <= len(anatomy_words) <= 3
            and set(anatomy_words) <= spatial_anchor_words | {"on", "in", "at", "the"}
            and set(anatomy_words) & spatial_anchor_words
        )
        anatomy_relation = bool(
            2 <= len(anatomy_words) <= 3
            and anatomy_words[0] in {
                "on", "near", "beside", "behind", "by", "holding", "carrying", "grasping"
            }
            and not set(anatomy_words[1:])
                & (HUMAN_WORDS | IMPLICIT_OWNER_WORDS | {"it", "one", "ones", "there", "his", "her", "their", "its", "and", "or"})
        )
        if not (anatomy_spatial or anatomy_relation):
            visual_anchor = ""
    confidence = _space(value.get("confidence")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "high" if target_span else "low"
    normalized_name = normalize_literal_spelling(name)
    if normalized_name != name:
        name = normalized_name
        warnings.append("target_spelling_normalized")
    candidates = []
    for item in value.get("candidates") or []:
        if not isinstance(item, (str, dict)):
            continue
        span_value = item if isinstance(item, str) else item.get("span")
        span = _exact_query_span(query, _space(span_value))
        candidate_name = normalize_literal_spelling(_clean_target_name(span))
        if not span or not candidate_name or len(_words(candidate_name)) > 6:
            continue
        candidates.append({
            "id": f"T{len(candidates) + 1}",
            "span": span,
            "target_name": candidate_name.lower(),
            "target_scope": "whole_object",
        })
    return {
        "target_name": name.lower(),
        "target_scope": scope,
        "source": source,
        "evidence_span": target_span,
        "target_span": target_span,
        "reference_span": reference_span,
        "visual_anchor": visual_anchor,
        "role": _space(value.get("role")).lower() or "unspecified",
        "confidence": confidence,
        "candidates": candidates,
        "parse_warnings": warnings,
    }, None


def parse_target_candidate_response(raw: str, candidates: List[dict]):
    value = extract_json(raw) or {}
    candidate_id = _space(value.get("candidate_id")).upper()
    allowed = {item["id"]: item for item in candidates}
    return allowed.get(candidate_id), None if candidate_id in allowed else "candidate_id"


def apply_target_candidate(target: dict, candidate: dict) -> dict:
    updated = dict(target)
    updated.update({
        "target_name": candidate["target_name"],
        "target_scope": "whole_object",
        "source": "explicit_query",
        "evidence_span": candidate["span"],
        "target_span": candidate["span"],
        "reference_span": "",
        "visual_anchor": "",
        "role": "candidate_id_lock",
        "confidence": "medium",
    })
    return updated


def parse_query_plan_response(raw: str, target_name: str, candidates: List[dict]) -> dict:
    value = extract_json(raw)
    items = value.get("reference_ids") if isinstance(value, dict) else None
    if not isinstance(items, list):
        start, end = str(raw or "").find("["), str(raw or "").rfind("]")
        try:
            items = json.loads(str(raw)[start:end + 1]) if 0 <= start < end else None
        except (TypeError, ValueError):
            items = None
    if not isinstance(items, list):
        return {"reference_objects": [], "parse_error": "query_plan_schema"}
    allowed = {
        _space(item.get("id")).upper(): item
        for item in candidates
        if isinstance(item, dict) and item.get("id") and item.get("query_phrase")
    }
    references = []
    for raw_id in items:
        candidate = allowed.get(_space(raw_id).upper())
        if not candidate:
            continue
        prompt = _space(candidate["query_phrase"]).strip(" ,.;:")
        relation = _space(candidate.get("relation"))
        prompt_words = _words(prompt)
        if not prompt_words or len(prompt_words) > 8:
            continue
        target_head = canonical_name(target_name).split()[-1:]
        prompt_head = canonical_name(prompt).split()[-1:]
        human_conflict = bool(
            target_head and target_head[0] in HUMAN_WORDS
            and prompt_head and prompt_head[0] in HUMAN_WORDS
        )
        owned_with = (
            _norm(relation) == "with"
            and target_head and target_head[0] in HUMAN_WORDS
            and bool(set(prompt_words) & IMPLICIT_OWNER_WORDS)
        )
        if human_conflict or owned_with or names_equivalent(prompt, target_name):
            continue
        references.append({
            "candidate_id": _space(candidate["id"]).upper(),
            "name": prompt.lower(),
            "sam_prompt": prompt.lower(),
            "relation": relation,
        })
        if len(references) == MAX_REFERENCE_OBJECTS:
            break
    return {"reference_objects": references, "parse_error": None}






def required_literal_cues(static_plan: dict, target_name: str,
                          query: Optional[str] = None,
                          target: Optional[dict] = None) -> List[str]:
    """Return typed query slots; never promote arbitrary target-span tokens."""
    del query
    target_head = canonical_name(target_name).split()[-1:]
    target_words = _words(canonical_name(target_name))
    color_keys = {_norm(item) for item in static_plan.get("color_phrases", [])}
    ordinal_key = _norm((static_plan.get("ordinal") or {}).get("evidence_span"))
    output = []
    cues = list(static_plan.get("literal_cues", []))
    for cue in cues:
        words = _words(cue)
        if ordinal_key and _norm(cue) == ordinal_key and target_words:
            width = len(target_words)
            match = next(
                (
                    index for index in range(len(words) - width + 1)
                    if names_equivalent(" ".join(words[index:index + width]), target_name)
                ),
                None,
            )
            if match is not None:
                words = words[:match] + words[match + width:]
                cue = " ".join(words)
        if (
            _norm(cue) in color_keys
            and len(words) > 1
            and target_head
            and any(
                canonical_name(word) in set(_words(canonical_name(target_name)))
                for word in words[-1:]
            )
        ):
            cue = " ".join(words[:-1])
        if cue and _norm(cue) not in {_norm(item) for item in output}:
            output.append(cue)
    return output


def lossless_fallback_description(query: str, target: dict) -> str:
    """Preserve the raw query after the locked prefix; no model-made cue survives."""
    cleaned = _space(query).strip(" .")
    name = target["target_name"]
    target_pattern = r"\s+".join(re.escape(word) for word in _words(name))
    side_location = re.match(
        rf"^(?P<side>left|right)\s+side\s+of\s+(?:(?:a|an|the)\s+)?{target_pattern}\b(?P<rest>.*)$",
        cleaned,
        re.I,
    ) if target_pattern else None
    if side_location:
        rest = _space(side_location.group("rest"))
        rest = re.sub(
            r"^in\s+(foreground|background)\b",
            r"in the \1",
            rest,
            flags=re.I,
        )
        return f"The {name} on the {side_location.group('side').lower()} side{(' ' + rest) if rest else ''}."
    match = re.search(
        r"\b" + target_pattern + r"\b",
        cleaned,
        re.I,
    ) if target_pattern else None
    if not match and target.get("role") == "implicit_owner_lock":
        proxy = _space(target.get("owner_proxy_span"))
        if proxy:
            anchor = _space(target.get("visual_anchor"))
            suffix = f", {anchor}" if anchor else ""
            return f"The {name} with {cleaned}{suffix}."
        anchor = _space(target.get("visual_anchor"))
        suffix = f", {anchor}" if anchor else ""
        return f"The {name} that is {cleaned}{suffix}."
    if match:
        before = re.sub(r"^(?:a|an|the)\s+", "", cleaned[:match.start()].strip(), flags=re.I)
        after = cleaned[match.end():].strip(" ,")
        if before and not re.search(r"\b(?:of|by|with|to|from)\s*(?:a|an|the)?$", before, re.I):
            tail = f", {after}" if after else ""
            copula = "are" if _words(name)[-1] in {"pants", "shorts", "people", "men", "women"} else "is"
            return f"The {name} that {copula} {before}{tail}."
        if not before:
            return f"The {name}{(' ' + after) if after else ''}."
    return f"The {name}; specifically, {cleaned}."


def normalize_support_relation(description: str, static_plan: dict) -> str:
    """Disambiguate a structurally detected half-on-half support relation."""
    if not static_plan.get("support_reference"):
        return description
    return re.sub(
        r"\bon\s+(?:the\s+)?other\s+(half|part)\b",
        r"on top of the other \1",
        description,
        flags=re.I,
    )


def normalize_side_location(query: str, target: dict, description: str,
                            static_plan: dict) -> str:
    """Resolve the grammar pattern `left/right side of X` as located X."""
    if not static_plan.get("side_location") or target.get("target_scope") != "whole_object":
        return description
    return lossless_fallback_description(query, target)


def normalize_literal_spelling(description: str) -> str:
    """Canonicalize known query typos before literal-cue validation."""
    substitutions = {
        "visibile": "visible", "visiable": "visible", "behidn": "behind",
        "dudue": "dude", "giraffee": "giraffe", "cuchine": "cushion",
    }
    return re.sub(
        r"\b(?:visibile|visiable|behidn|dudue|giraffee|cuchine)\b",
        lambda match: substitutions[match.group(0).lower()],
        description,
        flags=re.I,
    )


def normalize_description_prefix(description: str, target_name: str) -> str:
    """Move leading modifiers behind the mandatory `The <target_name>` prefix."""
    description = _space(description)
    target_words = _words(target_name)
    if not description or not target_words:
        return description
    required = ["the"] + _words(target_name)
    if _words(description)[:len(required)] == required:
        target_pattern = r"\s+".join(re.escape(word) for word in target_words)
        return re.sub(
            rf"^the\s+{target_pattern}\s+(is|are|was|were)\b",
            lambda match: f"The {target_name} that {match.group(1).lower()}",
            description,
            count=1,
            flags=re.I,
        )
    target_pattern = r"\s+".join(re.escape(word) for word in target_words)
    match = re.match(
        rf"^the\s+(?P<mods>(?:[a-z0-9'-]+\s+){{1,4}}){target_pattern}\b(?P<rest>.*)$",
        description,
        re.I,
    )
    if not match:
        return description
    modifiers = _space(match.group("mods"))
    rest = _space(match.group("rest"))
    normalized = _space(f"The {target_name} that is {modifiers} {rest}")
    return re.sub(r"\s+([,.;:!?])", r"\1", normalized)


def ensure_scope_cue(description: str, target_name: str, static_plan: dict) -> str:
    if static_plan.get("scope_hint") not in {"part", "region"}:
        return description
    scope_label = _space(static_plan.get("scope_label"))
    target_pattern = r"\s+".join(re.escape(word) for word in _words(target_name))
    possessive_scope = scope_label in {"part", "body", "back", "rear"}
    if possessive_scope and target_pattern:
        description = re.sub(
            rf"^The\s+{target_pattern}\s+({re.escape(scope_label)})\b",
            rf"The {target_name} whose \1",
            description,
            count=1,
            flags=re.I,
        )
    if scope_label and _contains_token_sequence(description, scope_label):
        return description
    cues = static_plan.get("scope_cues") or [static_plan["scope_hint"]]
    if any(_contains_token_sequence(description, cue) for cue in cues):
        return description
    prefix = f"The {target_name}"
    if possessive_scope and description.lower().startswith(prefix.lower()):
        description = f"{prefix} whose {cues[0]}{description[len(prefix):]}"
    return re.sub(r"\s+([,.;:!?])", r"\1", _space(description))




def sanitize_unsupported_ordinals(query: str, description: str) -> str:
    ordinal_pattern = r"\b(?:first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th))\b"
    if re.search(ordinal_pattern, query, re.I):
        return description
    description = re.sub(ordinal_pattern, "", description, flags=re.I)
    description = re.sub(r"\s+([,.;:!?])", r"\1", _space(description))
    return description


def _unsupported_structural_cues(query: str, description: str) -> List[str]:
    families = {
        "ordinal": r"\b(?:first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th))\b",
        "spatial": r"\b(?:left|right|top|bottom|front|back|leftmost|rightmost|topmost|bottommost)\b",
        "comparison": r"\b(?:closer|closest|farther|farthest|higher|highest|lower|lowest|bigger|biggest|smaller|smallest|taller|tallest|shorter|shortest)\b",
        "negation": r"\b(?:no|not|without|neither|never)\b",
        "count": r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
    }
    return [
        name
        for name, pattern in families.items()
        if re.search(pattern, description, re.I) and not re.search(pattern, query, re.I)
    ]


def parse_ordinal_response(raw: str, width: int, height: int):
    value = extract_json(raw)
    if value is None:
        return None, None, "ordinal_json"
    candidate = _space(value.get("candidate_id")).upper()
    if candidate not in {"A", "B", "C"}:
        return None, None, "ordinal_candidate"
    box = value.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return candidate, None, "ordinal_bbox"
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return candidate, None, "ordinal_bbox"
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return candidate, None, "ordinal_bbox"
    return candidate, [x1, y1, x2, y2], None


def description_violations(query: str, target: dict, description: str, static_plan: dict) -> List[str]:
    problems = []
    required_prefix = _words(f"the {target['target_name']}")
    if _words(description)[:len(required_prefix)] != required_prefix:
        problems.append("target_prefix")
    for cue in required_literal_cues(static_plan, target["target_name"], query, target):
        if not _contains_token_sequence(description, cue):
            problems.append(f"cue_lost:{_space(cue)}")
    owner_proxy = _space(target.get("owner_proxy_span"))
    if owner_proxy and not _contains_selector_content(
        description, owner_proxy, target["target_name"]
    ):
        problems.append(f"owner_proxy_lost:{owner_proxy}")
    visual_anchor = _space(target.get("visual_anchor"))
    if visual_anchor and not _contains_selector_content(
        description, visual_anchor, target["target_name"]
    ):
        problems.append(f"visual_anchor_lost:{visual_anchor}")
    reference_span = _space(target.get("reference_span"))
    if reference_span and not _contains_reference_content(description, reference_span):
        problems.append(f"reference_lost:{reference_span}")
    supported_text = _space(f"{query} {target.get('visual_anchor', '')}")
    if static_plan.get("support_reference"):
        supported_text += " top"
    problems.extend(
        f"unsupported_{family}"
        for family in _unsupported_structural_cues(supported_text, description)
    )
    query_words, description_words = set(_words(supported_text)), set(_words(description))
    for word in sorted((description_words - query_words) & COLOR_WORDS):
        problems.append(f"unsupported_color:{word}")
    for word in sorted((description_words - query_words) & ACTION_WORDS):
        if word == "wearing" and query_words & IMPLICIT_OWNER_WORDS:
            continue
        problems.append(f"unsupported_action:{word}")
    if re.search(r"\b(?:therefore|the answer|target name is|full description)\b", description, re.I):
        problems.append("meta_output")
    return sorted(set(problems))




reasoning_prompts.update({
    "v4_target_name": TARGET_NAME_PROMPT,
    "v4_target_candidate": TARGET_CANDIDATE_PROMPT,
    "v4_query_plan": QUERY_PLAN_PROMPT,
    "v4_full_description": FULL_DESCRIPTION_PROMPT,
    "v4_final_bbox": FINAL_BBOX_PROMPT,
    "v4_contract_bbox": CONTRACT_BBOX_PROMPT,
    "v4_ordinal_bbox": ORDINAL_BBOX_PROMPT,
})
