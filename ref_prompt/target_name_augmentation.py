"""Generate conservative SAM3 target-name candidates for RefSeg."""

import json
import re


SYSTEM_AUGMENT = (
    "Generate at most three short visual concepts for one unchanged SAM3 target. "
    "Preserve identity and mask extent. Return one JSON object only."
)

INVALID_NAMES = {
    "", "none", "object", "thing", "item", "entity", "area", "stuff", "noun",
}
POSITION_WORDS = {
    "left", "right", "top", "bottom", "upper", "lower", "front", "back",
    "background", "foreground", "middle", "center", "centre", "first", "second",
    "third", "last", "next", "leftmost", "rightmost",
}
COLOR_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "brown", "gray",
    "grey", "orange", "purple", "pink", "tan", "gold", "golden", "silver",
}
COLOR_QUALIFIERS = {"light", "dark", "bright", "pale"}
PATTERN_WORDS = {
    "striped", "spotted", "plaid", "checked", "checkered", "patterned",
    "floral", "dotted", "polka-dot",
}
MATERIAL_WORDS = {
    "wood", "wooden", "metal", "metallic", "plastic", "glass", "ceramic",
    "leather", "denim", "stone", "brick", "concrete", "straw", "paper",
    "cardboard", "rubber", "fabric", "cloth",
}
SIZE_WORDS = {
    "small", "large", "big", "tiny", "tall", "short", "long", "wide",
    "narrow", "thick", "thin", "mini", "oversized",
}
STATE_WORDS = {
    "open", "closed", "empty", "full", "ripe", "cooked", "raw", "broken",
    "wet", "dry", "folded", "unfolded", "cut", "sliced", "peeled", "whole",
    "blurred",
}
HUMAN_TARGETS = {
    "person", "people", "human", "humans", "man", "men", "woman", "women",
    "boy", "boys", "girl", "girls", "child", "children", "kid", "kids",
    "player", "players", "athlete", "athletes", "rider", "riders", "worker",
    "workers", "chef", "chefs", "officer", "officers", "bride", "groom",
}
CLOTHING_NOUNS = {
    "shirt", "shirts", "jacket", "jackets", "coat", "coats", "dress", "dresses",
    "suit", "suits", "uniform", "uniforms", "jersey", "jerseys", "sweater",
    "sweaters", "hoodie", "hoodies", "top", "tops", "pants", "shorts", "skirt",
    "skirts", "jeans", "trousers", "hat", "hats", "cap", "caps", "helmet",
    "helmets", "tie", "ties", "scarf", "scarves", "vest", "vests", "robe",
    "robes", "clothes", "clothing", "outfit", "outfits", "attire", "garment",
    "garments", "sleeve", "sleeves", "tshirt", "t-shirt", "tee", "tees",
}
GARMENT_DESCRIPTORS = {
    "short", "long", "short-sleeve", "long-sleeve", "sport", "sports",
    "athletic", "casual", "dress", "polo", "button", "buttoned", "sleeveless",
}
WEARER_BRIDGE_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "left", "right", "top",
    "bottom", "center", "centre", "middle", "front", "back", "foreground",
    "background", "closest", "nearest", "near", "far", "side", "standing",
    "sitting", "walking", "lying", "kneeling", "crouching", "running",
}
SCOPE_MAP = {
    "instance": "whole_object",
    "collection": "group",
    "part": "part",
    "region": "region",
}
TARGET_NAME_MAX_WORDS = 5


def augment_names_prompt(full_description: str, target_name: str, target_scope: str) -> str:
    return f"""FULL DESCRIPTION: {full_description}
LOCKED TARGET: {target_name}
LOCKED SCOPE: {target_scope}

SAM3 understands short visual concepts, not referring sentences. Produce lexical
components for at most three final concepts while keeping the same physical target,
scope, number, and mask extent. Prefer one to three effective words per concept;
allow four or five only for a necessary compound plus one visual modifier.

1. exact is always LOCKED TARGET.
2. canonical is one of:
   - same_class_alias: a familiar synonym naming the same visual class and extent;
   - broader_category: one nearby recognition category used only to rescue an
     uncommon subtype, role, demographic, named variety, or specialized term;
   - repeat: copy LOCKED TARGET when no safe distinct category exists.
   Never broaden a target that is already a common directly segmentable category
   merely to create lexical variety. Never use a sibling or related object class.
3. Atomic is always built from LOCKED TARGET and at most one target-owned
   visible modifier. canonical and alias must never become its grammatical head.
   Supply the lexical components below.

alias must be a modern, unambiguous same-class substitute with the same extent.
It must never be a broader category. Use an empty string when none exists.

modifier must be one or two consecutive attribute words copied exactly from FULL
DESCRIPTION. modifier_owner is direct_target only for an intrinsic target color,
pattern, material, size, or simple visible state in one explicit local form:
modifier plus the complete LOCKED TARGET; LOCKED TARGET that/is modifier;
LOCKED TARGET specifically modifier; or LOCKED TARGET covered in modifier with
no following content noun. It is target_worn only for a visible color or pattern
when the complete LOCKED TARGET occurs before the wearing cue: in/wearing plus modifier
(optionally followed immediately by a clothing noun), or with plus modifier plus a clothing noun.
The clothing noun is ownership evidence only and is not part of modifier. Never
assign target_worn to hair, backpacks, luggage, sports equipment, or a later
reference object. Use none and an empty modifier whenever ownership is uncertain.

Never use positions, directions, ordinals, counts, actions, relations, scene
terms, reference objects, explanations, duplicate words, or unsupported
attributes. Never use generic placeholders such as object, thing, item, entity,
area, stuff, or noun.

Return exactly one JSON object and no markdown:
{{"exact":"{target_name}","canonical":"","canonical_relation":"same_class_alias|broader_category|repeat","alias":"","modifier_type":"color|pattern|material|size|simple_state|wearing_color|none","modifier":"","modifier_owner":"direct_target|target_worn|none"}}"""


def _normalize(value) -> str:
    return " ".join(str(value or "").strip().split()).strip(" ,.;:[]{}()\"'")[:100]


def _valid_name(value, max_words=None) -> bool:
    max_words = TARGET_NAME_MAX_WORDS if max_words is None else max_words
    text = _normalize(value)
    words = re.findall(r"[A-Za-z0-9]+", text)
    return (
        1 <= len(words) <= max_words
        and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 '\-]{0,99}", text))
        and not all(word.isdigit() for word in words)
        and text.lower() not in INVALID_NAMES
        and not set(word.lower() for word in words) <= POSITION_WORDS
    )


def _extract_json(raw: str):
    match = re.search(r"\{[\s\S]*\}", raw or "")
    if match is None:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _modifier_matches_type(modifier_type: str, modifier: str) -> bool:
    words = set(re.findall(r"[a-z0-9-]+", modifier.lower()))
    if modifier_type in {"color", "wearing_color"}:
        return bool(words & COLOR_WORDS) and words <= COLOR_WORDS | COLOR_QUALIFIERS
    if modifier_type == "pattern":
        return bool(words & PATTERN_WORDS) and words <= PATTERN_WORDS | COLOR_WORDS
    if modifier_type == "material":
        return bool(words) and words <= MATERIAL_WORDS
    if modifier_type == "size":
        return bool(words) and words <= SIZE_WORDS
    if modifier_type == "simple_state":
        return bool(words) and words <= STATE_WORDS
    return False


def _modifier_is_grounded(full_description: str, target_name: str, modifier: str, owner: str) -> bool:
    full = " ".join(full_description.lower().split())
    target = re.escape(_normalize(target_name).lower()).replace(r"\ ", r"\s+")
    attribute = re.escape(_normalize(modifier).lower()).replace(r"\ ", r"\s+")
    if owner == "direct_target":
        patterns = (
            rf"\b{attribute}\s+{target}\b",
            rf"\b{target}\s+(?:(?:that\s+)?is\s+)?{attribute}\b",
            rf"\b{target}\s+(?:(?:that\s+)?is\s+)?covered\s+in\s+{attribute}\b",
        )
        return any(re.search(pattern, full) for pattern in patterns)
    head = re.findall(r"[a-z0-9]+", target_name.lower())[-1:]
    if owner != "target_worn" or not head or head[0] not in HUMAN_TARGETS:
        return False
    pattern = re.compile(
        rf"\b{target}\b(?P<bridge>.*?)\b(?P<cue>in|wearing|with)\s+"
        rf"(?:(?:a|an|the)\s+)?{attribute}\b(?P<suffix>.*)$"
    )
    for match in pattern.finditer(full):
        bridge_words = re.findall(r"[a-z0-9-]+", match.group("bridge"))
        if any(word not in WEARER_BRIDGE_WORDS for word in bridge_words):
            continue
        suffix = match.group("suffix").strip()
        if not suffix:
            return match.group("cue") != "with"
        if suffix.startswith((",", ";")):
            continue
        suffix_words = re.findall(r"[a-z0-9-]+", suffix)[:3]
        for word in suffix_words:
            if word in CLOTHING_NOUNS:
                return True
            if word not in GARMENT_DESCRIPTORS:
                break
    return False


def parse_target_name_candidates(raw: str, target_name: str, full_description: str):
    target = _normalize(target_name)
    value = _extract_json(raw)
    if value is None:
        return [target]

    candidates = [target]
    relation = str(value.get("canonical_relation") or "").strip().lower()
    canonical = _normalize(value.get("canonical"))
    alias = _normalize(value.get("alias"))
    if relation == "repeat":
        canonical = target
    elif relation not in {"same_class_alias", "broader_category"}:
        canonical = ""
    second = canonical if _valid_name(canonical) else alias if _valid_name(alias) else target
    if second.lower() not in {candidate.lower() for candidate in candidates}:
        candidates.append(second)

    modifier_type = str(value.get("modifier_type") or "none").strip().lower()
    modifier_owner = str(value.get("modifier_owner") or "none").strip().lower()
    modifier = _normalize(value.get("modifier"))
    if (
        _valid_name(modifier, max_words=2)
        and _modifier_matches_type(modifier_type, modifier)
        and _modifier_is_grounded(full_description, target, modifier, modifier_owner)
    ):
        atomic = (
            f"{target} in {modifier}"
            if modifier_owner == "target_worn"
            else f"{modifier} {target}"
        )
        if atomic.lower() not in {candidate.lower() for candidate in candidates}:
            candidates.append(atomic)
    return candidates[:3]


def generate_target_name_candidates(qwen, full_description: str, target_name: str, target_scope: str):
    scope = SCOPE_MAP.get(target_scope, target_scope)
    raw = qwen.generate(
        augment_names_prompt(full_description, target_name, scope),
        sys_prompt=SYSTEM_AUGMENT,
        use_ori_image=False,
    ).strip()
    return parse_target_name_candidates(raw, target_name, full_description)
