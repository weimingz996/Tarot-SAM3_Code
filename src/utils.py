import os
import datetime
import re
from pathlib import Path
from types import SimpleNamespace
import cv2
import yaml
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
from typing import List



def save_mask(mask, save_path):
    mask = np.asarray(mask)
    if mask.dtype == bool or (mask.size > 0 and mask.max() <= 1):
        mask = (mask.astype(np.uint8) * 255)
    else:
        mask = mask.astype(np.uint8)
    cv2.imwrite(save_path, mask)


def _normalize_visual_mask(mask, shape):
    value = np.asarray(mask).squeeze().astype(bool)
    if value.shape != tuple(shape):
        raise ValueError(f"mask/image shape mismatch: {value.shape} vs {tuple(shape)}")
    return value


def _visual_bgr_image(image):
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected BGR image with shape HxWx3, got {value.shape}")
    return value


def _visual_rgb_image(image):
    return cv2.cvtColor(_visual_bgr_image(image), cv2.COLOR_BGR2RGB)


def save_labeled_mask_overlay(image, mask, label, output_path, color):
    source = Image.fromarray(_visual_rgb_image(image), "RGB").convert("RGBA")
    mask = _normalize_visual_mask(mask, (source.height, source.width))
    rgba = np.zeros((source.height, source.width, 4), dtype=np.uint8)
    rgba[mask, :3] = color
    rgba[mask, 3] = 112
    merged = Image.alpha_composite(source, Image.fromarray(rgba, "RGBA"))
    draw = ImageDraw.Draw(merged)
    box = draw.textbbox((0, 0), label)
    draw.rectangle((5, 5, box[2] + 15, box[3] + 15), fill=(0, 0, 0, 220))
    draw.text((10, 10), label, fill=(255, 255, 255, 255))
    merged.convert("RGB").save(output_path, quality=92)


def _labeled_cv_panel(body, label, header):
    height, width = body.shape[:2]
    value = np.zeros((height + header, width, 3), dtype=np.uint8)
    value[header:] = body
    y = 32 if header == 48 else 41
    scale = 0.55 if header == 48 else 0.62
    cv2.putText(
        value, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return value


def save_full_long_short_bbox_board(image, bboxes, output_path):
    image = _visual_bgr_image(image)
    height, width = image.shape[:2]
    header = max(48, int(round(height * 0.06)))
    font_scale = max(0.55, min(height, width) / 800)
    thickness = max(3, int(round(min(height, width) * 0.003)))
    panels = []
    for method in ("full", "long", "short"):
        panel = image.copy()
        bbox = bboxes.get(method)
        label = method.title()
        if bbox is None:
            label += ": None"
        else:
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
            cv2.rectangle(
                panel, (x1, y1), (x2, y2), (0, 0, 255), thickness,
                cv2.LINE_AA,
            )
        labeled = np.zeros((height + header, width, 3), dtype=np.uint8)
        labeled[header:] = panel
        cv2.putText(
            labeled,
            label,
            (12, int(header * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(2, thickness // 2),
            cv2.LINE_AA,
        )
        panels.append(labeled)

    output_path = Path(output_path)
    if not cv2.imwrite(str(output_path), np.hstack(panels)):
        raise RuntimeError(f"failed to write bbox board: {output_path}")
    return output_path


def save_mask_comparison_board(image, baseline_mask, proposal_mask, output_path):
    image = _visual_bgr_image(image).copy()
    shape = image.shape[:2]
    baseline = _normalize_visual_mask(baseline_mask, shape)
    proposal = _normalize_visual_mask(proposal_mask, shape)

    def cutout(mask):
        value = np.zeros_like(image)
        value[mask] = image[mask]
        return value

    common = np.logical_and(baseline, proposal)
    baseline_only = np.logical_and(baseline, np.logical_not(proposal))
    proposal_only = np.logical_and(proposal, np.logical_not(baseline))
    top = np.hstack([
        _labeled_cv_panel(image, "ORIGINAL", 48),
        _labeled_cv_panel(cutout(baseline), "BASELINE MASK", 48),
        _labeled_cv_panel(cutout(proposal), "PROPOSAL MASK", 48),
    ])
    bottom = np.hstack([
        _labeled_cv_panel(cutout(common), "COMMON PIXELS", 48),
        _labeled_cv_panel(cutout(baseline_only), "REMOVED BY PROPOSAL", 48),
        _labeled_cv_panel(cutout(proposal_only), "ADDED BY PROPOSAL", 48),
    ])
    if not cv2.imwrite(str(output_path), np.vstack([top, bottom])):
        raise RuntimeError(f"failed to write temporary board: {output_path}")




def _semantic_mask_bodies(image, mask):
    outline = image.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(outline, contours, -1, (255, 255, 0), 3, cv2.LINE_AA)
    cutout = np.zeros_like(image)
    cutout[mask] = image[mask]
    return outline, cutout


def save_semantic_mask_views(image, baseline_mask, proposal_mask, output_dir):
    image = _visual_bgr_image(image).copy()
    output_dir = Path(output_dir)
    paths = []
    for mask, prefix in (
        (_normalize_visual_mask(baseline_mask, image.shape[:2]), "BASELINE"),
        (_normalize_visual_mask(proposal_mask, image.shape[:2]), "PROPOSAL"),
    ):
        outline, cutout = _semantic_mask_bodies(image, mask)
        for suffix, body, label in (
            ("outline", outline, f"{prefix}: EXACT MASK OUTLINE ON ORIGINAL"),
            ("cutout", cutout, f"{prefix}: SAME-COORDINATE BLACK CUTOUT"),
        ):
            target = output_dir / f"{prefix.lower()}_{suffix}.png"
            if not cv2.imwrite(str(target), _labeled_cv_panel(body, label, 64)):
                raise RuntimeError(f"failed to write temporary view: {target}")
            paths.append(str(target))
            cv2.imwrite(f"{prefix.lower()}_{suffix}.png", _labeled_cv_panel(body, label, 64))
    return paths


def save_absolute_mask_views(image, candidate_mask, output_dir):
    image = _visual_bgr_image(image).copy()
    mask = _normalize_visual_mask(candidate_mask, image.shape[:2])
    outline, cutout = _semantic_mask_bodies(image, mask)
    panels = (
        ("image_1_original.png", image, "IMAGE 1: ORIGINAL"),
        ("image_2_outline.png", outline, "IMAGE 2: EXACT CANDIDATE MASK CONTOUR"),
        ("image_3_cutout.png", cutout, "IMAGE 3: SAME-COORDINATE BLACK CUTOUT"),
    )
    paths = []
    for filename, body, label in panels:
        target = Path(output_dir) / filename
        if not cv2.imwrite(str(target), _labeled_cv_panel(body, label, 64)):
            raise RuntimeError(f"failed to write temporary view: {target}")
        paths.append(str(target))
    return paths


def save_target_candidate_views(
    image, items, target_name, overview_path, montage_path, palette,
):
    source_image = Image.fromarray(_visual_rgb_image(image), "RGB")
    source_array = np.asarray(source_image)
    overview_array = (source_array.astype("float32") * 0.78).astype("uint8")
    panels = []
    for index, item in enumerate(items, 1):
        x1, y1, x2, y2 = item["bbox"]
        pad = max(12, int(0.60 * max(x2 - x1, y2 - y1)))
        left = max(0, int(x1) - pad)
        top = max(0, int(y1) - pad)
        right = min(source_image.width, int(x2 + 0.999) + pad)
        bottom = min(source_image.height, int(y2 + 0.999) + pad)
        crop = source_array[top:bottom, left:right].copy()
        mask = _normalize_visual_mask(
            item["mask"], (source_image.height, source_image.width)
        )
        color = np.asarray(palette[(index - 1) % len(palette)], dtype="float32")
        overview_array[mask] = (
            0.70 * overview_array[mask].astype("float32") + 0.30 * color
        ).astype("uint8")
        crop_mask = mask[top:bottom, left:right]
        highlighted = crop.copy()
        orange = np.zeros_like(highlighted)
        orange[:, :, 0] = 255
        orange[:, :, 1] = 128
        highlighted[crop_mask] = (
            0.52 * highlighted[crop_mask] + 0.48 * orange[crop_mask]
        ).astype(highlighted.dtype)
        candidate = Image.fromarray(highlighted, "RGB")
        resampling = getattr(Image, "Resampling", Image)
        candidate.thumbnail((280, 190), resampling.LANCZOS)
        panel = Image.new("RGB", (300, 240), "black")
        panel.paste(candidate, ((300 - candidate.width) // 2, 42))
        ImageDraw.Draw(panel).text(
            (12, 10), f"TARGET {item['id']}: {target_name} (orange mask)",
            fill=(255, 190, 80),
        )
        panels.append(panel)
        candidate.close()

    overview = Image.fromarray(overview_array, "RGB")
    draw = ImageDraw.Draw(overview)
    for index, item in enumerate(items, 1):
        color = palette[(index - 1) % len(palette)]
        x1, y1, x2, y2 = [int(round(value)) for value in item["bbox"]]
        for inset in range(3):
            if x1 + inset > x2 - inset or y1 + inset > y2 - inset:
                break
            draw.rectangle(
                (x1 + inset, y1 + inset, x2 - inset, y2 - inset),
                fill=None, outline=color,
            )
        label_x, label_y = max(0, x1), max(0, y1 - 18)
        draw.rectangle(
            (label_x, label_y, label_x + 34, label_y + 17), fill=(0, 0, 0)
        )
        draw.text((label_x + 3, label_y + 2), item["id"], fill=color)
    overview.save(overview_path, quality=94)
    overview.close()
    source_image.close()

    columns = 1 if len(panels) == 1 else 2
    rows = (len(panels) + columns - 1) // columns
    montage = Image.new("RGB", (columns * 300, rows * 240), "black")
    for index, panel in enumerate(panels):
        montage.paste(panel, ((index % columns) * 300, (index // columns) * 240))
    montage.save(montage_path, quality=92)
    montage.close()
    for panel in panels:
        panel.close()


def save_selected_target_views(image, mask, bbox, output_dir):
    source = Image.fromarray(_visual_rgb_image(image), "RGB")
    source_array = np.asarray(source)
    mask = _normalize_visual_mask(mask, (source.height, source.width))
    orange = np.zeros_like(source_array)
    orange[:, :, 0] = 255
    orange[:, :, 1] = 128

    context = (source_array.astype("float32") * 0.42).astype("uint8")
    context[mask] = (
        0.48 * source_array[mask].astype("float32")
        + 0.52 * orange[mask].astype("float32")
    ).astype("uint8")
    output_dir = Path(output_dir)
    context_path = output_dir / "selected_target_context.jpg"
    Image.fromarray(context, "RGB").save(context_path, quality=95)

    focus = np.full_like(source_array, 12)
    focus[mask] = source_array[mask]
    focus_path = output_dir / "selected_target_focus.jpg"
    Image.fromarray(focus, "RGB").save(focus_path, quality=95)

    x1, y1, x2, y2 = bbox
    pad = max(8, int(0.12 * max(x2 - x1, y2 - y1)))
    left, top = max(0, int(x1) - pad), max(0, int(y1) - pad)
    right = min(source.width, int(x2 + 0.999) + pad)
    bottom = min(source.height, int(y2 + 0.999) + pad)
    clean_crop = source_array[top:bottom, left:right].copy()
    clean_mask = mask[top:bottom, left:right]
    clean = (clean_crop.astype("float32") * 0.22).astype("uint8")
    clean[clean_mask] = clean_crop[clean_mask]
    clean_path = output_dir / "selected_target_clean.jpg"
    Image.fromarray(clean, "RGB").save(clean_path, quality=95)
    source.close()
    return {
        "context": str(context_path), "focus": str(focus_path),
        "clean_crop": str(clean_path),
    }


class ExperimentLogger:
    def __init__(self, log_dir, fname, resume=False):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, f"{fname}.log")
        mode = 'a' if resume and os.path.exists(self.filepath) else 'w'
        self.start_time = datetime.datetime.now()
        with open(self.filepath, mode, encoding='utf-8') as f:
            if not resume:
                f.write(f"=== Log for {fname} ===\n")
                f.write(f"Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
            else:
                f.write(f"\n\n>>> Resumed at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} <<<\n")
                f.write("-" * 50 + "\n")

    def log(self, phase, content):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] [{phase}] {content}"
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(entry + "\n")

    def log_raw(self, content):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(content + "\n")


def dict_to_namespace(d):
    x = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(x, k, dict_to_namespace(v))
        else:
            setattr(x, k, v)
    return x


def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)




def extract_bbox_from_mask(mask: np.ndarray) -> List[int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    try:
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
    except IndexError:
        return [0, 0, 0, 0]
    return [x0, y0, x1, y1]










def clean_output(raw: str):
    match = re.search(r'\{([^{}]+)\}', raw, re.DOTALL)
    if match: return match.group(1).strip()
    clean = raw.strip()
    if ":" in clean: parts = clean.split(":", 1); clean = parts[1] if len(parts[1]) > 1 else parts[0]
    clean = clean.replace('*', '').replace('_', '')
    clean = clean.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    clean = clean.replace('"', '').replace("'", "")
    clean = re.sub(r'^[\d\.\-\s\•]+', '', clean)
    return clean.strip()


def split_phases(raw_text: str):
    normalized_text = raw_text.replace(',', '\n')
    results = []
    for line in normalized_text.split('\n'):
        clean = clean_output(line)
        if len(clean) > 2 and "Here are" not in clean and "Output" not in clean:
            results.append(clean)
    return deduplicate_prompts(results)


def deduplicate_prompts(prompts: list):
    seen = set()
    unique = []
    for p in prompts:
        if not isinstance(p, str): continue
        norm = p.lower().strip()
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(p)
    return unique
