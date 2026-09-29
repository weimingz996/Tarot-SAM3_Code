from typing import Optional, List, Dict, Any
import base64
import mimetypes
import cv2
import numpy as np
import torch
from PIL import Image
from openai import OpenAI
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from src.utils import extract_bbox_from_mask


class QwenEngine:
    _instance = None

    def __new__(cls, qwen_cfg, timeout=120, max_tokens=256, temperature=0.0):
        if cls._instance is None:
            cls._instance = super(QwenEngine, cls).__new__(cls)
            cls._instance.qwen_cfg = qwen_cfg
            cls._instance.timeout = timeout
            cls._instance.max_tokens = max_tokens
            cls._instance.temperature = temperature
            cls._instance._load_model()
            cls._instance.context = []
            cls._instance.ori_image = None
        return cls._instance

    def _load_model(self):
        # print(f"[Qwen] Loading from {self.qwen_cfg.model_dir}...")
        # try:
        #     torch.cuda.empty_cache()
        #     self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #         self.qwen_cfg.model_dir,
        #         torch_dtype="auto",
        #         device_map="auto",
        #         trust_remote_code=True,
        #         attn_implementation="eager"
        #     )
        # except Exception as e:
        #     print(f"[Qwen Error] Failed to load model. Error: {e}")
        #     raise e
        # self.processor = AutoProcessor.from_pretrained(self.qwen_cfg.model_dir, trust_remote_code=True)
        # self.processor.tokenizer.padding_side = "left"
        # print("[Qwen] Ready.")
        self.client = OpenAI(
            base_url=self.qwen_cfg.base_url,
            api_key=self.qwen_cfg.api_key,
            timeout=self.timeout,
        )

    @staticmethod
    def _file_to_data_url(path: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "application/octet-stream"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def load_image(self, img_path: str):
        self.ori_image = self._file_to_data_url(img_path)

    @torch.no_grad()
    def generate(self, prompt: str, image_paths: List[str] = [], sys_prompt: Optional[str] = None,
                 use_context=False, init_context=False, use_ori_image: bool = True) -> str:
        content = [{
            "type": "image_url",
            "image_url": {"url": self.ori_image}
        }] if use_ori_image else[]
        for img_path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._file_to_data_url(img_path)}
            })
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        if init_context:
            self.context = []
        if use_context:
            messages = self.context + messages

        if sys_prompt:
            messages = [{"role": "system", "content": sys_prompt}] + messages

        kwargs = dict(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.qwen_cfg.model:
            kwargs["model"] = self.qwen_cfg.model

        resp = self.client.chat.completions.create(**kwargs)
        answer = (resp.choices[0].message.content or "").strip()

        if use_context:
            self.context.append({"role": "user", "content": content})
            self.context.append({"role": "assistant", "content": answer})
        answer = self.clean_text(answer)
        return answer

    def clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = text.replace("addCriterion", " ")
        return text


class SAM3Engine:
    _instance = None

    @staticmethod
    def _stability_scores(mask_values: np.ndarray, threshold: float, delta: float = 0.05) -> np.ndarray:
        values = np.asarray(mask_values)
        flat = values.reshape(values.shape[0], int(np.prod(values.shape[1:])))
        area_i = np.sum(flat > threshold + delta, axis=1)
        area_u = np.sum(flat > threshold - delta, axis=1)
        return np.divide(
            area_i,
            area_u,
            out=np.ones(area_i.shape, dtype=float),
            where=area_u > 0,
        )

    def __new__(cls, ckpt_path: Optional[str] = None, device: str = "cuda", conf_thresh: float = 0.6):
        if cls._instance is None:
            cls._instance = super(SAM3Engine, cls).__new__(cls)
            cls._instance.ckpt_path = ckpt_path
            cls._instance.device = device
            cls._instance.conf_thresh = conf_thresh
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        print(f"[SAM3] Loading from {self.ckpt_path}...")
        torch.cuda.empty_cache()
        self.model = build_sam3_image_model(checkpoint_path=self.ckpt_path, load_from_HF=False, device=self.device,
                                            enable_inst_interactivity=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor(self.model, device=self.device)
        print(f"[SAM3] Ready on {self.device}.")

    def load_image(self, image_path: str):
        self.image_pil = Image.open(image_path).convert("RGB")
        with torch.cuda.device(self.device):
            self.inference_state = self.processor.set_image(self.image_pil)

    def predict_text(self, text_prompt: str) -> List[Dict[str, Any]]:
        clean_prompt = text_prompt.split("\n")[0].strip()[:100]
        self.processor.reset_all_prompts(self.inference_state)
        if not clean_prompt: return []
        with torch.cuda.device(self.device):
            output = self.processor.set_text_prompt(state=self.inference_state, prompt=clean_prompt)
            masks, scores = output["masks"], output["scores"]
            results = []
            masks_np, scores_np = masks.cpu().numpy() > 0, scores.cpu().to(torch.float).numpy()
            mask_logits = output["masks_logits"].cpu().to(torch.float).numpy()
            stability_scores = self._stability_scores(mask_logits, threshold=0.5)
            for i in range(len(scores_np)):
                m, s = masks_np[i], float(scores_np[i])
                # if s < self.conf_thresh:
                #     continue
                while m.ndim > 2: m = m.squeeze(0)
                results.append({
                    'mask': m,
                    'conf': s,
                    'stability_score': float(stability_scores[i]),
                    'source': 'text',
                })
            return results

    def predict_box(self, box: List[int]) -> List[Dict[str, Any]]:
        width, height = self.image_pil.size
        self.processor.reset_all_prompts(self.inference_state)
        with torch.cuda.device(self.device):
            x1, y1, x2, y2 = box
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))

            cx = (x1 + (x2 - x1) / 2) / width
            cy = (y1 + (y2 - y1) / 2) / height
            nw = (x2 - x1) / width
            nh = (y2 - y1) / height
            normalized_box = [cx, cy, nw, nh]

            output = self.processor.add_geometric_prompt(box=normalized_box, label=True, state=self.inference_state)
            masks, scores = output["masks"], output["scores"]
            results = []
            masks_np, scores_np = masks.cpu().numpy() > 0, scores.to(torch.float).cpu().numpy()
            mask_logits = output["masks_logits"].cpu().to(torch.float).numpy()
            stability_scores = self._stability_scores(mask_logits, threshold=0.5)

            for i in range(len(scores_np)):
                m, s = masks_np[i], float(scores_np[i])
                # if s < self.conf_thresh:
                #     continue
                while m.ndim > 2: m = m.squeeze(0)
                results.append({
                    'mask': m,
                    'conf': s,
                    'stability_score': float(stability_scores[i]),
                    'source': 'bbox',
                })
            return results


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute the IoU of two binary masks.
    mask_*: bool or 0/1 ndarrays with matching shapes
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def nms_masks(
    dets: list,
    thres: float = 0.5,
) -> list:
    if not dets:
        return []
    masks = []
    confs = []
    for i, d in enumerate(dets):
        m = d["mask"]
        masks.append(m)
        confs.append(float(d["conf"]))
    order = np.argsort(-np.asarray(confs))
    keep = []
    suppressed = np.zeros(len(dets), dtype=bool)
    for idx_pos, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        mi = masks[i]
        for j in order[idx_pos + 1:]:
            if suppressed[j]:
                continue
            iou = mask_iou(mi, masks[j])
            if iou > thres:
                suppressed[j] = True
    keep_sorted_by_input = sorted(keep)
    return [dets[k] for k in keep_sorted_by_input]

def filter_masks_by_bboxes(masks_info: List[Dict], bboxes: List, bbox_iou_thresh: float = 0.5):
    if len(bboxes) == 0:
        return masks_info
    filtered = []
    for mask_info in masks_info:
        if mask_info["mask"] is None or not np.any(mask_info["mask"]):
            continue
        mask_bbox = extract_bbox_from_mask(mask_info["mask"])
        for bbox in bboxes:
            if bbox_iou(mask_bbox, bbox) >= bbox_iou_thresh:
                filtered.append(mask_info)
                break
    return filtered


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    if union == 0:
        return 1.0
    return inter_area / union
