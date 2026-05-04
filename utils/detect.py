from PIL import Image
from utils.logger import setup_logging, get_logger
from typing import List 
import json 
setup_logging()
log = get_logger(__name__)
try:
    from rex_omni import RexOmniWrapper, RexOmniVisualize
except ImportError:
    log.warning(f'Install rex omni (https://github.com/IDEA-Research/Rex-Omni) if you want to enable detection')

def box_xyx2xywh(box):
    x0, y0, x1, y1 = box
    x = x0
    y = y0
    w = x1 - x0
    h = y1 - y0
    return [x, y, w, h]


def convert_box(preds: dict) -> dict:
    out = {}
    for cat, detections in preds.items():
        out[cat] = []
        for det in detections:
            det_copy = dict(det)
            if det_copy.get("type") == "box":
                coords = det_copy.get("coords")
                del det_copy["coords"]
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    det_copy["xywh"] = box_xyx2xywh(coords)
            out[cat].append(det_copy)
    return out


def detect_objects(image_path: str | List[str], categories: List[str] | List[List[str]], output_path: str | List[str] = 'detect_bbox.json'):
    rex = RexOmniWrapper(
        model_path="IDEA-Research/Rex-Omni",   # HF repo or local path
        backend="transformers",                # or "vllm" for high-throughput inference
        # Inference/generation controls (applied across backends)
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )

    if isinstance(image_path, str):
        image_path = [image_path]
    if isinstance(categories, list) and all(isinstance(c, str) for c in categories):
        categories = [categories] * len(image_path)
    if isinstance(output_path, str):
        output_path = [output_path]
    for img, cats, out in zip(image_path, categories, output_path):
        image = Image.open(img).convert("RGB")
        results = rex.inference(images=image, task="detection", categories=cats)
        result = results[0]
        with open(out, 'w') as f:
            json.dump(convert_box(result["extracted_predictions"]), f, indent=4)
        # 4) Visualize
        vis = RexOmniVisualize(
            image=image,
            predictions=result["extracted_predictions"],
            font_size=20,
            draw_width=5,
            show_labels=True,
        )
        vis.save(out.replace('.json', '.jpg'))