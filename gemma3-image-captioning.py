# ===== TEST GEMMA IMAGE CAPTION SCRIPT =====
# Temporary small CPU model for endpoint testing

import os
import argparse
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration

# Tiny CPU-only model for testing
MODEL_ID = "Salesforce/blip-image-captioning-base"
DEFAULT_PROMPT = (
    "photo of [CLASS_TOKEN] woman, [CORE_FEATURES], [EXPRESSION_AND_HEAD_ANGLE], "
    "[OUTFIT], [POSTURE_AND_POSE], [LIGHTING_AND_SHADOWS], [BACKGROUND]"
)

_model = None
_processor = None

def setup_model(model_id=MODEL_ID, hf_token=None, use_quantization=False):
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    _model = BlipForConditionalGeneration.from_pretrained(model_id).eval()
    _processor = AutoProcessor.from_pretrained(model_id)
    return _model, _processor

def load_image_from_url(url: str):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def caption_image(image, model, processor, prompt, max_new_tokens=256):
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption.replace("\n", " ").strip()

def generate_caption(image_url: str, prompt: str = DEFAULT_PROMPT,
                     model_id: str = MODEL_ID, hf_token=None,
                     quantize=False, max_new_tokens=256):
    model, processor = setup_model(model_id, hf_token, quantize)
    image = load_image_from_url(image_url)
    return caption_image(image, model, processor, prompt, max_new_tokens)

def main():
    parser = argparse.ArgumentParser(description="Gemma Image Captioning (Test)")
    parser.add_argument("--image_url", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    caption = generate_caption(
        image_url=args.image_url,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens
    )
    print(caption, flush=True)

if __name__ == "__main__":
    main()
