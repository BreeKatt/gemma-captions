# ===== MASTER GEMMA3 IMAGE CAPTION SCRIPT =====
# Works for any Miss Dallia persona. Can be run standalone or imported by handler.

import os
import argparse
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

MODEL_ID = "unsloth/gemma-3-27b-pt"
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

    token_param = {"token": hf_token} if hf_token else None
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if use_quantization else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    _model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        **(token_param or {})
    ).eval()
    _processor = AutoProcessor.from_pretrained(model_id, **(token_param or {}))
    return _model, _processor

def load_image_from_url(url: str):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def caption_image(image, model, processor, prompt, max_new_tokens=256):
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image}
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_tokens = outputs[0][input_len:]
    caption = processor.decode(generated_tokens, skip_special_tokens=True)
    return caption.replace("\n", " ").strip()

def generate_caption(image_url: str, prompt: str = DEFAULT_PROMPT,
                     model_id: str = MODEL_ID, hf_token=None,
                     quantize=False, max_new_tokens=256):
    model, processor = setup_model(model_id, hf_token, quantize)
    image = load_image_from_url(image_url)
    return caption_image(image, model, processor, prompt, max_new_tokens)

def main():
    parser = argparse.ArgumentParser(description="Gemma 3 Image Captioning")
    parser.add_argument("--image_url", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    caption = generate_caption(
        image_url=args.image_url,
        prompt=args.prompt,
        model_id=args.model_id,
        hf_token=args.hf_token,
        quantize=args.quantize,
        max_new_tokens=args.max_new_tokens
    )
    print(caption, flush=True)

if __name__ == "__main__":
    main()
