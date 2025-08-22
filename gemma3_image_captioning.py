# ===== MASTER DALLIA IMAGE CAPTION SCRIPT ===== #
# Works for Kohya training. Uses Gemma3 to generate clean, factual captions.

import os
import argparse
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

# Default model and template prompt
MODEL_ID = "unsloth/gemma-3-27b-pt"
DEFAULT_PROMPT = (
    "photo of innocent_3dcgi_base woman, dimples, medium-long wavy honey blonde hair, "
    "medium honey brown eyes, beige skin, include face angle, expression, pose, lighting, shadows, and background"
)

# Cache model/processor to avoid reloading
_model = None
_processor = None

def setup_model(model_id=MODEL_ID, hf_token=None, use_quantization=False):
    """Load (and cache) the Gemma 3 model and processor."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    token_param = {"token": hf_token} if hf_token else {}

    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if use_quantization else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    _model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        **token_param
    ).eval()

    _processor = AutoProcessor.from_pretrained(model_id, **token_param)
    return _model, _processor

def load_image_from_url(url: str):
    """Download image and convert to RGB."""
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

def caption_image(image, model, processor, prompt, max_new_tokens=256):
    """Generate a factual caption optimized for Kohya training."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    generated_tokens = outputs[0][input_len:]
    caption = processor.decode(generated_tokens, skip_special_tokens=True)
    return caption.replace("\n", " ").strip()

def generate_caption(image_url: str, prompt: str = DEFAULT_PROMPT,
                     model_id: str = MODEL_ID, hf_token=None,
                     quantize=False, max_new_tokens=256):
    """Convenience wrapper: load model once, return caption from URL."""
    model, processor = setup_model(model_id, hf_token, quantize)
    image = load_image_from_url(image_url)
    return caption_image(image, model, processor, prompt, max_new_tokens)
