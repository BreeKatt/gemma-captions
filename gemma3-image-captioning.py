# ===== MASTER GEMMA3 IMAGE CAPTIONING SCRIPT =====
# Works for any Miss Dallia persona. Provide prompt via RunPod request.

import os
import argparse
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

# Default model and template prompt (overridden via JSON request)
MODEL_ID = "unsloth/gemma-3-27b-pt"
DEFAULT_PROMPT = (
    "photo of [CLASS_TOKEN] woman, [CORE_FEATURES], [EXPRESSION_AND_HEAD_ANGLE], "
    "[OUTFIT], [POSTURE_AND_POSE], [LIGHTING_AND_SHADOWS], [BACKGROUND]"
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Gemma 3 Image Captioning')
    parser.add_argument('--image_url', type=str, required=True,
                        help='URL of the image to caption')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT,
                        help='Prompt for image captioning')
    parser.add_argument('--model_id', type=str, default=MODEL_ID,
                        help='Gemma 3 model ID')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Hugging Face API token (optional, required for gated models)')
    parser.add_argument('--quantize', action='store_true',
                        help='Use 8-bit quantization to reduce memory usage')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum number of tokens to generate')
    return parser.parse_args()

def setup_model(model_id, hf_token, use_quantization):
    """Load the Gemma 3 model and processor."""
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        token_param = {"token": hf_token}
    else:
        token_param = {}

    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if use_quantization else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        **token_param
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, **token_param)
    return model, processor

def load_image_from_url(url):
    """Download image from URL and convert to RGB."""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

def caption_image(image, model, processor, prompt, max_new_tokens):
    """Generate caption for a single image."""
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
    return caption.replace('\n', ' ').strip()

def main():
    args = parse_arguments()

    print(f"Loading model: {args.model_id}...")
    model, processor = setup_model(args.model_id, args.hf_token, args.quantize)
    print("Model loaded.")

    print(f"Downloading image from URL: {args.image_url}")
    image = load_image_from_url(args.image_url)

    print(f"Generating caption using prompt: {args.prompt}")
    caption = caption_image(image, model, processor, args.prompt, args.max_new_tokens)
    
    print("\n===== GENERATED CAPTION =====")
    print(caption)
    print("=============================")

if __name__ == "__main__":
    main()
