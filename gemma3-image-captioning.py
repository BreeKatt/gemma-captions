# Trigger rebuild - minor comment #002
def generate_caption(image_url: str, prompt: str = DEFAULT_PROMPT,
                     model_id: str = MODEL_ID, hf_token=None,
                     quantize=False, max_new_tokens=256):
    """Convenience wrapper to load model once and caption directly from URL."""
    model, processor = setup_model(model_id, hf_token, quantize)
    image = load_image_from_url(image_url)
    return caption_image(image, model, processor, prompt, max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description="Gemma 3 Image Captioning")
    parser.add_argument("--image_url", type=str, required=True,
                        help="URL of the image to caption")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Prompt for image captioning")
    parser.add_argument("--model_id", type=str, default=MODEL_ID,
                        help="Gemma 3 model ID")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face API token (optional)")
    parser.add_argument("--quantize", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}...")
    caption = generate_caption(
        image_url=args.image_url,
        prompt=args.prompt,
        model_id=args.model_id,
        hf_token=args.hf_token,
        quantize=args.quantize,
        max_new_tokens=args.max_new_tokens
    )

    print("\n===== GENERATED CAPTION =====")
    print(caption)
    print("=============================")


if __name__ == "__main__":
    main()
