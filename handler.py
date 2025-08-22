import runpod

print(">>> Starting handler.py...")

try:
    print(">>> Importing gemma3_image_captioning...")
    from gemma3_image_captioning import generate_caption
    print(">>> Successfully imported gemma3_image_captioning.")
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print(">>> FAILED to import gemma3_image_captioning.")
    print(tb)
    raise e

def handler(job):
    print(">>> Handler called with job:", job)
    job_input = job.get("input", {}) or {}
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt", "")

    if not image_url:
        return {"error": "image_url is required"}

    try:
        caption = generate_caption(image_url, prompt)
        print(">>> Caption generated:", caption)
        return {"caption": caption}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(">>> Error during caption generation.")
        print(tb)
        return {"error": str(e), "traceback": tb}

print(">>> Finished defining handler.")

runpod.serverless.start({"handler": handler})
print(">>> runpod.serverless.start called.")
