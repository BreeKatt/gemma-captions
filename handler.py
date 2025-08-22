import runpod
from gemma3_image_captioning import generate_caption  # import your function

def handler(job):
    """
    Expects JSON like:
    {
      "input": {
        "image_url": "https://.../my.jpg",
        "prompt": "optional caption style"
      }
    }
    """
    job_input = job.get("input", {}) or {}
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt", "")

    if not image_url:
        return {"error": "image_url is required"}

    try:
        caption = generate_caption(image_url, prompt)  # call function directly
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
