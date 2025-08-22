import traceback
import runpod
from gemma3_image_captioning import generate_caption

def handler(job):
    """
    Expects JSON like:
    {
      "input": {
        "image_url": "https://i.postimg.cc/WpQLpkvD/TEST.png",
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
        caption = generate_caption(image_url, prompt)
        return {"caption": caption}
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

runpod.serverless.start({"handler": handler})
