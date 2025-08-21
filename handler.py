import subprocess
import runpod

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

    # Build command to run your captioning script
    command = ["python3", "gemma3-image-captioning.py", "--image_url", image_url]
    if prompt:
        command += ["--prompt", prompt]

    # Run the script
    result = subprocess.run(command, capture_output=True, text=True)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip()
    }

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
