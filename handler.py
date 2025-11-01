import os
import base64
import requests
import runpod

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generate"

def generate_image(prompt):
    if not GEMINI_API_KEY:
        return {"error": "Missing GEMINI_API_KEY env variable"}

    payload = {
        "prompt": prompt,
        "mimeType": "image/png"
    }

    r = requests.post(
        GEMINI_URL + f"?key={GEMINI_API_KEY}",
        json=payload
    )

    try:
        data = r.json()
    except:
        return {"error": "Non-JSON response", "raw": r.text}

    if "generatedImages" not in data:
        return {"error": data}

    img_b64 = data["generatedImages"][0]["bytesBase64Encoded"]

    return {
        "prompt": prompt,
        "image_base64": img_b64
    }


# ENTRY POINT for RunPod
def handler(event):
    input_data = event.get("input", {})

    prompt = input_data.get("prompt")
    if not prompt:
        return {"error": "Missing 'prompt' field"}

    result = generate_image(prompt)
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
