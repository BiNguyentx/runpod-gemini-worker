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
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }

    r = requests.post(
        GEMINI_URL + f"?key={GEMINI_API_KEY}",
        json=payload
    )

    try:
        data = r.json()
    except:
        return {"error": "Response not JSON", "raw": r.text}

    # Gemini returns images inside candidates -> content -> parts
    try:
        b64img = data["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
        return {
            "prompt": prompt,
            "image_base64": b64img
        }
    except Exception as e:
        return {"error": "Failed to parse image result", "debug": data}


def handler(event):
    prompt = event.get("input", {}).get("prompt")
    if not prompt:
        return {"error": "Missing prompt"}

    return generate_image(prompt)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
