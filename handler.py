import os
import requests
import runpod

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Imagen models available
IMAGEN_MODELS = {
    "imagen-4-ultra": "imagen-4.0-generate-preview-06-06",
    "imagen-4": "imagen-4.0-generate-preview-06-06", 
    "imagen-3": "imagen-3.0-generate-002"
}


def generate_image(prompt, model="imagen-3", sample_count=1, aspect_ratio="1:1", 
                   negative_prompt="", person_generation="allow_adult"):
    """
    Generate image using Gemini Imagen API
    
    Args:
        prompt: Text prompt for image generation
        model: Model version (imagen-3, imagen-4, imagen-4-ultra)
        sample_count: Number of images (1-4 for imagen-3, 1 for imagen-4-ultra)
        aspect_ratio: "1:1", "3:4", "4:3", "9:16", "16:9"
        negative_prompt: What to avoid in the image
        person_generation: "dont_allow", "allow_adult", "allow_all"
    """
    
    if not GEMINI_API_KEY:
        return {"error": "Missing GEMINI_API_KEY environment variable"}
    
    # Get model name
    model_name = IMAGEN_MODELS.get(model, IMAGEN_MODELS["imagen-3"])
    
    # Build API URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:predict"
    
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Build payload according to Gemini API docs
    payload = {
        "instances": [
            {
                "prompt": prompt
            }
        ],
        "parameters": {
            "sampleCount": sample_count,
            "aspectRatio": aspect_ratio,
            "personGeneration": person_generation
        }
    }
    
    # Add negative prompt if provided
    if negative_prompt:
        payload["parameters"]["negativePrompt"] = negative_prompt
    
    try:
        print(f"Calling Imagen API...")
        print(f"Model: {model_name}")
        print(f"Prompt: {prompt}")
        print(f"Samples: {sample_count}, Aspect Ratio: {aspect_ratio}")
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=90
        )
        
        # Check response status
        if response.status_code != 200:
            error_text = response.text
            return {
                "error": f"API returned status {response.status_code}",
                "details": error_text
            }
        
        data = response.json()
        
        # Extract images from predictions
        predictions = data.get("predictions", [])
        
        if not predictions:
            return {
                "error": "No images generated",
                "response": data
            }
        
        # Collect all generated images
        images = []
        for prediction in predictions:
            # Gemini returns base64 in bytesBase64Encoded or image field
            if "bytesBase64Encoded" in prediction:
                images.append(prediction["bytesBase64Encoded"])
            elif "image" in prediction:
                # Sometimes it's nested in 'image' object
                img_data = prediction["image"]
                if isinstance(img_data, dict) and "bytesBase64Encoded" in img_data:
                    images.append(img_data["bytesBase64Encoded"])
                elif isinstance(img_data, str):
                    images.append(img_data)
        
        if not images:
            return {
                "error": "Could not extract images from response",
                "response": data
            }
        
        return {
            "success": True,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": model_name,
            "aspect_ratio": aspect_ratio,
            "images": images,
            "count": len(images)
        }
        
    except requests.exceptions.Timeout:
        return {"error": "Request timeout after 90 seconds"}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def handler(event):
    """
    RunPod handler function
    
    Expected input format:
    {
        "input": {
            "prompt": "A cute cat wearing sunglasses",
            "model": "imagen-3",  # optional: imagen-3, imagen-4, imagen-4-ultra
            "sample_count": 2,  # optional: 1-4 (1 for ultra)
            "aspect_ratio": "1:1",  # optional: 1:1, 3:4, 4:3, 9:16, 16:9
            "negative_prompt": "blurry, low quality",  # optional
            "person_generation": "allow_adult"  # optional
        }
    }
    """
    
    try:
        input_data = event.get("input", {})
        
        # Required parameter
        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing 'prompt' in input"}
        
        # Optional parameters with defaults
        model = input_data.get("model", "imagen-3")
        sample_count = input_data.get("sample_count", 1)
        aspect_ratio = input_data.get("aspect_ratio", "1:1")
        negative_prompt = input_data.get("negative_prompt", "")
        person_generation = input_data.get("person_generation", "allow_adult")
        
        # Validate sample_count
        if model == "imagen-4-ultra":
            sample_count = 1  # Ultra only supports 1 image
        else:
            sample_count = max(1, min(4, sample_count))
        
        # Validate aspect_ratio
        valid_ratios = ["1:1", "3:4", "4:3", "9:16", "16:9"]
        if aspect_ratio not in valid_ratios:
            aspect_ratio = "1:1"
        
        # Validate person_generation
        valid_person_gen = ["dont_allow", "allow_adult", "allow_all"]
        if person_generation not in valid_person_gen:
            person_generation = "allow_adult"
        
        # Generate images
        result = generate_image(
            prompt=prompt,
            model=model,
            sample_count=sample_count,
            aspect_ratio=aspect_ratio,
            negative_prompt=negative_prompt,
            person_generation=person_generation
        )
        
        # Log result
        if result.get("success"):
            print(f"✓ Successfully generated {result['count']} image(s)")
        else:
            print(f"✗ Error: {result.get('error')}")
        
        return result
        
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}


if __name__ == "__main__":
    print("=" * 70)
    print("RunPod Serverless Worker - Google Gemini Imagen API")
    print("=" * 70)
    print(f"GEMINI_API_KEY: {'✓ Configured' if GEMINI_API_KEY else '✗ Not set'}")
    print(f"Available models: {', '.join(IMAGEN_MODELS.keys())}")
    print("=" * 70)
    
    runpod.serverless.start({"handler": handler})
