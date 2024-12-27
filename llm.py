import requests
import base64
import json

def analyze_classroom(photo_path):
    """
    Analyze a classroom image using NVIDIA API and return the response.

    Args:
        photo_path (str): Path to the classroom image file.

    Returns:
        dict: Parsed JSON response from the API.

    Raises:
        Exception: If the API call fails or the response is invalid.
    """
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
    api_key = "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"  # Replace with your actual API key
    
    # Read and encode the image in Base64
    try:
        with open(photo_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        raise Exception(f"Image file not found: {photo_path}")
    except Exception as e:
        raise Exception(f"Error encoding image file: {e}")
    
    # Headers for the API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    # Payload for the API request
    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    f'"Analyze the image and provide a detailed report. '
                    f'Identify activities or interactions occurring '
                    f'Are they focused or distracted? What is the overall classroom engagement level? '
                    f'Does the infrastructure of the classroom seem suitable for group activities and learning? '
                    f'Provide insights on how the students\' interactions can be improved for better engagement. '
                    f'<img src="data:image/jpeg;base64,{image_b64}" />"'
                )
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 0,
        "stream": False  # Ensure this is set to False
    }
    
    
    # Make the API request
    try:
        response = requests.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    
    # Parse and return the response
    try:
        return response.json()
    except json.JSONDecodeError:
        raise Exception("Failed to decode API response as JSON")
