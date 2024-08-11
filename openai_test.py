import base64
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if you're storing the API key in a .env file)
load_dotenv()

# OpenAI API Key from .env or hardcoded
api_key = os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_API_KEY"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "testone.PNG"

# Getting the base64 string
base64_image = encode_image(image_path)

# Headers for the API request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Payload for the API request
payload = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Tell me about this image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
}

# Send the POST request to the OpenAI API
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Parse the JSON response
response_json = response.json()

# Extract the content you want to save
content = response_json['choices'][0]['message']['content']

# Save the output to a separate file
with open("image_analysis_output.txt", "w") as file:
    file.write(content) 

# Optional: print a confirmation message
print("Output saved to image_analysis_output.txt")
