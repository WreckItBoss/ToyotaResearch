import base64
import requests
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file (if you're storing the API key in a .env file)
load_dotenv()

# OpenAI API Key from .env or hardcoded
api_key = os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_API_KEY"

# Function to encode the image (if needed)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_context_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)
    
file_name = "../leap.json"  # Ensure this file is in the same directory as your script
context_data = load_context_data(file_name)

# Convert to JSON format for clarity in the API request
context_json = json.dumps(context_data, indent=4)

# Headers for the API request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Update payload to include the context for ChatGPT
payload = {
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": (
                f"Here is some context information in JSON format. "
                f"Please generate the use purpose for this dataset:\n\n{context_json}"
            )
        }
    ],
}

# Send the POST request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_json = response.json()
generated_purpose = response_json['choices'][0]['message']['content']

# Save the output to a separate file
output_dir = "generatedText"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "leap.txt")
with open("generatedText/leap.txt", "w") as file:
    file.write(generated_purpose)

print("Output saved to leap.txt")

# Reference metadata for BLEU score calculation
reference = [
    [
        "to", "develop", "machine", "learning", "models", "that", "emulate", 
        "subgrid", "atmospheric", "processes", "within", "the", "e3sm-mmf", 
        "climate", "model", "thus", "improving", "the", "accuracy", "of", 
        "long-term", "climate", "predictions", "and", "supporting", 
        "policymaking", "to", "mitigate", "climate", "change", "impacts"
    ]
]

generated_tokens = generated_purpose.split()

# Calculate BLEU score with smoothing to avoid zero scores for short texts
# Using smoothing function for better handling of shorter texts
smoothie = SmoothingFunction().method4

# Adjust weights to focus on unigrams and bigrams if text is short
bleu_score = sentence_bleu(reference, generated_tokens, weights=(0.5, 0.5), smoothing_function=smoothie)

# Print the BLEU score
print("BLEU Score:", bleu_score)

