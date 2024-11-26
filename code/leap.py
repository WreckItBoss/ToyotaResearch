import base64
import requests
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_API_KEY"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_context_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)
    
file_name = "../leap.json"  # Ensure this file is in the same directory as your script
context_data = load_context_data(file_name)

context_json = json.dumps(context_data, indent=4)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

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

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_json = response.json()
generated_purpose = response_json['choices'][0]['message']['content']

output_dir = "generatedText"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "leap.txt")
with open("generatedText/leap.txt", "w") as file:
    file.write(generated_purpose)

print("Output saved to leap.txt")

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

smoothie = SmoothingFunction().method4

# Adjust weights to focus on unigrams and bigrams if text is short
bleu_score = sentence_bleu(reference, generated_tokens, weights=(0.5, 0.5), smoothing_function=smoothie)

print("BLEU Score:", bleu_score)

