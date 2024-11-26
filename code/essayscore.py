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
    
file_name = "../essayscore.json"  # Ensure this file is in the same directory as your script
context_data = load_context_data(file_name)

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
                f"Please generate the use purpose for this dataset in 50 words:\n\n{context_data}"
            )
        }
    ],
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_json = response.json()
generated_purpose = response_json['choices'][0]['message']['content']

output_dir = "generatedText"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "essayscore.txt")
with open("generatedText/essayscore.txt", "w") as file:
    file.write(generated_purpose)

print("Output saved to essayscore.txt")

references = [
    ["to", "develop", "a", "model", "that", "accurately", "scores", "essays", "based", "on", "varying", "grading", "criteria", 
     "by", "leveraging", "pre-training", "fine-tuning", "pseudo-labelling", "and", "ensembling", "techniques"]
]

generated_tokens = generated_purpose.split()

smoothie = SmoothingFunction().method4

# Adjust weights to focus on unigrams and bigrams if text is short
bleu_score = sentence_bleu(references, generated_tokens, weights=(0.5, 0.5), smoothing_function=smoothie)

print("BLEU Score:", bleu_score)
