import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the API key from the .env file
api_key = os.getenv("OPENAI_API_KEY")

# Define the headers, including the authorization with the API key
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Define the payload with the model and message you want to send
payload = {
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ]
}

# Send the POST request to the OpenAI API
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Parse the JSON response
chat_completion = response.json()

# Extract the content you want to save
output_content = chat_completion['choices'][0]['message']['content']

# Save the output to a separate file
with open("output.txt", "w") as file:
    file.write(output_content)

# Optional: print a confirmation message
print("Output saved to output.txt")
