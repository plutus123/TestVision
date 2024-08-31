import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_test_cases_with_llm(context, images):
    messages = [
        {
            "role": "system",
            "content": "You are an expert QA tester. Generate detailed test cases for the Red Bus mobile app features shown in the screenshots. Each test case should include a description, pre-conditions, testing steps, and expected results."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Generate test cases for the following Red Bus app features. Additional context: {context}"
                }
            ]
        }
    ]

    for image in images:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}"
            }
        })

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4-vision-preview",
            "messages": messages,
            "max_tokens": 1000
        }
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception("Failed to generate test cases")