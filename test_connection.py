from openai import OpenAI
import os

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8010/v1"
)

try:
    print("Listing models...")
    models = client.models.list()
    print("Models:", models)
    
    print("\nAttempting chat completion...")
    response = client.chat.completions.create(
        model="LGAI-EXAONE/EXAONE-4.0-32B",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print(f"Error: {e}")
