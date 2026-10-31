"""
Shows that an unmodified OpenAI client works through the gateway - the only
change is base_url. Start the gateway first:
    uvicorn app.main:app --port 8000 --env-file .env
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="gateway-key-or-anything")

resp = client.chat.completions.create(
    model="any",  # overridden by UPSTREAM_MODEL if set
    messages=[{
        "role": "user",
        "content": "Write a short, polite payment reminder to Rahul Sharma "
                   "(rahul.s@gmail.com, 9876543210) about his overdue loan.",
    }],
)
print(resp.choices[0].message.content)
