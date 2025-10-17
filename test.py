from openai import OpenAI
import os

# If you already set the env var, this can be OpenAI() without api_key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":"Say hi in one word."}]
)

print(resp.choices[0].message.content)
