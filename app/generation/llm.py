import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find this in the documents."

Context:
{context}

Question: {question}"""

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    },
}


def generate_answer(question, chunks):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    cfg = PROVIDERS[provider]
    client = OpenAI(base_url=cfg["base_url"], api_key=os.getenv(cfg["key_env"]))

    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        temperature=0,
    )
    return resp.choices[0].message.content