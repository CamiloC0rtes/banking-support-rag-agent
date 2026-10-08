import os

# Unit tests never call OpenAI; the client only needs a key to be constructed.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
