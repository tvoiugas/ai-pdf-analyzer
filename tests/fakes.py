"""Stand-ins for Ollama: no network, deterministic 1024-dim vectors, recorded prompts."""
import hashlib
import random
from types import SimpleNamespace

EMBEDDING_DIMENSIONS = 1024


def fake_vector(text):
    """A fixed 1024-dim vector per text, so stored embeddings have the real column size."""
    rng = random.Random(hashlib.sha256(text.encode()).digest())
    return [rng.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSIONS)]


class FakeEmbedder:
    def __init__(self):
        self.documents = []
        self.queries = []

    def embed_documents(self, texts):
        self.documents.append(list(texts))
        return [fake_vector(text) for text in texts]

    async def aembed_query(self, text):
        self.queries.append(text)
        return fake_vector(text)


class FakeLLM:
    def __init__(self, reply="fake answer"):
        self.reply = reply
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.reply)

    @property
    def last_user_message(self):
        return self.calls[-1][1].content
