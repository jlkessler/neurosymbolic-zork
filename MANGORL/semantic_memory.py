"""Optional frozen neural state encoder shared by Actor retrieval and critic.

No optimizer or fine-tuning is used. Vectors are cached, not persisted with
private model state. Omit this backend for the explicitly lexical ablation.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math


class FrozenStateEncoder:
    def __init__(self, model_name=None, *, model=None, cache_size=4096):
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install requirements-semantic.txt to use neural memory retrieval.") from exc
            model = SentenceTransformer(model_name, trust_remote_code=False)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.model = model
        self.cache = OrderedDict()
        self.cache_size = max(1, int(cache_size))

    def encode(self, state):
        text = json.dumps(state, sort_keys=True, ensure_ascii=False)
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        vector = tuple(float(v) for v in self.model.encode(text, normalize_embeddings=True))
        if not vector or not all(math.isfinite(v) for v in vector):
            raise ValueError("State encoder returned an invalid vector.")
        norm = math.sqrt(sum(v * v for v in vector))
        if not norm:
            raise ValueError("State encoder returned a zero vector.")
        self.cache[text] = tuple(v / norm for v in vector)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return self.cache[text]

    def similarity(self, left, right):
        a, b = self.encode(left), self.encode(right)
        if len(a) != len(b):
            raise ValueError("State embedding dimensions changed.")
        return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))
