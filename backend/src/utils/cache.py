"""
LRU Cache with TTL for LexVed retrieval and reranking results.
No external dependencies (no Redis required).
"""
import hashlib
import time
from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size=500, ttl_seconds=600):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str):
        h = self._hash_key(key)
        if h in self._cache:
            entry = self._cache[h]
            if time.time() - entry["ts"] < self._ttl:
                self._cache.move_to_end(h)
                return entry["value"]
            else:
                del self._cache[h]
        return None

    def put(self, key: str, value):
        h = self._hash_key(key)
        if h in self._cache:
            del self._cache[h]
        self._cache[h] = {"value": value, "ts": time.time()}
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self):
        self._cache.clear()

    def size(self):
        return len(self._cache)


# Global cache instances
retrieval_cache = LRUCache(max_size=500, ttl_seconds=600)
crossencoder_cache = LRUCache(max_size=1000, ttl_seconds=600)
