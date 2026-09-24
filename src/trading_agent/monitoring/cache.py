"""Optional disposable research cache. Never used for risk, orders or balances."""

from time import monotonic

from redis import Redis
from redis.exceptions import RedisError


class OptionalCache:
    def __init__(self, url: str | None):
        self._client: Redis | None = (
            Redis.from_url(
                url, socket_timeout=0.2, socket_connect_timeout=0.2, decode_responses=True
            )
            if url
            else None
        )
        self._memory: dict[str, tuple[float, str]] = {}

    def available(self) -> bool:
        try:
            return bool(self._client is not None and self._client.ping())
        except (RedisError, OSError):
            return False

    def get(self, key: str) -> str | None:
        if self._client is not None:
            try:
                value = self._client.get(key)
                return str(value) if value is not None else None
            except (RedisError, OSError):
                pass
        entry = self._memory.get(key)
        if entry is None:
            return None
        if entry[0] <= monotonic():
            del self._memory[key]
            return None
        return entry[1]

    def set(self, key: str, value: str, ttl: int = 60) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if self._client is not None:
            try:
                self._client.set(key, value, ex=ttl)
                return
            except (RedisError, OSError):
                pass
        now = monotonic()
        self._memory = {k: v for k, v in self._memory.items() if v[0] > now}
        if len(self._memory) >= 1000:
            self._memory.pop(next(iter(self._memory)))
        self._memory[key] = (now + ttl, value)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
