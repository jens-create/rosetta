import typing as typ
from pathlib import Path

from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_random_exponential


class BaseClient:
    """Generic client for Vllm, Azure, and OpenAI.

    It should take in a request dict and a config and return a response dict.
    """

    fn: typ.Callable

    def __init__(
        self,
        fn: typ.Callable,
        cache_dir: str | Path,
        cache_reset: bool = False,
    ) -> None:
        """Initialize client."""
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir, "memory").resolve()
        memory = Memory(cache_dir, verbose=0)
        if cache_reset:
            memory.clear(warn=False)
        self.fn = memory.cache(fn)

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=0, max=15))
    def __call__(self, *args: typ.Any, **kwargs: dict[str, typ.Any]) -> str:
        """Generate a response. This should be implemented in the child class."""
        return self.fn(*args, **kwargs)
