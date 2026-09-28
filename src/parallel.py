from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar


T = TypeVar("T")


def run_parallel_calls(calls: Sequence[Callable[[], T]]) -> list[T]:
    """Run independent calls concurrently and return results in input order."""
    if len(calls) < 2:
        return [call() for call in calls]
    with ThreadPoolExecutor(max_workers=min(2, len(calls))) as executor:
        return list(executor.map(lambda call: call(), calls))
