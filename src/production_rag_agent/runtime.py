import asyncio
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from time import monotonic
from typing import Callable, TypeVar

from anyio import to_thread

T = TypeVar("T")


class BusyError(Exception):
    pass


class Runtime:
    def __init__(self, capacity: int, requests_per_minute: int) -> None:
        self.executor = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="rag")
        self.slots = threading.BoundedSemaphore(capacity)
        self.lock = threading.Lock()
        self.capacity = capacity
        self.rate = requests_per_minute
        self.tokens = float(requests_per_minute)
        self.updated = monotonic()
        self.active = 0
        self.counters: Counter[str] = Counter()

    def count(self, event: str) -> None:
        with self.lock:
            self.counters[event] += 1

    def allow(self) -> bool:
        with self.lock:
            now = monotonic()
            self.tokens = min(float(self.rate), self.tokens + (now - self.updated) * self.rate / 60)
            self.updated = now
            if self.tokens < 1:
                self.counters["rate_rejected"] += 1
                return False
            self.tokens -= 1
            return True

    def metrics(self) -> dict[str, object]:
        with self.lock:
            return {
                "active_workers": self.active,
                "capacity": self.capacity,
                "counters": dict(self.counters),
            }

    async def execute(self, function: Callable[[], T], deadline: float) -> T:
        if not self.slots.acquire(blocking=False):
            self.count("capacity_rejected")
            raise BusyError
        with self.lock:
            self.active += 1
        try:
            future = self.executor.submit(function)
        except BaseException:
            with self.lock:
                self.active -= 1
            self.slots.release()
            raise

        def completed(_future: object) -> None:
            # A disconnected or timed-out HTTP request does not stop synchronous provider work.
            with self.lock:
                self.active -= 1
                self.counters["work_completed"] += 1
            self.slots.release()

        future.add_done_callback(completed)
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.wait_for(asyncio.shield(wrapped), timeout=deadline)

    async def close(self) -> None:
        await to_thread.run_sync(
            lambda: self.executor.shutdown(wait=True, cancel_futures=True)
        )
