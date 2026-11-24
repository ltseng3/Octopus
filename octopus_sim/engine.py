"""Tiny deterministic event scheduler, behind a deliberately narrow adapter boundary.

This stdlib engine is tested in the offline build environment. A SimPy runtime
can implement the same scheduling/resource semantics without changing policy,
world, provenance or evaluator objects.
"""
from __future__ import annotations
import heapq
from collections.abc import Callable


def to_us(ms: float) -> int:
    return int(round(ms * 1_000))


class EventLoop:
    def __init__(self):
        self.now_us = 0
        self._counter = 0
        self._queue: list[tuple[int, int, Callable[[], None], bool]] = []
        self._counted_pending = 0
        self.peak_pending_events = 0

    def at(self, at_us: int, callback: Callable[[], None], *, internal: bool = False) -> None:
        if at_us < self.now_us:
            raise ValueError('Cannot schedule in the past')
        self._counter += 1
        heapq.heappush(self._queue, (at_us, self._counter, callback, not internal))
        if not internal:
            self._counted_pending += 1
            self.peak_pending_events = max(self.peak_pending_events, self._counted_pending)

    def after(self, delay_us: int, callback: Callable[[], None]) -> None:
        if delay_us < 0:
            raise ValueError('Negative duration')
        self.at(self.now_us + delay_us, callback)

    def run(self) -> None:
        while self._queue:
            time_us, _, callback, counted = heapq.heappop(self._queue)
            if counted:
                self._counted_pending -= 1
            self.now_us = time_us
            callback()


class FIFOResource:
    """Non-preemptive FIFO server with deterministic worker allocation.

    Each request is submitted at the current simulation time. Work can queue;
    callbacks fire only upon completion. Propagation delays are not part of
    reserved link time unless explicitly scheduled as such by the caller.
    """
    def __init__(self, loop: EventLoop, workers: int):
        if workers < 1:
            raise ValueError('Resource workers must be positive')
        self.loop = loop
        self.available_at_us = [0] * workers
        self.jobs = 0
        self.busy_us = 0
        self.queue_wait_us = 0

    def request(self, duration_us: int, on_complete: Callable[[], None],
                *, on_start: Callable[[], None] | None = None) -> None:
        if duration_us < 0:
            raise ValueError('Negative service time')
        i = min(range(len(self.available_at_us)), key=lambda j: (self.available_at_us[j], j))
        start = max(self.loop.now_us, self.available_at_us[i])
        finish = start + duration_us
        self.available_at_us[i] = finish
        self.jobs += 1
        self.busy_us += duration_us
        self.queue_wait_us += start - self.loop.now_us
        if on_start is not None:
            self.loop.at(start, on_start, internal=True)
        self.loop.at(finish, on_complete)
