from __future__ import annotations

from dataclasses import dataclass

from .kv_cache import PagedKVCache
from .request import InferenceRequest


@dataclass
class EngineMetrics:
    total_steps: int
    completed_requests: int
    average_ttft_steps: float
    peak_kv_pages: int


class InferenceScheduler:
    def __init__(
        self,
        *,
        kv_cache: PagedKVCache,
        max_batch_size: int,
    ) -> None:
        self.kv_cache = kv_cache
        self.max_batch_size = max_batch_size
        self.waiting: list[InferenceRequest] = []
        self.active: list[InferenceRequest] = []
        self.completed: list[InferenceRequest] = []
        self.current_step = 0
        self.peak_kv_pages = 0

    def submit(self, request: InferenceRequest) -> None:
        self.waiting.append(request)

    def _admit_requests(self) -> None:
        while self.waiting and len(self.active) < self.max_batch_size:
            candidate = self.waiting[0]
            needed_pages = self.kv_cache.pages_for_tokens(candidate.total_tokens())
            if not self.kv_cache.allocate(needed_pages):
                break
            candidate.kv_pages = needed_pages
            self.waiting.pop(0)
            self.active.append(candidate)
            self.peak_kv_pages = max(self.peak_kv_pages, self.kv_cache.used_pages)

    def step(self) -> None:
        self.current_step += 1
        self._admit_requests()

        finished: list[InferenceRequest] = []
        for request in self.active:
            if not request.prompt_processed:
                request.mark_prompt_processed(self.current_step)
                continue

            if not request.is_finished():
                request.emit_token(self.current_step)

            if request.is_finished():
                request.finish_step = self.current_step
                request.history.append(f"step {self.current_step}: finished")
                finished.append(request)

        for request in finished:
            self.active.remove(request)
            self.completed.append(request)
            self.kv_cache.free(request.kv_pages)

    def run(self) -> EngineMetrics:
        while self.waiting or self.active:
            self.step()

        ttft_values = [request.ttft_step for request in self.completed if request.ttft_step is not None]
        average_ttft = float(sum(ttft_values) / len(ttft_values)) if ttft_values else 0.0
        return EngineMetrics(
            total_steps=self.current_step,
            completed_requests=len(self.completed),
            average_ttft_steps=average_ttft,
            peak_kv_pages=self.peak_kv_pages,
        )
