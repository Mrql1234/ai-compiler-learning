from __future__ import annotations

import unittest

from engine import InferenceRequest, InferenceScheduler, PagedKVCache


class SchedulerTests(unittest.TestCase):
    def test_scheduler_completes_requests(self) -> None:
        scheduler = InferenceScheduler(
            kv_cache=PagedKVCache(total_pages=16, tokens_per_page=8),
            max_batch_size=2,
        )
        scheduler.submit(InferenceRequest("a", prompt_tokens=8, generate_tokens=3))
        scheduler.submit(InferenceRequest("b", prompt_tokens=8, generate_tokens=2))
        metrics = scheduler.run()
        self.assertEqual(metrics.completed_requests, 2)
        self.assertGreater(metrics.total_steps, 0)

    def test_kv_budget_can_delay_admission(self) -> None:
        scheduler = InferenceScheduler(
            kv_cache=PagedKVCache(total_pages=4, tokens_per_page=8),
            max_batch_size=2,
        )
        scheduler.submit(InferenceRequest("a", prompt_tokens=16, generate_tokens=8))
        scheduler.submit(InferenceRequest("b", prompt_tokens=16, generate_tokens=8))
        metrics = scheduler.run()
        self.assertEqual(metrics.completed_requests, 2)
        self.assertGreaterEqual(metrics.peak_kv_pages, 3)


if __name__ == "__main__":
    unittest.main()
