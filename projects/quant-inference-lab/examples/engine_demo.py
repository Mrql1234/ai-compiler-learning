from __future__ import annotations

from engine import InferenceRequest, InferenceScheduler, PagedKVCache


def main() -> None:
    scheduler = InferenceScheduler(
        kv_cache=PagedKVCache(total_pages=24, tokens_per_page=8),
        max_batch_size=2,
    )
    scheduler.submit(InferenceRequest("req-a", prompt_tokens=24, generate_tokens=6))
    scheduler.submit(InferenceRequest("req-b", prompt_tokens=12, generate_tokens=4))
    scheduler.submit(InferenceRequest("req-c", prompt_tokens=40, generate_tokens=5))

    metrics = scheduler.run()

    print("=== Engine Demo ===")
    print(f"total steps: {metrics.total_steps}")
    print(f"completed requests: {metrics.completed_requests}")
    print(f"average ttft steps: {metrics.average_ttft_steps:.2f}")
    print(f"peak kv pages: {metrics.peak_kv_pages}")
    print()
    for request in scheduler.completed:
        print(
            f"{request.request_id}: ttft={request.ttft_step}, "
            f"finish={request.finish_step}, kv_pages={request.kv_pages}"
        )


if __name__ == "__main__":
    main()
