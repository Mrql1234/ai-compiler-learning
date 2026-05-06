from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InferenceRequest:
    request_id: str
    prompt_tokens: int
    generate_tokens: int
    prompt_processed: bool = False
    generated_tokens: int = 0
    ttft_step: int | None = None
    finish_step: int | None = None
    kv_pages: int = 0
    emitted_first_token: bool = False
    history: list[str] = field(default_factory=list)

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generate_tokens

    def mark_prompt_processed(self, step: int) -> None:
        self.prompt_processed = True
        self.history.append(f"step {step}: prompt processed")

    def emit_token(self, step: int) -> None:
        self.generated_tokens += 1
        if not self.emitted_first_token:
            self.emitted_first_token = True
            self.ttft_step = step
            self.history.append(f"step {step}: first token")

    def is_finished(self) -> bool:
        return self.generated_tokens >= self.generate_tokens
