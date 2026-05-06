from __future__ import annotations


class PagedKVCache:
    def __init__(self, total_pages: int, tokens_per_page: int) -> None:
        self.total_pages = total_pages
        self.tokens_per_page = tokens_per_page
        self.used_pages = 0

    def pages_for_tokens(self, token_count: int) -> int:
        if token_count <= 0:
            return 0
        return (token_count + self.tokens_per_page - 1) // self.tokens_per_page

    def can_allocate(self, pages: int) -> bool:
        return self.used_pages + pages <= self.total_pages

    def allocate(self, pages: int) -> bool:
        if not self.can_allocate(pages):
            return False
        self.used_pages += pages
        return True

    def free(self, pages: int) -> None:
        self.used_pages = max(0, self.used_pages - pages)

    def free_pages(self) -> int:
        return self.total_pages - self.used_pages
