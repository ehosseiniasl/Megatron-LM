from mistral_fork.mistral_common.base import MistralBase


class Span(MistralBase):
    start: int
    end: int

    def __add__(self, other: int) -> "Span":
        return Span(start=self.start + other, end=self.end + other)

    def __iadd__(self, other: int) -> "Span":
        self.start += other
        self.end += other
        return self
