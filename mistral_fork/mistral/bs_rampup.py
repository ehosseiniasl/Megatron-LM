from dataclasses import dataclass

import numpy as np


@dataclass
class RampupLevel:
    global_batch_size: int
    num_microbatches: int
    local_batch_size: int
    steps: int
    tokens_processed: int
    max_lr_rescale: float

    @property
    def microbatch_size(self) -> int:
        return self.global_batch_size // self.num_microbatches


class BatchSizeSchedule:
    def __init__(
        self,
        rampup_levels: list[RampupLevel],
    ) -> None:
        self.rampup_levels: list[RampupLevel] = rampup_levels
        self.top_bs = self.rampup_levels[-1].global_batch_size

    def get_level_by_tokens(self, seen_tokens: int) -> RampupLevel:
        for level in self.rampup_levels:
            if seen_tokens < level.tokens_processed:
                return level
        return self.rampup_levels[-1]
