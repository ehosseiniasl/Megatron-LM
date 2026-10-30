import logging
from dataclasses import dataclass

from mistral_fork.shared.args import Args

logger = logging.getLogger(__name__)


# -- Training feature args --


@dataclass
class TwoPhaseArgs(Args):
    data_phase_two: str
    mix_start_tok: int
    mix_end_tok: int
    image_data_phase_two: str = ""
    ocr_data_phase_two: str = ""

    def __post_init__(self) -> None:
        assert self.mix_start_tok <= self.mix_end_tok
