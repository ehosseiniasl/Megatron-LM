from enum import StrEnum
from pathlib import Path


class SampleType(StrEnum):
    PRETRAIN = "pretrain"
    INSTRUCT = "instruct"
    PRETRAIN_IMAGE = "pretrain_image"
    PRETRAIN_OCR = "pretrain_ocr"

    def is_multimodal(self) -> bool:
        return self in (SampleType.PRETRAIN_IMAGE, SampleType.PRETRAIN_OCR)


def get_src_name(path: Path) -> str:
    if str(path).endswith(".jsonl"):
        prefix = "_".join([path.parents[i].name for i in range(2, -1, -1)])
        return f"{prefix}_{path.name}"
    else:
        # not cluster independant but since nesting is different for each dataset at least it is simple...
        return "__".join(path.parts[1:])  # 0 is "/"
