import enum
import typing

from pydantic import Field, field_validator, model_validator

from mistral_fork.mistral.data.pretokenized.cache_key import TokenizerName
from mistral_fork.mistral.data.pretokenized.document_filters import (
    BadDocumentFilter,
    PdsFilter,
    RareTokenFilter,
)
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.source import SampleType
from mistral_fork.mistral_common.base import MistralBase


class ImageAnnotationType(enum.StrEnum):
    MOCK = "mock"
    IMG_BBOX_FOR_TRIM = "img_bbox_for_trim"
    L2_RECONSTRUCTION_SCORE = "l2_reconstruction_score"


class ImageTransformType(enum.StrEnum):
    NO_TRANSFORM = "no_transform"
    CRAWL_DEFAULT_NO_HIGH_RES = "crawl_default_no_high_res"
    GROUNDING_DEFAULT = "grounding_default"
    SOMETIMES_HIGH_RES_DEFAULT = "sometimes_high_res_default"


IMAGE_TRANSFORM_TYPE_TO_REQUIRED_ANNOTATIONS: dict[
    ImageTransformType, set[ImageAnnotationType]
] = {
    ImageTransformType.NO_TRANSFORM: set(),
    ImageTransformType.SOMETIMES_HIGH_RES_DEFAULT: set(),
    ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES: {
        ImageAnnotationType.L2_RECONSTRUCTION_SCORE
    },
    ImageTransformType.GROUNDING_DEFAULT: {
        ImageAnnotationType.L2_RECONSTRUCTION_SCORE
    },
}


def required_image_annotations_for_transform(
    image_transform_type: ImageTransformType | None,
) -> set[ImageAnnotationType]:
    if image_transform_type is None:
        return set()
    try:
        return set(IMAGE_TRANSFORM_TYPE_TO_REQUIRED_ANNOTATIONS[image_transform_type])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported image_transform_type: {image_transform_type}"
        ) from exc


class MaskingStrategy(enum.StrEnum):
    "We use an enum to describe loss_mask function to be able to serialize them"

    MOCK = "MOCK"
    MASK_FILE_SEPARATOR = "MASK_FILE_SEPARATOR"
    MASK_FILE_SEPARATOR_NO_LAZINESS = "MASK_FILE_SEPARATOR_NO_LAZINESS"
    MASK_REASONING_TRACES_QUESTIONS = "MASK_REASONING_TRACES_QUESTIONS"
    MASK_MIDTRAINING_SEGMENTS = "MASK_MIDTRAINING_SEGMENTS"
    MASK_GITHUB_PR_FORMATTING = "MASK_GITHUB_PR_FORMATTING"
    MASK_GITHUB_PR_EXPANDED_CONTEXT = "MASK_GITHUB_PR_EXPANDED_CONTEXT"
    MASK_GITHUB_PR_EXPANDED_CONTEXT_AND_CONTEXT = (
        "MASK_GITHUB_PR_EXPANDED_CONTEXT_AND_CONTEXT"
    )
    MASK_LOCODIFF_MARKERS = "MASK_LOCODIFF_MARKERS"
    MASK_MMDOC_CONTENT = "MASK_MMDOC_CONTENT"
    INSTRUCT = "INSTRUCT"


class PretokenizedDatasourceMetadataV2(MistralBase):
    """V2 metadata with all fields required — no backward-compat optionals.

    Written by the migration script alongside existing pretokenization_metadata.json.
    """

    filter_name: FilterArgsName
    tokenizer_name: TokenizerName
    filter_args_json: str
    tokenizer_path: str
    original_source: str
    n_chunks: int
    validation_fraction: float
    guarantee_validation_per_chunk: bool
    has_image_metadata: bool
    sample_type: SampleType
    masking_strategy: MaskingStrategy | None = None
    rare_token_filter: RareTokenFilter | None = None
    pds_filter: PdsFilter | None = None
    bad_document_filter: BadDocumentFilter | None = None
    ngram_blocklist_path: str | None = None
    ngram_blocklist_hash: str | None = None
    available_image_annotations: set[ImageAnnotationType] = Field(default_factory=set)
    grounding_fields_to_tokenize: list[str] | None = None

    @property
    def has_loss_mask(self) -> bool:
        return (
            self.masking_strategy is not None or self.ngram_blocklist_path is not None
        )

    @model_validator(mode="after")
    def validate_ngram_blocklist_metadata(self) -> typing.Self:
        if (self.ngram_blocklist_path is None) != (self.ngram_blocklist_hash is None):
            raise ValueError(
                "ngram_blocklist_path and ngram_blocklist_hash must either both "
                f"be set or both be None. Got {self.ngram_blocklist_path=} and "
                f"{self.ngram_blocklist_hash=}."
            )
        return self

    @field_validator("filter_name", mode="after")
    @classmethod
    def validate_filter_name(cls, v: FilterArgsName) -> FilterArgsName:
        if not isinstance(v, FilterArgsName):
            return FilterArgsName(v)
        return v

    @field_validator("tokenizer_name", mode="after")
    @classmethod
    def validate_tokenizer_name(cls, v: TokenizerName) -> TokenizerName:
        if not isinstance(v, TokenizerName):
            return TokenizerName(v)
        return v

    @field_validator("masking_strategy", mode="after")
    @classmethod
    def validate_masking_strategy(
        cls, v: MaskingStrategy | None
    ) -> MaskingStrategy | None:
        if v is not None and not isinstance(v, MaskingStrategy):
            return MaskingStrategy(v)
        return v

    @field_validator("rare_token_filter", mode="after")
    @classmethod
    def validate_rare_token_filter(
        cls, v: RareTokenFilter | None
    ) -> RareTokenFilter | None:
        if v is not None and not isinstance(v, RareTokenFilter):
            return RareTokenFilter(v)
        return v

    @field_validator("pds_filter", mode="after")
    @classmethod
    def validate_pds_filter(cls, v: PdsFilter | None) -> PdsFilter | None:
        if v is not None and not isinstance(v, PdsFilter):
            return PdsFilter(v)
        return v

    @field_validator("bad_document_filter", mode="after")
    @classmethod
    def validate_bad_document_filter(
        cls, v: BadDocumentFilter | None
    ) -> BadDocumentFilter | None:
        if v is not None and not isinstance(v, BadDocumentFilter):
            return BadDocumentFilter(v)
        return v

    @field_validator("sample_type", mode="after")
    @classmethod
    def validate_sample_type(cls, v: SampleType) -> SampleType:
        if not isinstance(v, SampleType):
            return SampleType(v)
        return v

    @field_validator("available_image_annotations", mode="after")
    @classmethod
    def validate_available_image_annotations(
        cls, v: set[ImageAnnotationType]
    ) -> set[ImageAnnotationType]:
        return {
            item if isinstance(item, ImageAnnotationType) else ImageAnnotationType(item)
            for item in v
        }


class ChunkStats(MistralBase):
    n_docs_train: int
    n_docs_validation: int
    n_docs_filtered: int
    n_docs_total: int
    n_tokens_train: int
    n_docs_tokenization_errors: int
    n_tokens_validation: int = 0
    n_masked_tokens: int = 0
    n_docs_nv_quarantine_filtered: int = 0

    @classmethod
    def merge(cls, all_stats: list[typing.Self]) -> typing.Self:
        return cls(
            n_docs_train=sum(s.n_docs_train for s in all_stats),
            n_docs_validation=sum(s.n_docs_validation for s in all_stats),
            n_docs_filtered=sum(s.n_docs_filtered for s in all_stats),
            n_docs_total=sum(s.n_docs_total for s in all_stats),
            n_tokens_train=sum(s.n_tokens_train for s in all_stats),
            n_tokens_validation=sum(s.n_tokens_validation for s in all_stats),
            n_docs_tokenization_errors=sum(
                s.n_docs_tokenization_errors for s in all_stats
            ),
            n_masked_tokens=sum([s.n_masked_tokens for s in all_stats]),
            n_docs_nv_quarantine_filtered=sum(
                s.n_docs_nv_quarantine_filtered for s in all_stats
            ),
        )
