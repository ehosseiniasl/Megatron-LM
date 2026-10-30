import dataclasses
import logging
from io import BytesIO
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from PIL import Image

# --- Commented out: modules not available in minimal Megatron codebase ---
# from mistral_fork.mistral.conf_store.data.config.filters import ConfStoreFilters
# from mistral_fork.mistral.data.args import DataArgs, DataLoaderType, FilterArgs, ImageDataArgs
# from mistral_fork.mistral.data.pretokenized.build_pretokenized import (
#     _build_components_for_pretokenized_dataloader,
# )
# from mistral_fork.mistral.data.pretokenized.pretokenized_data_loader import (
#     PretokenizedFlexibleImageEncoder,
# )
# from mistral_fork.mistral.data.pretokenized.processing.pretokenize_chunks import (
#     PretrainImagePretokenizer, _process_json_file,
# )
# from mistral_fork.mistral.model.multimodal.image_transforms.args import ImagePreprocessArgs
# from mistral_fork.mistral.model.multimodal.image_transforms.image_transforms import (
#     L2_SCORE_TEST_KEY, BaseImageTransform, DefaultTransform,
# )
# from mistral_fork.mistral.model.multimodal.image_transforms.transform_registry import (
#     ImageTransformType,
# )
# from mistral_fork.mistral_common.tokens.tokenizers.special_tokens import SpecialImageIDs
# from mistral_fork.shared.cluster import Cluster, cluster
# from mistral_fork.shared.geometry import ImageBoundingBox
# from mistral_fork.shared.multimodal_document import MISSING_IMAGE_BUCKETS
# --- End commented out imports ---

from mistral_fork.mistral.data.iterator.utils import SequenceOfSamples
from mistral_fork.mistral.data.pretokenized.cache_key import TokenizerName
from mistral_fork.mistral.data.pretokenized.components import filters as component_filters
from mistral_fork.mistral.data.pretokenized.datasource import (
    ConcatenateLogger,
    CorruptedEpochIndex,
    Datasource,
    DatasourceSpecs,
    EpochStats,
    MissingEpochIndex,
    PretokenizedImageEncoder,
    _gather_epoch_sequences_batched,
    _get_epoch_index_path,
    _is_valid_sequence,
    _make_document_entries,
    _make_image_entries,
    _replace_document_bos_with_eos,
    _scatter_epoch_sequences,
    _shuffle_document_index,
    build_sequence,
    concatenate_iterator,
)
from mistral_fork.mistral.data.pretokenized.document_filters import PdsFilter
from mistral_fork.mistral.data.pretokenized.filter_profiles import FilterArgsName
from mistral_fork.mistral.data.pretokenized.image_transform_metadata import (
    TransformImageMetadata,
)
from mistral_fork.mistral.data.pretokenized.indexed_mask import IndexedMaskBuilder
from mistral_fork.mistral.data.pretokenized.masking import DocumentMask
from mistral_fork.mistral.data.pretokenized.megatron_index_reader import (
    IndexedDatasetBuilder,
    _IndexReader,
    _ListTokensBinReader,
)
from mistral_fork.mistral.data.pretokenized.metadata import (
    DocumentIndexEntry,
    ImageIndexEntry,
    TextIndexEntry,
    _MetadataReader,
    _TarOffset,
    dump_metadata,
)
from mistral_fork.mistral.data.pretokenized.pretokenized_args import PretokenizedArgs
from mistral_fork.mistral.data.pretokenized.processing.datasource_metadata import (
    MaskingStrategy,
    PretokenizedDatasourceMetadataV2,
)
from mistral_fork.mistral.data.pretokenized.processing.paths_helpers import (
    PrefixPaths,
    get_datasource_metadata_path,
    get_datasource_metadata_v2_path,
)
from mistral_fork.mistral.data.source import SampleType
from mistral_fork.mistral_common.tokens.tokenizers.base import Span
from mistral_fork.shared.image_utils import TrainingImageEncoding
from mistral_fork.shared.tartar import TAR_HEADER_SIZE, DiskTar, MemoryTar

if TYPE_CHECKING:
    from mistral_fork.shared.common_utils import TarOffset

BOS_ID = -1
EOS_ID = -2
PAD_ID = -3
IMG_ID = -4
IMG_SEP = -5
IMG_END = -6
TOKEN_ID_DROPOUT = -7
TOKEN_ID_PREFIX = -8
TOKEN_ID_MIDDLE = -9
TOKEN_ID_SUFFIX = -10
# IMAGE_IDS = SpecialImageIDs(IMG_ID, IMG_SEP, IMG_END)  # SpecialImageIDs not available


def test_shuffle_document_index_preserves_existing_order_and_alignment() -> None:
    doc_ids = np.array([11, 4, 8, 2, 17], dtype=np.int32)
    sequence_lengths = np.array([101, 102, 103, 104, 105], dtype=np.int32)
    sequence_pointers = np.array([201, 202, 203, 204, 205], dtype=np.int64)
    seed = (7, 3, len(doc_ids), 0)

    expected_doc_ids = doc_ids.copy()
    np.random.RandomState(seed).shuffle(expected_doc_ids)
    expected_by_doc_id = {
        int(doc_id): (int(length), int(pointer))
        for doc_id, length, pointer in zip(
            doc_ids, sequence_lengths, sequence_pointers, strict=True
        )
    }

    shuffled_doc_ids, shuffled_lengths, shuffled_pointers = _shuffle_document_index(
        doc_ids,
        sequence_lengths,
        sequence_pointers,
        seed,
    )

    np.testing.assert_array_equal(shuffled_doc_ids, expected_doc_ids)
    assert [
        (int(length), int(pointer))
        for length, pointer in zip(
            shuffled_lengths, shuffled_pointers, strict=True
        )
    ] == [expected_by_doc_id[int(doc_id)] for doc_id in shuffled_doc_ids]


def test_epoch_index_skips_documents_with_quarantined_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quarantined_image = object()
    clean_image = object()
    monkeypatch.setattr(
        component_filters,
        "image_index_entry_is_pds_quarantined",
        lambda image, pds_filter: image is quarantined_image,
    )

    class RecordingPool:
        def __init__(self) -> None:
            self.tasks: list[object] = []

        def imap(self, function: object, tasks: object, chunksize: int):
            self.tasks = list(tasks)
            return iter(())

    pool = RecordingPool()
    stats_dict = EpochStats(n_removed_imgs_too_big=0).to_dict()
    entries = list(
        _make_document_entries(
            doc_ids=np.array([0, 1], dtype=np.int32),
            sequence_lengths=np.array([4, 3], dtype=np.int32),
            sequence_pointers=np.array([0, 16], dtype=np.int64),
            epoch_id=0,
            metadata_reader=[[quarantined_image, clean_image], []],
            image_encoder=object(),
            item_size=4,
            n_documents=2,
            specs=_DUMMY_SPECS,
            mask_reader=None,
            stats_dict=stats_dict,
            image_transform_pool=pool,
            epoch_index_pds_filter=PdsFilter.NVIDIA_PDS_QUARANTINE_V3,
        )
    )

    assert entries[0] is None
    assert entries[1] is not None
    assert entries[1].doc_id == 1
    assert stats_dict["n_removed_docs_online_quarantine"] == 1
    assert stats_dict["n_removed_toks_online_quarantine"] == 3
    assert stats_dict["n_removed_imgs_online_quarantine"] == 2
    assert pool.tasks == []


def test_filtered_epoch_index_can_be_loaded_for_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix_path, _, _, num_toks, _ = _build_datasource(
        tmp_path / "source",
        datasource_id=0,
        original_source="original",
        max_seq_length=12,
        rng=np.random.RandomState(seed=0),
        n_docs=12,
        add_images=True,
    )
    metadata_reader = _MetadataReader(str(prefix_path) + ".metadata")
    try:
        quarantined_offset = metadata_reader[0][0].tar_offset.offset
    finally:
        metadata_reader.close()
    monkeypatch.setattr(
        component_filters,
        "image_index_entry_is_pds_quarantined",
        lambda image, pds_filter: image.tar_offset.offset == quarantined_offset,
    )
    monkeypatch.setattr(
        component_filters,
        "pds_filter_args",
        lambda pds_filter: {"manifests": [], "n_keys": 1, "n_rows": 1},
    )
    specs = DatasourceSpecs(
        prefix_path=str(prefix_path),
        seed=0,
        seq_len=32,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
        percentage_data=1.0,
        no_shuffling=True,
        no_partial_sequence=False,
    )
    common_kwargs = {
        "specs": specs,
        "num_toks": num_toks,
        "image_encoder_build_fn": lambda _: DebugImageEncoder(),
        "image_transform_type": "no_transform",
        "selection_strategy": None,
    }
    filtered_root = tmp_path / "filtered"
    filtered = Datasource(
        cache_root_path=filtered_root,
        allow_epoch_index_building=True,
        epoch_index_pds_filter=PdsFilter.NVIDIA_PDS_QUARANTINE_V3,
        **common_kwargs,
    )

    assert filtered._epoch_stats[0].n_removed_docs_online_quarantine == 1
    assert filtered._epoch_stats[0].n_removed_toks_online_quarantine > 0
    assert filtered._epoch_stats[0].n_removed_imgs_online_quarantine > 0
    filtered_tokens = [
        token
        for sequence_id in range(filtered._sequence_count_per_epoch[0])
        for token in filtered.get_sequence(sequence_id).x
    ]
    assert get_token_id(0, 0, 0) not in filtered_tokens

    loaded_for_training = Datasource(
        cache_root_path=filtered_root,
        allow_epoch_index_building=False,
        **common_kwargs,
    )
    assert loaded_for_training._epoch_stats == filtered._epoch_stats

    with pytest.raises(RuntimeError, match="built with"):
        Datasource(
            cache_root_path=filtered_root,
            allow_epoch_index_building=False,
            epoch_index_pds_filter=PdsFilter.NVIDIA_PDS_QUARANTINE_V1,
            **common_kwargs,
        )


def test_concatenate_logger_reports_document_token_and_image_rates(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    times = iter([100.0, 102.0])
    monkeypatch.setattr(
        "mistral_fork.mistral.data.pretokenized.datasource.time.monotonic",
        lambda: next(times),
    )
    caplog.set_level(logging.INFO)

    progress_logger = ConcatenateLogger(
        n_documents=4,
        epoch_id=3,
        log_every_percent=10.0,
    )
    progress_logger.on_document_processed(n_text_tokens=2_048, n_images=4)

    assert any(
        "[Epoch 3] Processed 25.0%, 0.5 doc/s, 1024.0 tokens/s, 2.0 img/s"
        in record.message
        for record in caplog.records
    )


def get_token_id(datasource_id: int, doc_id: int, pos_in_doc: int) -> int:
    assert datasource_id < 1000
    assert doc_id < 1000
    assert pos_in_doc < 1000
    return datasource_id * 10**6 + doc_id * 10**3 + pos_in_doc


def _create_mask(doc_id: int, doc_size: int) -> DocumentMask:
    assert doc_size >= 3
    no_bos_eos_doc_size = doc_size - 2
    n_segments = max(min(1 + (doc_id % 5), no_bos_eos_doc_size), 1)
    segment_size = no_bos_eos_doc_size // n_segments
    reminder = no_bos_eos_doc_size - segment_size * n_segments
    assert reminder >= 0, reminder
    segments = [segment_size] * n_segments
    if reminder > 0:
        segments[n_segments - 1] += reminder
    values = [
        i % 2 == 0 for i in range(n_segments - 1, -1, -1)
    ]  # reverse order, last one is always 1
    segments[-1] += 1  # adding EOS
    if values[0]:
        segments[0] += 1
    else:
        # we append True at the beginning
        segments = [1, *segments]
        values = [True, *values]
    assert sum(segments) == doc_size
    return DocumentMask(sizes=segments, mask=values)


def _build_expected_text_mask_array(doc_id: int, doc_size: int) -> np.ndarray:
    # note that we build here the expected text mask array for text, not full mask array
    mask = _create_mask(doc_id=doc_id, doc_size=doc_size)
    return np.repeat(a=mask.mask, repeats=mask.sizes)


def _check_mask(sequence: SequenceOfSamples, doc_lengths: list[int]) -> None:
    print("sequence_length", len(sequence.y), sequence.sizes)
    for y_tok_id, mask_value in zip(sequence.y, sequence.mask, strict=True):
        assert y_tok_id != BOS_ID
        if y_tok_id == EOS_ID:
            print("EOS_ID")
            assert mask_value
        elif y_tok_id == IMG_ID:
            assert not mask_value
            print("IMG_ID")
        else:
            _, doc_id, pos_in_doc = parse_token_id(y_tok_id)
            doc_size = doc_lengths[doc_id]
            mask = _create_mask(doc_id=doc_id, doc_size=doc_size)
            text_mask_array = _build_expected_text_mask_array(
                doc_id=doc_id, doc_size=doc_size
            )
            assert len(text_mask_array) == doc_size, (
                f"mask array supposed to have same size than doc_size, did you removed the BOS from it {len(text_mask_array)} != {doc_size}"
            )
            print(
                doc_id,
                mask,
                pos_in_doc,
                mask_value,
                text_mask_array[pos_in_doc + 1] == mask_value,
            )
            # +1 since pos_in_doc don't take bos into account
            # assert text_mask_array[pos_in_doc + 1] == mask_value, (text_mask_array, pos_in_doc)


def parse_token_id(tok_id: int) -> tuple[int, int, int]:
    assert tok_id >= 0, tok_id
    datasource_id = tok_id // (10**6)
    doc_id = (tok_id // 1000) % 1000
    pos_in_doc = tok_id % 1000
    return datasource_id, doc_id, pos_in_doc


MAX_RGB_INT = (1 << 24) - 1  # 16,777,215


def pil_to_bytes(img: Image.Image) -> bytes:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_dummy_pil_image_with_integer(
    height: int, width: int, value: int
) -> Image.Image:
    if not (0 <= value <= MAX_RGB_INT):
        raise ValueError(
            f"Value must be between 0 and {MAX_RGB_INT} (inclusive). Got: {value}"
        )

    r = (value >> 16) & 0xFF
    g = (value >> 8) & 0xFF
    b = value & 0xFF

    return Image.new("RGB", (height, width), (r, g, b))


def read_integer_from_image_bytes(image_bytes: bytes) -> int:
    buffer = BytesIO(image_bytes)
    img = Image.open(buffer)
    return _integer_from_image(img)


def _integer_from_image(img: Image.Image | np.ndarray) -> int:
    data = np.array(img)
    r, g, b = data[0, 0]
    return (r << 16) | (g << 8) | b


def _build_datasource(
    dst: Path,
    datasource_id: int,
    original_source: str,
    n_docs: int,
    max_seq_length: int,
    rng: np.random.RandomState,
    add_images: bool = False,
    add_mask: bool = False,
    max_image_size: int = 10,
) -> tuple[Path, set[int], int, int, int]:
    """Returns present tokens"""
    assert not dst.exists(), dst
    dst.mkdir(parents=True)
    prefix_path = dst / "chunks_0000-00001"
    num_text_tokens = 0
    num_images = 0
    present_tokens: set[int] = set()

    tar_path = Path(str(prefix_path) + ".tar")
    tmp_folder = str(prefix_path) + "/tmp"
    Path(tmp_folder).mkdir(exist_ok=False, parents=True)
    builder = IndexedDatasetBuilder(str(prefix_path))
    metadata = []
    imgs: list[Image.Image] = []
    document_masks: list[DocumentMask] = []
    with DiskTar.open(tar_path, compress=False) as f_tar:
        for doc_id in range(n_docs):
            doc_size = rng.randint(1, max_seq_length)
            doc = [BOS_ID]

            for pos_is_doc in range(doc_size):
                tok_id = get_token_id(datasource_id, doc_id, pos_is_doc)
                doc.append(tok_id)
                assert tok_id not in present_tokens, (tok_id, present_tokens)
                present_tokens.add(tok_id)

            if add_images:
                n_images = min(doc_size, 10)
                positions_in_doc = sorted(
                    rng.randint(1, doc_size + 1, size=(n_images,))
                )
                # we encode the "next" text token in image as image uuid
                imgs_as_integers = [doc[pos] for pos in positions_in_doc]
                imgs = [
                    create_dummy_pil_image_with_integer(
                        height=rng.randint(1 + max_image_size // 4, max_image_size),
                        width=rng.randint(1 + max_image_size // 4, max_image_size),
                        value=integer,
                    )
                    for integer in imgs_as_integers
                ]
                t = MemoryTar()
                img_offsets: list[TarOffset] = []
                for i, (img, integer) in enumerate(
                    zip(imgs, imgs_as_integers, strict=True)
                ):
                    img_bytes = pil_to_bytes(img)
                    retrieved_img_id = read_integer_from_image_bytes(img_bytes)
                    assert retrieved_img_id == integer, (retrieved_img_id, integer)
                    img_offsets.append(t.write_bytes(img_bytes, name=f"{i}"))
                tmp_tar = Path(tmp_folder) / f"{doc_id}.tar"
                t.export(tmp_tar)
                doc_offset = f_tar.write_file(tmp_tar)

                tar_offsets = [
                    _TarOffset(
                        offset=doc_offset.offset
                        + TAR_HEADER_SIZE
                        + sub_tar_offset.offset,
                        size=sub_tar_offset.size,
                        new_tar_format=sub_tar_offset.new_tar_format,
                    )
                    for sub_tar_offset in img_offsets
                ]
            else:
                positions_in_doc = []
                tar_offsets = []
            doc.append(EOS_ID)
            if add_mask:
                document_masks.append(_create_mask(doc_id=doc_id, doc_size=len(doc)))
            num_text_tokens += len(doc) - 1  # we remove EOD
            assert num_text_tokens >= 2, (doc_id, num_text_tokens)

            imgs_metadata = []
            for img_id, (img, pos_in_doc, tar_offset) in enumerate(
                zip(imgs, positions_in_doc, tar_offsets, strict=True)
            ):
                imgs_metadata.append(
                    ImageIndexEntry(
                        img_id_in_doc=img_id,
                        tar_offset=_TarOffset(
                            offset=tar_offset.offset,
                            size=tar_offset.size,
                            new_tar_format=tar_offset.new_tar_format,
                            tar_path=str(tar_path),
                        ),
                        pos_in_text=int(pos_in_doc),
                        metadata=TransformImageMetadata(
                            width=img.width,
                            height=img.height,
                        ),
                    )
                )
            num_images += len(imgs_metadata)
            metadata.append(imgs_metadata)
            builder.add_document(
                torch.tensor(doc, dtype=torch.int32), lengths=[len(doc)]
            )
        builder.finalize()

        if add_images:
            assert len(metadata) == n_docs
            dump_metadata(
                metadata,
                metadata_path=prefix_path.with_name(prefix_path.name + ".metadata"),
            )

        if add_mask:
            assert len(document_masks) == n_docs
            with IndexedMaskBuilder(PrefixPaths(prefix_path).mask) as mask_builder:
                for doc_mask in document_masks:
                    mask_builder.add_document(doc_mask)

    datasource_metadata_v2 = PretokenizedDatasourceMetadataV2(
        filter_name=FilterArgsName.NO_FILTER,
        tokenizer_name=TokenizerName.PRETRAIN_V3_LOBSTENIZER,
        original_source=original_source,
        tokenizer_path="debug",
        filter_args_json="no filter args",
        masking_strategy=MaskingStrategy.MOCK if add_mask else None,
        n_chunks=1,
        validation_fraction=0.0,
        guarantee_validation_per_chunk=False,
        sample_type=SampleType.PRETRAIN
        if not add_images
        else SampleType.PRETRAIN_IMAGE,
        has_image_metadata=add_images,
    )
    infos_path = get_datasource_metadata_path(dst)
    assert not infos_path.exists()
    infos_path.write_text(datasource_metadata_v2.model_dump_json(indent=2))
    get_datasource_metadata_v2_path(dst).write_text(
        datasource_metadata_v2.model_dump_json(indent=2)
    )

    return prefix_path, present_tokens, n_docs, num_text_tokens, num_images


def _num_toks_per_epochs_and_doc_lengths(
    datasource: Datasource,
) -> tuple[int, list[int]]:
    # can be slow only for tests
    index = _IndexReader(
        idx_path=str(PrefixPaths.from_prefix(datasource._specs.prefix_path).index),
        multimodal=False,
    )
    return int(sum(index.sequence_lengths)) - int(index.sequence_count), list(
        index.sequence_lengths
    )


@pytest.mark.parametrize("no_shuffling", [False, True])
@pytest.mark.parametrize("masking", [False, True])
def test_single_datasource(tmp_path: Path, no_shuffling: bool, masking: bool) -> None:
    rng = np.random.RandomState(seed=0)
    n_sources = 1
    seq_len = 5

    prefix_path, present_tokens, n_docs, num_toks, _ = _build_datasource(
        tmp_path / "single_source",
        datasource_id=0,
        original_source="original_datasource",
        max_seq_length=20,
        rng=rng,
        n_docs=n_sources * 30,
        add_images=False,
        max_image_size=10,
        add_mask=masking,
    )
    print(f"{num_toks=}")
    datasource = Datasource(
        specs=DatasourceSpecs(
            prefix_path=str(prefix_path),
            seq_len=seq_len,
            seed=0,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            no_shuffling=no_shuffling,
            percentage_data=1.0,
            no_partial_sequence=False,
        ),
        num_toks=4 * num_toks,
        cache_root_path=tmp_path / "cache_root",
        image_encoder_build_fn=None,
        image_transform_type=None,
        allow_epoch_index_building=True,
        selection_strategy=None,
    )

    num_toks_per_epoch, doc_lengths = _num_toks_per_epochs_and_doc_lengths(datasource)
    assert num_toks_per_epoch == num_toks

    expected_num_sequences_per_epoch = num_toks_per_epoch // datasource.seq_len

    assert len(doc_lengths) == n_docs

    # 3 epochs
    seq_id = 0
    for epoch_id in range(3):
        num_sequences_per_epoch = datasource._sequence_count_per_epoch[epoch_id]
        if not masking:
            # we don't mask, so we don't drop sequences
            assert expected_num_sequences_per_epoch == num_sequences_per_epoch, (
                expected_num_sequences_per_epoch,
                num_sequences_per_epoch,
            )
        remaining_tokens = (
            num_toks_per_epoch - num_sequences_per_epoch * datasource.seq_len
        )
        seen_tokens: set[int] = set()
        seen_labels: set[int] = set()
        for _ in range(num_sequences_per_epoch):
            sequence = datasource.get_sequence(seq_id)
            assert datasource.current_epoch_index is not None
            assert datasource.current_epoch_index.epoch_id == epoch_id, (
                seq_id,
                datasource.current_epoch_index.epoch_id,
                epoch_id,
            )
            _check_sequence(sequence, expected_datasource_id=0)
            if masking:
                _check_mask(sequence, doc_lengths)
            seq_id += 1

            for tok in sequence.x:
                if tok == BOS_ID:
                    continue
                assert tok != EOS_ID

                assert tok not in seen_tokens, (tok, seen_tokens)
                seen_tokens.add(tok)

            for label in sequence.y:
                if label == EOS_ID:
                    continue
                assert label != BOS_ID
                assert label not in seen_labels, (label, seen_labels)
                seen_labels.add(label)

        assert seen_tokens.issubset(present_tokens)
        assert seen_labels.issubset(present_tokens)

        unseen_tokens = present_tokens - seen_tokens
        unseen_labels = present_tokens - seen_labels
        print(remaining_tokens, unseen_tokens, unseen_labels)

        assert len(unseen_tokens) <= remaining_tokens
        assert len(unseen_labels) <= remaining_tokens


def test_prebuild_specific_epoch_does_not_require_other_epochs(tmp_path: Path) -> None:
    prefix_path, _, _, _, _ = _build_datasource(
        tmp_path / "single_epoch_source",
        datasource_id=0,
        original_source="original_datasource",
        max_seq_length=20,
        rng=np.random.RandomState(seed=0),
        n_docs=30,
        add_images=False,
    )
    specs = DatasourceSpecs(
        prefix_path=str(prefix_path),
        seq_len=5,
        seed=0,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
        no_shuffling=False,
        percentage_data=1.0,
        no_partial_sequence=False,
    )
    cache_root_path = tmp_path / "cache_root"
    kwargs = {
        "specs": specs,
        "cache_root_path": cache_root_path,
        "image_encoder_build_fn": None,
        "image_transform_type": None,
        "selection_strategy": None,
        "skip_images": False,
    }

    Datasource.prebuild_epoch_index(
        epoch_id=2, allow_epoch_index_building=True, **kwargs
    )

    epoch_paths = [
        _get_epoch_index_path(
            epoch_indexes_root_dir=cache_root_path,
            specs=specs,
            epoch_id=epoch_id,
            selection_strategy=None,
            image_encoder=None,
            has_loss_mask=False,
        )
        for epoch_id in range(3)
    ]
    assert [path.exists() for path in epoch_paths] == [False, False, True]

    Datasource.prebuild_epoch_index(
        epoch_id=2, allow_epoch_index_building=False, **kwargs
    )
    with pytest.raises(MissingEpochIndex) as exc_info:
        Datasource.prebuild_epoch_index(
            epoch_id=1, allow_epoch_index_building=False, **kwargs
        )
    assert exc_info.value.path == epoch_paths[1]

    Datasource.prebuild_epoch_index(
        epoch_id=0, allow_epoch_index_building=True, **kwargs
    )
    datasource = Datasource(
        num_toks=1_000_000,
        allow_epoch_index_building=False,
        allow_partial_epoch_index_cache=True,
        **kwargs,
    )
    state = datasource.initial_state()
    datasource.check_state(state)
    datasource._num_sequences_to_validate = datasource._cum_sequence_count_per_epoch[-1]
    datasource._allow_partial_epoch_index_cache = False
    datasource.check_state(state)
    datasource._allow_partial_epoch_index_cache = True
    state.sequence_id = datasource._cum_sequence_count_per_epoch[-1]
    with pytest.raises(MissingEpochIndex) as exc_info:
        datasource.check_state(state)
    assert exc_info.value.path == epoch_paths[1]


@pytest.mark.parametrize("no_shuffling", [False, True])
@pytest.mark.parametrize(
    "masking", [False]
)  # masking True no yet supported for multimodal
def test_single_datasource_with_images(
    tmp_path: Path, no_shuffling: bool, masking: bool
) -> None:
    rng = np.random.RandomState(seed=0)
    n_sources = 1
    seq_len = 10

    prefix_path, present_tokens, n_docs, num_toks, n_expected_images = (
        _build_datasource(
            tmp_path / "single_source",
            datasource_id=0,
            original_source="original_datasource",
            max_seq_length=20,
            rng=rng,
            n_docs=n_sources * 30,
            add_images=True,
            add_mask=masking,
        )
    )
    datasource = Datasource(
        specs=DatasourceSpecs(
            prefix_path=str(prefix_path),
            seq_len=seq_len,
            seed=0,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            percentage_data=1.0,
            no_shuffling=no_shuffling,
            no_partial_sequence=False,
        ),
        num_toks=10 * num_toks,
        cache_root_path=tmp_path / "cache_root",
        image_encoder_build_fn=lambda x: DebugImageEncoder(),
        image_transform_type="no_transform",
        allow_epoch_index_building=True,
        selection_strategy=None,
    )

    assert len(datasource._cum_sequence_count_per_epoch) > 2, (
        datasource._cum_sequence_count_per_epoch
    )
    epoch_stats = datasource._epoch_stats[0]

    assert isinstance(epoch_stats, EpochStats)

    # we check that test have images / consecutives images too big
    assert epoch_stats.n_removed_imgs_too_big > 0
    assert epoch_stats.n_removed_imgs_consecutive > 0

    seq_id = 0
    for epoch_id in range(2):
        seen_tokens: set[int] = set()
        seen_labels: set[int] = set()
        n_images = 0

        for _ in range(datasource._sequence_count_per_epoch[epoch_id]):
            print(f"sampling sequence {seq_id}")
            sequence = datasource.get_sequence(seq_id)
            n_images += len(sequence.images)

            _check_sequence(sequence, expected_datasource_id=0)

            for tok in sequence.x:
                if tok in {BOS_ID, IMG_ID, PAD_ID}:
                    continue
                assert tok != EOS_ID
                assert tok not in seen_tokens, (
                    tok,
                    seen_tokens,
                    datasource._cum_sequence_count_per_epoch,
                    sequence.x,
                    seq_id,
                )
                seen_tokens.add(tok)

            for label in sequence.y:
                if label in {EOS_ID, IMG_ID, PAD_ID}:
                    continue
                assert label != BOS_ID
                assert label not in seen_labels, (label, seen_labels)
                seen_labels.add(label)

            seq_id += 1

        assert seen_tokens.issubset(present_tokens)
        assert seen_labels.issubset(present_tokens)

        unseen_tokens = present_tokens - (seen_tokens | seen_labels)

        assert len(unseen_tokens) < seq_len

        epoch_stats = datasource._epoch_stats[epoch_id]
        assert n_images == epoch_stats.n_imgs
        assert (
            n_expected_images
            == n_images
            + epoch_stats.n_removed_imgs_consecutive
            + epoch_stats.n_removed_imgs_too_big
            + epoch_stats.n_removed_imgs_too_small
            + epoch_stats.n_removed_imgs_end_epoch
        ), (n_expected_images, n_images, epoch_stats)


def test_image_transform_process_pool_matches_serial(tmp_path: Path) -> None:
    rng = np.random.RandomState(seed=0)
    prefix_path, _, _, num_toks, _ = _build_datasource(
        tmp_path / "source",
        datasource_id=0,
        original_source="original_datasource",
        max_seq_length=20,
        rng=rng,
        n_docs=100,
        add_images=True,
    )
    specs = DatasourceSpecs(
        prefix_path=str(prefix_path),
        seq_len=10,
        seed=0,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
        percentage_data=1.0,
        no_shuffling=False,
        no_partial_sequence=False,
    )

    for workers in (1, 2):
        Datasource(
            specs=specs,
            num_toks=num_toks,
            cache_root_path=tmp_path / f"cache_{workers}",
            image_encoder_build_fn=lambda _: DebugImageEncoder(),
            image_transform_type="no_transform",
            allow_epoch_index_building=True,
            selection_strategy=None,
            image_transform_workers=workers,
        )

    serial_files = sorted(
        path.relative_to(tmp_path / "cache_1")
        for path in (tmp_path / "cache_1").rglob("*")
        if path.is_file()
    )
    parallel_files = sorted(
        path.relative_to(tmp_path / "cache_2")
        for path in (tmp_path / "cache_2").rglob("*")
        if path.is_file()
    )
    assert parallel_files == serial_files
    for relative_path in serial_files:
        assert (tmp_path / "cache_2" / relative_path).read_bytes() == (
            tmp_path / "cache_1" / relative_path
        ).read_bytes()


# DISABLED: test_cache_transform_per_source
# Requires: DataArgs, ImageDataArgs, ImagePreprocessArgs, DataLoaderType,
# _build_components_for_pretokenized_dataloader, PretokenizedFlexibleImageEncoder


# DISABLED: test_image_transforms
# Requires: ImageTransformType, PretokenizedFlexibleImageEncoder, IMAGE_IDS

# DISABLED: test_real_data_image_transforms
# Requires: ImageTransformType, Cluster, cluster, ConfStoreFilters, DataArgs,
# ImageDataArgs, FilterArgs, DataLoaderType, MULTIMODAL_MEDIA_SOURCES,
# MediaSource, MediaSourceType, PretrainImagePretokenizer, _process_json_file,
# MultimodalJSONLDataDir, SpecialImageIDs, PretokenizedFlexibleImageEncoder


def _check_sequence(sequence: SequenceOfSamples, expected_datasource_id: int) -> None:
    x = sequence.x
    y = sequence.y
    y_mask = sequence.mask
    doc_start_offset = 0
    for size in sequence.sizes:
        doc_chunk = x[doc_start_offset : doc_start_offset + size]
        y_chunk = y[doc_start_offset : doc_start_offset + size]
        y_mask_chunk = y_mask[doc_start_offset : doc_start_offset + size]
        # print("x", [i for i in doc_chunk])
        # print("y", [j for j in y_chunk])
        # print("y_mask", [j for j in y_mask_chunk])
        if doc_start_offset != 0:
            assert doc_chunk[0] == BOS_ID

        expected_pos_in_doc = None
        expected_doc_id = None
        padding_started = False
        for i, tok in enumerate(doc_chunk):
            offset_in_sequence = doc_start_offset + i

            assert tok != EOS_ID

            if tok == PAD_ID:
                assert i > 0, (i, x, y, sequence)
                if not padding_started:
                    # if we are first pad for this doc in x, previous y is not a pad
                    assert y_mask_chunk[i - 1], (
                        i,
                        tok,
                        y_chunk[i - 1],
                        y_mask_chunk[i - 1],
                    )
                    assert y_chunk[i - 1] != PAD_ID, (
                        i,
                        tok,
                        y_chunk[i - 1],
                        y_mask_chunk[i - 1],
                    )
                else:
                    assert not y_mask_chunk[i - 1], (
                        i,
                        tok,
                        y_chunk[i - 1],
                        y_mask_chunk[i - 1],
                    )
                    assert y_chunk[i - 1] == PAD_ID, (
                        i,
                        tok,
                        y_chunk[i - 1],
                        y_mask_chunk[i - 1],
                    )
                padding_started = True
                continue

            assert not padding_started, (sequence, "pad not a the end")

            if tok == BOS_ID:
                assert i == 0, sequence
            elif tok == IMG_ID:
                if i == 0:
                    assert offset_in_sequence == 0, (
                        "only first doc can start with image"
                    )
                else:
                    assert not y_mask_chunk[i - 1], (
                        i,
                        tok,
                        y_chunk[i - 1],
                        y_mask_chunk[i - 1],
                    )
                    assert y_chunk[i - 1] == IMG_ID, (
                        i,
                        tok,
                        y_chunk[i - 1],
                        y_mask_chunk[i - 1],
                    )
            else:
                if i > 0:
                    assert y_chunk[i - 1] == tok, (
                        i,
                        tok,
                        y_chunk[i - 1],
                        y_mask_chunk[i - 1],
                    )
                datasource_id, doc_id, pos_in_doc = parse_token_id(tok)
                if expected_pos_in_doc is None:
                    expected_pos_in_doc = pos_in_doc
                if expected_doc_id is None:
                    expected_doc_id = doc_id

                assert doc_id == expected_doc_id, (doc_id, expected_doc_id, sequence)
                assert pos_in_doc == expected_pos_in_doc, (
                    pos_in_doc,
                    expected_pos_in_doc,
                    sequence,
                )
                assert datasource_id == expected_datasource_id, (
                    datasource_id,
                    expected_datasource_id,
                )

                expected_pos_in_doc += 1
        doc_start_offset += size

        images = sequence.images
        spans = sequence.image_positions

        # assert False, images[0].dtype
        images_as_integers = [
            _integer_from_image(img.to(torch.uint8).numpy()) for img in images
        ]

        for i, (image, span, image_as_integer) in enumerate(
            zip(images, spans, images_as_integers, strict=True)
        ):
            if _next_image_is_contiguous(i, spans):
                assert images_as_integers[i] == images_as_integers[i + 1], (
                    i,
                    images_as_integers,
                    sequence,
                )
            else:
                # the next token to be predicted needs to match the id of the image (by construction)
                label = y[span.end - 1]
                print(f"{x=}")
                print(f"{y=}")
                assert label == image_as_integer, (
                    label,
                    image_as_integer,
                    x,
                    y,
                    span,
                    i,
                    spans,
                )
        # TODO: check span don't overlap
        # TODO: check that token IMG <=> tok inside a span
        #  check that EOS is only used in label at the end of a sequence
        #  Test that that the label can't be an EOS


def _next_image_is_contiguous(image_index: int, spans: list[Span]) -> bool:
    if image_index + 1 >= len(spans):
        return False
    return spans[image_index].end == spans[image_index + 1].start


class DebugImageEncoder(PretokenizedImageEncoder):
    def __init__(self, keep_original_size: bool = False) -> None:
        # useful to get predictable image size
        self.keep_original_size = keep_original_size

    def uuid(self) -> str:
        return "debug"

    def to_infos_dict(self) -> dict[str, Any]:
        return {"type": "debug"}

    def apply_transforms_and_encode_image(
        self,
        image: Image.Image,
        metadata: TransformImageMetadata,
        seed: tuple[int, ...],
    ) -> tuple[TrainingImageEncoding, TransformImageMetadata]:
        n_toks, new_metadata = self.precompute_transforms(metadata=metadata, seed=seed)
        return TrainingImageEncoding(
            tokens=[IMG_ID] * n_toks, image=np.array(image)
        ), new_metadata

    def precompute_transforms(
        self, metadata: TransformImageMetadata, seed: tuple[int, ...]
    ) -> tuple[int, TransformImageMetadata]:
        rng = np.random.RandomState(seed)
        if not self.keep_original_size:
            new_height = max(1, metadata.height - rng.randint(1, 4))
            new_metadata = dataclasses.replace(metadata, height=new_height)
        else:
            new_metadata = metadata
        n_toks = new_metadata.height + new_metadata.width
        assert n_toks > 0, n_toks
        return n_toks, new_metadata

    def get_transforms(self) -> Any:
        raise NotImplementedError("No image transforms in minimal codebase")


@pytest.mark.parametrize("no_shuffling", [False, True])
def test_single_datasource_percentage_data(tmp_path: Path, no_shuffling: bool) -> None:
    rng = np.random.RandomState(seed=0)
    n_sources = 1
    seq_len = 5

    prefix_path, present_tokens, n_docs, num_toks, _ = _build_datasource(
        tmp_path / "single_source",
        datasource_id=0,
        original_source="original_datasource",
        max_seq_length=20,
        rng=rng,
        n_docs=n_sources * 30,
        add_images=False,
    )

    def make(percentage_data: float) -> Datasource:
        return Datasource(
            specs=DatasourceSpecs(
                prefix_path=str(prefix_path),
                seq_len=seq_len,
                seed=0,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                pad_id=PAD_ID,
                no_shuffling=no_shuffling,
                percentage_data=percentage_data,
                no_partial_sequence=False,
            ),
            num_toks=4 * num_toks,
            cache_root_path=tmp_path / "cache_root",
            image_encoder_build_fn=None,
            image_transform_type=None,
            allow_epoch_index_building=True,
            selection_strategy=None,
        )

    full_datasource = make(percentage_data=1.0)
    half_datasource = make(percentage_data=0.5)

    num_toks_per_epoch, _ = _num_toks_per_epochs_and_doc_lengths(full_datasource)
    assert num_toks_per_epoch == num_toks

    num_sequences_per_epoch = num_toks_per_epoch // seq_len
    assert full_datasource._cum_sequence_count_per_epoch[0] == num_sequences_per_epoch
    assert (
        num_sequences_per_epoch // 2 + 1
        >= half_datasource._cum_sequence_count_per_epoch[0]
        > num_sequences_per_epoch // 2
    )


def test_datasource_iteration_does_not_log_started_iteration_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rng = np.random.RandomState(seed=0)
    prefix_path, _, _, num_toks, _ = _build_datasource(
        tmp_path / "single_source",
        datasource_id=0,
        original_source="original_datasource",
        max_seq_length=20,
        rng=rng,
        n_docs=30,
        add_images=False,
    )

    datasource = Datasource(
        specs=DatasourceSpecs(
            prefix_path=str(prefix_path),
            seq_len=5,
            seed=0,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            no_shuffling=True,
            percentage_data=1.0,
            no_partial_sequence=False,
        ),
        num_toks=4 * num_toks,
        cache_root_path=tmp_path / "cache_root",
        image_encoder_build_fn=None,
        image_transform_type=None,
        allow_epoch_index_building=True,
        selection_strategy=None,
    )

    caplog.clear()
    caplog.set_level(logging.INFO)
    datasource.get_sequence(0)

    assert not any("Started iteration" in record.message for record in caplog.records)


@pytest.mark.parametrize("no_shuffling", [True, False])
def test_single_datasource_no_shuffling(tmp_path: Path, no_shuffling: bool) -> None:
    rng = np.random.RandomState(seed=0)
    n_sources = 1
    seq_len = 5

    prefix_path, present_tokens, n_docs, num_toks, _ = _build_datasource(
        tmp_path / "single_source",
        datasource_id=0,
        original_source="original_datasource",
        max_seq_length=20,
        rng=rng,
        n_docs=n_sources * 30,
        add_images=False,
    )

    def make(no_shuffling: bool) -> Datasource:
        return Datasource(
            specs=DatasourceSpecs(
                prefix_path=str(prefix_path),
                seq_len=seq_len,
                seed=0,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                pad_id=PAD_ID,
                no_shuffling=no_shuffling,
                percentage_data=1.0,
                no_partial_sequence=False,
            ),
            num_toks=4 * num_toks,
            cache_root_path=tmp_path / "cache_root",
            image_encoder_build_fn=None,
            image_transform_type=None,
            allow_epoch_index_building=True,
            selection_strategy=None,
        )

    datasource = make(no_shuffling=no_shuffling)

    first_epoch = [
        datasource.get_sequence(seq_id=i)
        for i in range(datasource._cum_sequence_count_per_epoch[0])
    ]
    second_epoch = [
        datasource.get_sequence(seq_id=i)
        for i in range(
            datasource._cum_sequence_count_per_epoch[0],
            datasource._cum_sequence_count_per_epoch[1],
        )
    ]

    if no_shuffling:
        assert first_epoch == second_epoch
    else:
        assert first_epoch != second_epoch


def _img_index(
    img_id_in_doc: int, size: tuple[int, int], pos_in_text: int
) -> ImageIndexEntry:
    return ImageIndexEntry(
        img_id_in_doc=img_id_in_doc,
        tar_offset=_TarOffset(offset=23, size=12, tar_path="tar_path"),
        pos_in_text=pos_in_text,
        metadata=TransformImageMetadata(width=size[0], height=size[1]),
        size=size[0] + size[1],
        seed=(0, 0, 0, 0, 0),
    )


@pytest.mark.parametrize(
    ("drop_bos", "expected_x"),
    [
        (False, [BOS_ID, 0, 1, BOS_ID, 2, 3, 4]),
        (True, [EOS_ID, 0, 1, EOS_ID, 2, 3, 4]),
    ],
)
def test_runtime_drop_bos_preserves_sizes_and_doc_ids(
    drop_bos: bool, expected_x: list[int]
) -> None:
    stored_tokens = [
        BOS_ID,
        0,
        1,
        EOS_ID,
        BOS_ID,
        2,
        3,
        4,
        EOS_ID,
        BOS_ID,
        5,
        EOS_ID,
    ]
    docs = [
        DocumentIndexEntry(0, TextIndexEntry(0, 3), [], 3),
        DocumentIndexEntry(1, TextIndexEntry(4, 4), [], 4),
        DocumentIndexEntry(2, TextIndexEntry(9, 2), [], 2),
    ]
    packed = list(
        concatenate_iterator(
            chunked_doc_iterator=iter(docs),
            seq_len=7,
            no_partial_sequence=False,
        )
    )
    assert len(packed) == 1

    result = build_sequence(
        docs=packed[0],
        image_encoder=None,
        bin_reader=_ListTokensBinReader(stored_tokens),
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        is_masked_pretrain=False,
        drop_bos=drop_bos,
    )

    assert result.x == expected_x
    assert result.y == [0, 1, EOS_ID, 2, 3, 4, EOS_ID]
    assert result.sizes == [3, 4]
    assert result.doc_ids == [0, 1]
    result.check_pretrain_sample(
        seq_len=7,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        drop_bos=drop_bos,
    )


def test_runtime_drop_bos_matches_non_interdocument_stream() -> None:
    sizes = [2, 3, 1]
    x = _replace_document_bos_with_eos(
        x=[0, 1, BOS_ID, 2, 3, BOS_ID],
        sizes=sizes,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )
    sequence = SequenceOfSamples(
        x=x,
        y=[1, EOS_ID, 2, 3, EOS_ID, 4],
        mask=[True] * 6,
        sizes=sizes,
        doc_ids=[0, 1, 2],
    )

    assert sequence.x == [0, 1, EOS_ID, 2, 3, EOS_ID]
    assert sequence.sizes == [2, 3, 1]
    sequence.check_pretrain_sample(
        seq_len=6,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        drop_bos=True,
    )


@pytest.mark.parametrize(
    ("first_token", "expected_first_token"),
    [(BOS_ID, EOS_ID), (EOS_ID, EOS_ID), (0, 0)],
)
def test_runtime_drop_bos_handles_first_token(
    first_token: int, expected_first_token: int
) -> None:
    result = _replace_document_bos_with_eos(
        x=[first_token, 1, 2],
        sizes=[3],
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )

    assert result == [expected_first_token, 1, 2]


def test_runtime_drop_bos_rejects_bos_inside_document() -> None:
    with pytest.raises(AssertionError, match="only at indexed document boundaries"):
        _replace_document_bos_with_eos(
            x=[BOS_ID, 1, BOS_ID, 2, BOS_ID, 3],
            sizes=[4, 2],
            bos_id=BOS_ID,
            eos_id=EOS_ID,
        )


def test_concatenate_no_drop_images() -> None:
    # we check that when the number of consecutive image tokens is less than seq_len, no images are dropped
    document_entry = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=12),
        imgs=[
            _img_index(img_id_in_doc=0, size=(2, 3), pos_in_text=6),
            _img_index(img_id_in_doc=1, size=(2, 3), pos_in_text=6),
            _img_index(img_id_in_doc=2, size=(2, 3), pos_in_text=6),
        ],
        size=12 + 5 + 5 + 5,
    )

    stats_dict = EpochStats(n_removed_imgs_too_big=0).to_dict()

    finished_sequences, current_sequence = _concatenate(
        document_entries=[document_entry],
        seq_len=20,
        stats_dict=stats_dict,
    )
    pprint(finished_sequences)
    assert len(finished_sequences) == 2  # 5 + 20
    seq1 = finished_sequences[0]
    (entry1,) = seq1
    assert len(seq1) == 1
    assert len(entry1.imgs) == 0
    assert entry1.size == 6  # only text
    seq2 = finished_sequences[1]
    (entry2,) = seq2
    assert len(entry2.imgs) == 3
    assert entry2.size == 21


def test_concatenate_split_on_text() -> None:
    # we check that is the split size fall on text we do split the text
    document_entry = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=12),
        imgs=[
            _img_index(img_id_in_doc=0, size=(2, 3), pos_in_text=6),
            _img_index(img_id_in_doc=1, size=(2, 3), pos_in_text=6),
            _img_index(img_id_in_doc=2, size=(2, 3), pos_in_text=11),
        ],
        size=12 + 5 + 5 + 5,
    )

    stats_dict = EpochStats(n_removed_imgs_too_big=0).to_dict()
    seq_len = 20
    finished_sequences, current_sequences = _concatenate(
        document_entries=[document_entry],
        seq_len=seq_len,
        stats_dict=stats_dict,
    )
    pprint(finished_sequences)
    assert len(finished_sequences) == 1  # split on text and drop
    seq1 = finished_sequences[0]
    assert len(seq1) == 1
    (entry1,) = seq1
    assert entry1.text.size == 6 + 5  # 0 -> 5 + 17->21
    assert len(entry1.imgs) == 2
    assert entry1.size == 21  # split on text -> no padding
    assert (
        current_sequences[0].size
        == document_entry.size - len(finished_sequences) * seq_len
    )


def test_concatenate_split_on_image_start() -> None:
    # we check that if the split size fall on the first token of an image
    # we drop the image, we go back to the previous text but we still overlap
    document_entry = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=40),
        imgs=[
            _img_index(img_id_in_doc=0, size=(2, 3), pos_in_text=21),
        ],
        size=40 + 5,
    )

    stats_dict = EpochStats(n_removed_imgs_too_big=0).to_dict()
    seq_len = 20
    finished_sequences, current_sequences = _concatenate(
        document_entries=[document_entry],
        seq_len=seq_len,
        stats_dict=stats_dict,
    )
    pprint(stats_dict)
    pprint(finished_sequences)
    assert len(finished_sequences) == 2  #
    seq1 = finished_sequences[0]
    (entry1,) = seq1
    assert len(seq1) == 1
    assert entry1.text.size == 21  # 0 -> 20
    assert len(entry1.imgs) == 0
    seq2 = finished_sequences[1]
    (entry2,) = seq2
    assert len(seq2) == 1
    assert len(entry2.imgs) == 1
    assert entry2.size == 21  # split on text -> no padding
    assert (
        current_sequences[0].size
        == document_entry.size - len(finished_sequences) * seq_len
    )


def test_concatenate_split_if_seq_equal_seq_len_plus_one() -> None:
    # we check that if the split size fall on the first token of an image
    # we drop the image, we go back to the previous text but we still overlap

    seq_len = 20
    document_entry = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=seq_len + 1),
        imgs=[],
        size=seq_len + 1,
    )

    stats_dict = EpochStats(n_removed_imgs_too_big=0).to_dict()
    finished_sequences, current_sequences = _concatenate(
        document_entries=[document_entry],
        seq_len=seq_len,
        stats_dict=stats_dict,
    )
    pprint(stats_dict)
    pprint(finished_sequences)
    assert len(finished_sequences) == 1
    seq1 = finished_sequences[0]
    (entry1,) = seq1
    assert len(seq1) == 1
    assert entry1.text.size == 21  # 0 -> 20
    assert len(entry1.imgs) == 0
    assert len(current_sequences) == 1
    assert (
        current_sequences[0].size
        == document_entry.size - len(finished_sequences) * seq_len
    )


def test_concatenate_drop_first_image() -> None:
    # now we have too much images, we check that we drop the first one
    stats_dict = EpochStats(n_removed_imgs_too_big=0).to_dict()
    images, n_img_tokens = _make_image_entries(
        specs=DatasourceSpecs(
            prefix_path=str("path"),
            seq_len=20,
            seed=0,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            no_shuffling=False,
            percentage_data=1.0,
            no_partial_sequence=False,
        ),
        images=[
            _img_index(img_id_in_doc=0, size=(6, 4), pos_in_text=6),
            _img_index(img_id_in_doc=1, size=(3, 2), pos_in_text=6),
            _img_index(img_id_in_doc=2, size=(3, 2), pos_in_text=6),
            _img_index(img_id_in_doc=3, size=(3, 2), pos_in_text=6),
        ],
        image_encoder=DebugImageEncoder(keep_original_size=True),
        partial_transform_seed=(0, 0, 0, 0),
        stats_dict=stats_dict,
    )

    assert len(images) == 3  # we removed the first one

    document_entry = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=12),
        imgs=images,
        size=12 + 5 + 5 + 5,
    )

    concatenated, _ = _concatenate(
        document_entries=[document_entry],
        seq_len=20,
        stats_dict=stats_dict,
    )
    pprint(concatenated)
    assert len(concatenated) == 2  # 6 + 21
    seq1 = concatenated[0]
    (entry1,) = seq1
    assert len(seq1) == 1
    assert len(entry1.imgs) == 0
    assert entry1.size == 6  # only text
    seq2 = concatenated[1]
    (entry2,) = seq2
    assert len(entry2.imgs) == 3
    assert entry2.size == 21
    assert [img.img_id_in_doc for img in entry2.imgs] == [
        1,
        2,
        3,
    ]  # we removed the first one


def test_concatenate_drop_first_image_if_contiguous_img_sizes_equal_seq_len_minus_1() -> (
    None
):
    stats_dict = EpochStats(n_removed_imgs_too_big=0).to_dict()
    images, n_img_tokens = _make_image_entries(
        specs=DatasourceSpecs(
            prefix_path=str("path"),
            seq_len=20,
            seed=0,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            no_shuffling=False,
            percentage_data=1.0,
            no_partial_sequence=False,
        ),
        images=[
            _img_index(img_id_in_doc=0, size=(10, 9), pos_in_text=6),
            _img_index(img_id_in_doc=1, size=(2, 2), pos_in_text=6),
            _img_index(img_id_in_doc=2, size=(3, 2), pos_in_text=6),
            _img_index(img_id_in_doc=3, size=(3, 2), pos_in_text=6),
            _img_index(img_id_in_doc=4, size=(3, 2), pos_in_text=6),
        ],
        image_encoder=DebugImageEncoder(keep_original_size=True),
        partial_transform_seed=(0, 0, 0, 0),
        stats_dict=stats_dict,
    )

    assert len(images) == 4  # we removed the first one

    document_entry = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=12),
        imgs=images,
        size=12 + 4 + 5 + 5 + 5,
    )

    concatenated, _ = _concatenate(
        document_entries=[document_entry],
        seq_len=20,
        stats_dict=stats_dict,
    )

    pprint(concatenated)
    assert len(concatenated) == 2  # 6 + 21
    seq1 = concatenated[0]
    (entry1,) = seq1
    assert len(seq1) == 1
    assert len(entry1.imgs) == 0
    assert entry1.size == 6  # only text
    seq2 = concatenated[1]
    (entry2,) = seq2
    assert len(entry2.imgs) == 4
    assert entry2.size == 21


def test_concatenate_too_big_removed() -> None:
    stats_dict = EpochStats(n_removed_imgs_too_big=0).to_dict()
    images, n_img_tokens = _make_image_entries(
        specs=DatasourceSpecs(
            prefix_path=str("path"),
            seq_len=20,
            seed=0,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            no_shuffling=False,
            percentage_data=1.0,
            no_partial_sequence=False,
        ),
        images=[
            _img_index(img_id_in_doc=0, size=(10, 10), pos_in_text=6),
        ],
        image_encoder=DebugImageEncoder(keep_original_size=True),
        partial_transform_seed=(0, 0, 0, 0),
        stats_dict=stats_dict,
    )

    # now we check that we crash if an image is too big
    assert len(images) == 0


# DISABLED: test_make_image_entries_skip_missing_bucket
# Requires: MISSING_IMAGE_BUCKETS from mistral_fork.shared.multimodal_document


def test_concatenate_masking_dont_pad_sequences() -> None:
    # we check that if the split size fall on the first token of an image
    # we drop the image, we go back to the previous text but we still overlap

    seq_len = 20
    document_entry_0 = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=15),
        imgs=[],
        size=15,
        mask=([10, 3, 2], [1, 0, 1]),
    )
    document_entry_0.check()
    document_entry_1 = DocumentIndexEntry(
        doc_id=1,
        text=TextIndexEntry(offset=15, size=15),
        imgs=[],
        size=15,
        mask=([1, 12, 2], [1, 0, 1]),
    )
    document_entry_1.check()

    stats_dict = EpochStats().to_dict()
    finished_sequences, current_sequences = _concatenate(
        document_entries=[document_entry_0, document_entry_1],
        seq_len=seq_len,
        stats_dict=stats_dict,
    )
    pprint(stats_dict)
    pprint(finished_sequences)
    pprint(current_sequences)

    assert finished_sequences == [
        [
            document_entry_0,
            DocumentIndexEntry(
                doc_id=1,
                text=TextIndexEntry(offset=15, size=6),
                imgs=[],
                size=6,
                # mask finish with a False
                mask=([1, 5], [1, 0]),
            ),
        ]
    ]

    assert current_sequences == [
        DocumentIndexEntry(
            doc_id=1,
            text=TextIndexEntry(offset=20, size=10),
            imgs=[],
            size=10,
            mask=([8, 2], [0, 1]),
        )
    ]


def test_concatenate_truncate_trailing_masked_on_split_left() -> None:
    seq_len = 20
    document_entry_0 = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=15),
        imgs=[],
        size=15,
        mask=([10, 3, 2], [1, 0, 1]),
    )
    document_entry_1 = DocumentIndexEntry(
        doc_id=1,
        text=TextIndexEntry(offset=15, size=15),
        imgs=[],
        size=15,
        mask=([3, 10, 2], [1, 0, 1]),
    )

    stats_dict = EpochStats().to_dict()
    finished_sequences, current_sequences = _concatenate(
        document_entries=[document_entry_0, document_entry_1],
        seq_len=seq_len,
        stats_dict=stats_dict,
        truncate_trailing_masked=True,
    )

    assert finished_sequences == [
        [
            document_entry_0,
            DocumentIndexEntry(
                doc_id=1,
                text=TextIndexEntry(offset=15, size=3),
                imgs=[],
                size=3,
                mask=([3], [1]),
            ),
        ]
    ]
    assert current_sequences == [
        DocumentIndexEntry(
            doc_id=1,
            text=TextIndexEntry(offset=20, size=10),
            imgs=[],
            size=10,
            mask=([8, 2], [0, 1]),
        )
    ]
    assert stats_dict["n_removed_toks_trailing_masked"] == 2
    assert stats_dict["n_docs_truncated_trailing_masked"] == 1


def test_concatenate_truncate_handles_mask_longer_than_seq_len() -> None:
    seq_len = 5
    doc = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=22),
        imgs=[],
        size=22,
        mask=([2, 20], [1, 0]),
    )

    stats_dict = EpochStats().to_dict()
    finished_sequences, _ = _concatenate(
        document_entries=[doc],
        seq_len=seq_len,
        stats_dict=stats_dict,
        truncate_trailing_masked=True,
    )

    assert finished_sequences == [
        [
            DocumentIndexEntry(
                doc_id=0,
                text=TextIndexEntry(offset=0, size=2),
                imgs=[],
                size=2,
                mask=([2], [1]),
            )
        ]
    ]
    assert stats_dict["n_removed_toks_trailing_masked"] > 0


def test_concatenate_truncate_keeps_bos_boundary_for_previous_doc() -> None:
    seq_len = 10
    document_entry_0 = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=4),
        imgs=[],
        size=4,
        mask=([4], [1]),
    )
    document_entry_1 = DocumentIndexEntry(
        doc_id=1,
        text=TextIndexEntry(offset=4, size=11),
        imgs=[],
        size=11,
        mask=([1, 10], [1, 0]),
    )

    stats_dict = EpochStats().to_dict()
    finished_sequences, _ = _concatenate(
        document_entries=[document_entry_0, document_entry_1],
        seq_len=seq_len,
        stats_dict=stats_dict,
        truncate_trailing_masked=True,
    )

    assert finished_sequences == [
        [
            document_entry_0,
            DocumentIndexEntry(
                doc_id=1,
                text=TextIndexEntry(offset=4, size=1),
                imgs=[],
                size=1,
                mask=([1], [1]),
            ),
        ]
    ]
    assert stats_dict["n_removed_toks_trailing_masked"] == 5


def test_concatenate_masking_remove_invalid_sequences() -> None:
    seq_len = 20
    document_entry_0 = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=15),
        imgs=[],
        size=15,
        mask=([10, 3, 2], [1, 0, 1]),
    )
    document_entry_0.check()
    document_entry_1 = DocumentIndexEntry(
        doc_id=1,
        text=TextIndexEntry(offset=15, size=35),
        imgs=[],
        size=35,
        mask=([1, 32, 2], [1, 0, 1]),
    )
    document_entry_1.check()

    stats_dict = EpochStats().to_dict()
    finished_sequences, current_sequences = _concatenate(
        document_entries=[document_entry_0, document_entry_1],
        seq_len=seq_len,
        stats_dict=stats_dict,
    )
    pprint(stats_dict)
    pprint(finished_sequences)
    pprint(current_sequences)

    assert finished_sequences == [
        [
            document_entry_0,
            DocumentIndexEntry(
                doc_id=1,
                text=TextIndexEntry(offset=15, size=6),
                imgs=[],
                size=6,
                mask=([1, 5], [1, 0]),
            ),
        ]
    ]
    NEW_OFFSET = 15 + 6 + 20 - 1
    NEW_SIZE = 35 - 20 - 6 + 1

    assert current_sequences == [
        DocumentIndexEntry(
            doc_id=1,
            text=TextIndexEntry(offset=NEW_OFFSET, size=NEW_SIZE),
            imgs=[],
            size=NEW_SIZE,
            mask=([8, 2], [0, 1]),
        )
    ]


def test_concatenate_masking_removes_sequence_with_only_first_token_trainable() -> None:
    seq_len = 20
    document_entry = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=seq_len + 2),
        imgs=[],
        size=seq_len + 2,
        mask=([1, seq_len, 1], [1, 0, 1]),
    )

    stats_dict = EpochStats().to_dict()
    sequences = list(
        concatenate_iterator(
            chunked_doc_iterator=iter([document_entry]),
            seq_len=seq_len,
            no_partial_sequence=False,
            stats_dict=stats_dict,
        )
    )

    assert sequences == []
    assert stats_dict["num_sequences"] == 0
    assert stats_dict["n_removed_invalid_sequences"] == 1
    assert stats_dict["n_removed_toks_invalid_sequences"] == seq_len


def test_is_valid_sequence_ignores_image_tokens_as_targets() -> None:
    document_entry = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=1),
        imgs=[_img_index(img_id_in_doc=0, size=(10, 10), pos_in_text=1)],
        size=21,
    )

    assert not _is_valid_sequence([document_entry])


@pytest.mark.parametrize(
    "mask",
    [
        ([3], [1]),
        ([1, 1, 1], [1, 0, 1]),
    ],
)
def test_is_valid_sequence_maps_text_mask_around_images(
    mask: tuple[list[int], list[int]],
) -> None:
    document_entry = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=3),
        imgs=[_img_index(img_id_in_doc=0, size=(1, 1), pos_in_text=1)],
        size=5,
        mask=mask,
    )

    assert _is_valid_sequence([document_entry])


def _concatenate(
    document_entries: list[DocumentIndexEntry],
    seq_len: int,
    stats_dict: dict[str, int],
    no_partial_sequence: bool = False,
    truncate_trailing_masked: bool = False,
) -> tuple[list[list[DocumentIndexEntry]], list[DocumentIndexEntry]]:
    all_sequences = list(
        concatenate_iterator(
            chunked_doc_iterator=iter(document_entries),
            seq_len=seq_len,
            no_partial_sequence=no_partial_sequence,
            stats_dict=stats_dict,
            yield_last_sequence=True,
            truncate_trailing_masked=truncate_trailing_masked,
        )
    )
    target_size = seq_len + 1
    if all_sequences and sum(entry.size for entry in all_sequences[-1]) < target_size:
        current_sequence = all_sequences.pop()
    else:
        current_sequence = []
    return all_sequences, current_sequence


def test_concatenate_no_partial_sequence() -> None:
    """Test that no_partial_sequence prevents breaking documents across sequences."""
    # Create documents with varying sizes
    documents: list[DocumentIndexEntry] = [
        # First doc: size=4, fits in first sequence
        DocumentIndexEntry(doc_id=0, text=TextIndexEntry(offset=0, size=4), imgs=[]),
        # Second doc: size=7 (text=2, img=5), doesn't fit in first sequence (4+7 > 10)
        # Should start new sequence
        DocumentIndexEntry(
            doc_id=1,
            text=TextIndexEntry(offset=0, size=2),
            imgs=[_img_index(img_id_in_doc=0, size=(2, 3), pos_in_text=0)],
        ),
        # Third doc: size=3, fits in second sequence (7+3 <= 10)
        DocumentIndexEntry(doc_id=2, text=TextIndexEntry(offset=0, size=3), imgs=[]),
        # Fourth doc: size=7, doesn't fit in second sequence (10 + 7 > 10)
        # Should start new sequence
        DocumentIndexEntry(
            doc_id=3,
            text=TextIndexEntry(offset=0, size=3),
            imgs=[_img_index(img_id_in_doc=0, size=(2, 2), pos_in_text=0)],
        ),
        # Fifth doc: not finished (remaining in current_sequence)
        DocumentIndexEntry(
            doc_id=4,
            text=TextIndexEntry(offset=0, size=3),
            imgs=[_img_index(img_id_in_doc=0, size=(2, 2), pos_in_text=0)],
        ),
    ]

    stats_dict = EpochStats().to_dict()
    finished_sequences, current_sequence = _concatenate(
        document_entries=documents,
        seq_len=10,
        stats_dict=stats_dict,
        no_partial_sequence=True,
    )

    # Expected: 3 finished sequences
    # Seq 1: doc_id=0 (size=4), then padding
    # Seq 2: doc_id=1 (size=7) + doc_id=2 (size=3) = 10
    # Seq 3: doc_id=3 (size=7), then padding
    # current_sequence: doc_id=4
    expected_num_total_tokens = [4, 10, 7]
    assert len(finished_sequences) == 3, (
        f"Expected 3 finished sequences, got {len(finished_sequences)}"
    )
    for seqs, num_tokens in zip(
        finished_sequences, expected_num_total_tokens, strict=True
    ):
        total_size = sum(x.size for x in seqs)
        assert total_size == num_tokens, (
            f"Expected {num_tokens} tokens, got {total_size}"
        )


def test_concatenate_no_partial_sequence_too_long_crash() -> None:
    """Test that no_partial_sequence crashes when a doc is too big."""
    # This document is too big (12 text + 18 img = 30 > 20 seq_len)
    # But since we're not filtering it in _build_doc here, it should crash
    document = DocumentIndexEntry(
        doc_id=0,
        text=TextIndexEntry(offset=0, size=12),
        imgs=[_img_index(img_id_in_doc=0, size=(9, 9), pos_in_text=6)],
    )

    stats_dict = EpochStats().to_dict()
    with pytest.raises(
        AssertionError, match="Document too big, should have been removed before!"
    ):
        _ = _concatenate(
            document_entries=[document],
            seq_len=20,
            stats_dict=stats_dict,
            no_partial_sequence=True,
        )


class DebugImageEncoderWithFilters(PretokenizedImageEncoder):
    """Image encoder that triggers various filter conditions based on image size and seed."""

    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len

    def uuid(self) -> str:
        return f"debug_filters_{self.seq_len}"

    def to_infos_dict(self) -> dict[str, Any]:
        return {"type": "debug_filters", "seq_len": self.seq_len}

    def apply_transforms_and_encode_image(
        self,
        image: Image.Image,
        metadata: TransformImageMetadata,
        seed: tuple[int, ...],
    ) -> tuple[TrainingImageEncoding, TransformImageMetadata]:
        n_toks, new_metadata = self.precompute_transforms(metadata=metadata, seed=seed)
        return TrainingImageEncoding(
            tokens=[IMG_ID] * n_toks, image=np.array(image)
        ), new_metadata

    def precompute_transforms(
        self, metadata: TransformImageMetadata, seed: tuple[int, ...]
    ) -> tuple[int, TransformImageMetadata]:
        # Use seed to deterministically trigger different filters:
        # - img_id_in_doc % 7 == 0: return size=0 -> n_removed_imgs_too_small
        # - Large images (width + height >= seq_len): -> n_removed_imgs_too_big
        # - Medium images at same position: -> n_removed_imgs_consecutive
        # - Normal images: kept
        img_id_in_doc = seed[-1] if seed else 0
        if img_id_in_doc % 7 == 0:
            return 0, metadata  # too_small
        n_toks = metadata.height + metadata.width
        return n_toks, metadata

    def get_transforms(self) -> Any:
        raise NotImplementedError("No image transforms in minimal codebase")


@pytest.mark.parametrize("add_images", [False, True])
@pytest.mark.parametrize("no_partial_sequence", [False, True])
def test_epoch_stats_consistency(
    tmp_path: Path, add_images: bool, no_partial_sequence: bool
) -> None:
    """Test that epoch stats are consistent across different configurations.

    Stats invariants are checked in _build_epoch_index. This test additionally verifies:
    1. Actual images seen when iterating matches epoch_stats.n_imgs
    2. All image removal filters are triggered when add_images=True
    3. When no_partial_sequence=True with images, some docs are skipped (n_docs_too_long)
    """
    rng = np.random.RandomState(seed=42)
    # Use small seq_len to trigger various filters
    # max_image_size=40 ensures some images have height+width >= seq_len (too_big)
    # while still allowing some docs to fit after removing too-big images
    seq_len = 60

    prefix_path, _, _, num_toks, _ = _build_datasource(
        tmp_path / "test_source",
        datasource_id=0,
        original_source="original_datasource",
        max_seq_length=40,
        rng=rng,
        n_docs=50,
        add_images=add_images,
        max_image_size=40,
    )

    datasource = Datasource(
        specs=DatasourceSpecs(
            prefix_path=str(prefix_path),
            seq_len=seq_len,
            seed=0,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_id=PAD_ID,
            percentage_data=1.0,
            no_shuffling=True,
            no_partial_sequence=no_partial_sequence,
        ),
        num_toks=10 * num_toks,
        cache_root_path=tmp_path / "cache_root",
        image_encoder_build_fn=(
            (lambda _: DebugImageEncoderWithFilters(seq_len=seq_len))
            if add_images
            else None
        ),
        image_transform_type="no_transform" if add_images else None,
        allow_epoch_index_building=True,
        selection_strategy=None,
    )

    epoch_stats = datasource._epoch_stats[0]
    assert isinstance(epoch_stats, EpochStats)

    if add_images:
        # Verify all image removal filters are triggered
        assert epoch_stats.n_removed_imgs_too_big > 0, (
            "Expected n_removed_imgs_too_big > 0"
        )
        assert epoch_stats.n_removed_imgs_too_small > 0, (
            "Expected n_removed_imgs_too_small > 0"
        )
        assert epoch_stats.n_removed_imgs_consecutive > 0, (
            "Expected n_removed_imgs_consecutive > 0"
        )

    # When no_partial_sequence=True with images, verify some docs were skipped
    if no_partial_sequence and add_images:
        assert epoch_stats.n_docs_too_long > 0, (
            "Expected n_docs_too_long > 0 with no_partial_sequence=True"
        )

    # Verify actual images seen matches stats
    n_images_seen = 0
    for seq_id in range(datasource._sequence_count_per_epoch[0]):
        sequence = datasource.get_sequence(seq_id)
        n_images_seen += len(sequence.images)

    assert n_images_seen == epoch_stats.n_imgs, (
        f"Actual images ({n_images_seen}) != epoch_stats.n_imgs ({epoch_stats.n_imgs})"
    )


def test_epoch_stats_can_be_reloaded() -> None:
    expected = EpochStats(
        n_text_toks=12,
        num_sequences=34,
        n_imgs_toks=13,
        n_pad_toks=5,
        n_imgs=9,
        n_removed_toks_end_epoch=16,
        n_removed_imgs_end_epoch=17,
        n_removed_imgs_too_big=2,
        n_removed_imgs_consecutive=4,
        n_total_imgs=9 + 1,
        n_total_text_toks=12 + 16,
        n_docs_too_long=0,
    )

    to_test = [
        # 250814
        {
            "n_text_toks": 12,
            "n_imgs_toks": 13,
            "n_pad_toks": 5,
            "n_imgs": 9,
            "n_removed_toks_end_epoch": 16,
            "n_removed_imgs_end_epoch": 17,
            "n_removed_imgs_to_big": 2,
            "n_removed_imgs_consecutive": 4,
            "n_total_imgs": 10,
            "n_total_text_toks": 28,
            "num_sequences": 34,
        },
        # 251009
        {
            "n_text_toks": 12,
            "n_imgs_toks": 13,
            "n_pad_toks": 5,
            "n_imgs": 9,
            "n_removed_toks_end_epoch": 16,
            "n_removed_imgs_end_epoch": 17,
            "n_removed_imgs_too_big": 2,
            "n_removed_imgs_too_small": 0,
            "n_removed_imgs_consecutive": 4,
            "n_total_imgs": 10,
            "n_total_text_toks": 28,
            "num_sequences": 34,
        },
        # 260113
        {
            "n_imgs": 9,
            "n_imgs_toks": 13,
            "n_pad_toks": 5,
            "n_removed_imgs_consecutive": 4,
            "n_removed_imgs_end_epoch": 17,
            "n_removed_imgs_invalid_sequences": 0,
            "n_removed_imgs_too_big": 2,
            "n_removed_imgs_too_small": 0,
            "n_removed_invalid_sequences": 0,
            "n_removed_toks_end_epoch": 16,
            "n_removed_toks_invalid_sequences": 0,
            "n_text_toks": 12,
            "n_total_imgs": 10,
            "n_total_text_toks": 28,
            "num_sequences": 34,
        },
        # 260218 - added n_docs_too_long for no_partial_sequence support
        {
            "n_imgs": 9,
            "n_imgs_toks": 13,
            "n_pad_toks": 5,
            "n_removed_imgs_consecutive": 4,
            "n_removed_imgs_end_epoch": 17,
            "n_removed_imgs_invalid_sequences": 0,
            "n_removed_imgs_too_big": 2,
            "n_removed_imgs_too_small": 0,
            "n_removed_invalid_sequences": 0,
            "n_removed_toks_end_epoch": 16,
            "n_removed_toks_invalid_sequences": 0,
            "n_text_toks": 12,
            "n_total_imgs": 10,
            "n_total_text_toks": 28,
            "num_sequences": 34,
            "n_docs_too_long": 0,
        },
        # added n_removed_imgs_missing_bucket
        {
            "n_docs_too_long": 0,
            "n_imgs": 9,
            "n_imgs_toks": 13,
            "n_pad_toks": 5,
            "n_removed_imgs_consecutive": 4,
            "n_removed_imgs_end_epoch": 17,
            "n_removed_imgs_invalid_sequences": 0,
            "n_removed_imgs_missing_bucket": 0,
            "n_removed_imgs_too_big": 2,
            "n_removed_imgs_too_small": 0,
            "n_removed_invalid_sequences": 0,
            "n_removed_toks_end_epoch": 16,
            "n_removed_toks_invalid_sequences": 0,
            "n_text_toks": 12,
            "n_total_imgs": 10,
            "n_total_text_toks": 28,
            "num_sequences": 34,
        },
        # added truncate_trailing_masked counters
        {
            "n_docs_too_long": 0,
            "n_docs_truncated_trailing_masked": 0,
            "n_imgs": 9,
            "n_imgs_toks": 13,
            "n_pad_toks": 5,
            "n_removed_imgs_consecutive": 4,
            "n_removed_imgs_end_epoch": 17,
            "n_removed_imgs_invalid_sequences": 0,
            "n_removed_imgs_missing_bucket": 0,
            "n_removed_imgs_too_big": 2,
            "n_removed_imgs_too_small": 0,
            "n_removed_imgs_trailing_masked": 0,
            "n_removed_invalid_sequences": 0,
            "n_removed_toks_end_epoch": 16,
            "n_removed_toks_invalid_sequences": 0,
            "n_removed_toks_trailing_masked": 0,
            "n_text_toks": 12,
            "n_total_imgs": 10,
            "n_total_text_toks": 28,
            "num_sequences": 34,
        },
    ]
    to_test.append(
        {
            **to_test[-1],
            "n_removed_docs_online_quarantine": 0,
            "n_removed_toks_online_quarantine": 0,
            "n_removed_imgs_online_quarantine": 0,
        }
    )

    for i, old in enumerate(to_test):
        print(f"testing entry {i}")
        received = EpochStats.model_validate(old)
        assert received == expected, (
            i,
            received,
            expected,
            "Please modify EpochStats to be backward compatible",
        )

        # check the latest one
    latest = to_test[-1]
    # check that current serialized dict is the current one.
    # if not you need to store the serialization of the new one under
    serialized_now = expected.model_dump(mode="json")
    print("current serialisation:")
    pprint(serialized_now)
    assert latest == serialized_now, (
        "Did you change EpochStats? If yes you need to update the list of to_test "
        "by ADDING current serialisation (printed above) to `to_test`. DO NOT MODIFY "
        "EXISTING SERIALIZATIONS SINCE THE GOAL IS TO BE SURE THAT WE CAN RELOAD THEM"
    )


def random_string(rng: np.random.Generator, length: int) -> str:
    return "".join(
        rng.choice(
            list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=length
        )
    )


def _random_image_entry(rng: np.random.Generator) -> ImageIndexEntry:
    width = int(rng.integers(0, 10**5))
    height = int(rng.integers(0, 10**5))
    return ImageIndexEntry(
        img_id_in_doc=int(rng.integers(0, 10**6)),
        tar_offset=_TarOffset(
            offset=int(rng.integers(0, 10**6)),
            size=int(rng.integers(0, 10**6)),
            tar_path=random_string(rng, 10),
        ),
        pos_in_text=int(rng.integers(0, 10**6)),
        metadata=TransformImageMetadata(
            width=width,
            height=height,
        ),
    )


def test_metadata_reader(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=42)

    file_path = tmp_path / "chunks.metadata"

    original_images: list[list[ImageIndexEntry]] = [
        [_random_image_entry(rng) for _ in range(rng.integers(1, 10))]
        for _ in range(10)
    ]
    dump_metadata(original_images, file_path)
    metadata_reader = _MetadataReader(file_path)

    for original, from_reader in zip(original_images, metadata_reader, strict=True):
        assert len(original) == len(from_reader)
        for original_img, from_reader_img in zip(original, from_reader, strict=True):
            assert original_img == from_reader_img


# DISABLED: test_image_serialization_bw_compatibility
# Requires: ImageBoundingBox, l2_reconstruction_score, _img_bbox_for_trim fields

# DISABLED: test_image_serialization
# Requires: ImageBoundingBox, L2_SCORE_TEST_KEY, full TransformImageMetadata fields


def test_epoch_index_path_hash_stability(tmp_path: Path) -> None:
    """Detect future hash drift in the specs hash used for epoch index cache paths.

    If this test fails, the specs hash has changed and existing caches are unreachable.
    Do NOT simply update the expected hash here. Instead, add backward-compat support
    so both the new and old paths are tried (e.g. via a legacy-hash candidate in
    _get_epoch_index_path or _resolve_epoch_index_path), then update this test.
    """
    FAILURE_MSG = (
        "Specs hash changed — do NOT just update this value. "
        "Existing caches built with the old hash will be unreachable. "
        "First add backward-compat support, then update this test."
    )

    specs_base = DatasourceSpecs(
        prefix_path="/some/fixed/path/data.bin",
        seed=42,
        seq_len=4096,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        percentage_data=1.0,
        no_shuffling=False,
        no_partial_sequence=False,
    )

    # no_partial_sequence=False: field excluded from hash dict for backward compat
    path_false = _get_epoch_index_path(
        epoch_indexes_root_dir=tmp_path,
        specs=specs_base,
        epoch_id=0,
        selection_strategy=None,
        image_encoder=None,
        has_loss_mask=False,
    )
    assert (
        "data__9cde6a45ebd8f9a63b86f1824feee7a2__epoch000_seqlen4096_seed42__v2"
        == path_false.name
    ), f"{FAILURE_MSG} got: {path_false.name}"

    # no_partial_sequence=True: field included in hash -> different hash
    specs_true = dataclasses.replace(specs_base, no_partial_sequence=True)
    path_true = _get_epoch_index_path(
        epoch_indexes_root_dir=tmp_path,
        specs=specs_true,
        epoch_id=0,
        selection_strategy=None,
        image_encoder=None,
        has_loss_mask=False,
    )
    assert (
        "data__de35a4d52ab0d1be8a1630a1109ac512__epoch000_seqlen4096_seed42__v2"
        == path_true.name
    ), f"{FAILURE_MSG} got: {path_true.name}"
    assert path_false != path_true


def test_epoch_index_path_uses_shallow_source_hash_and_legacy_fallback(tmp_path: Path) -> None:
    specs = DatasourceSpecs(
        prefix_path="/some/fixed/path/data.bin",
        seed=42,
        seq_len=4096,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        percentage_data=1.0,
        no_shuffling=False,
        no_partial_sequence=False,
    )

    hashed_path = _get_epoch_index_path(
        epoch_indexes_root_dir=tmp_path,
        specs=specs,
        epoch_id=0,
        selection_strategy=None,
        image_encoder=None,
        has_loss_mask=False,
    )
    assert hashed_path.parent == (
        tmp_path / "3a793425cc27420b789fd0359fd4f4d1d9a2ec7f7bcc9d4342de5f624903b4cc"
    )

    legacy_path = tmp_path / "some/fixed/path" / hashed_path.name
    legacy_path.mkdir(parents=True)
    assert (
        _get_epoch_index_path(
            epoch_indexes_root_dir=tmp_path,
            specs=specs,
            epoch_id=0,
            selection_strategy=None,
            image_encoder=None,
            has_loss_mask=False,
        )
        == legacy_path
    )

    hashed_path.mkdir(parents=True)
    assert (
        _get_epoch_index_path(
            epoch_indexes_root_dir=tmp_path,
            specs=specs,
            epoch_id=0,
            selection_strategy=None,
            image_encoder=None,
            has_loss_mask=False,
        )
        == hashed_path
    )


def test_epoch_index_path_truncate_trailing_masked_hash(tmp_path: Path) -> None:
    specs_off = DatasourceSpecs(
        prefix_path="/some/fixed/path/data.bin",
        seed=42,
        seq_len=4096,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        percentage_data=1.0,
        no_shuffling=False,
        no_partial_sequence=False,
        truncate_trailing_masked=False,
    )
    specs_on = dataclasses.replace(specs_off, truncate_trailing_masked=True)

    def path(specs: DatasourceSpecs, has_loss_mask: bool) -> str:
        return _get_epoch_index_path(
            epoch_indexes_root_dir=tmp_path,
            specs=specs,
            epoch_id=0,
            selection_strategy=None,
            image_encoder=None,
            has_loss_mask=has_loss_mask,
        ).name

    baseline = "data__9cde6a45ebd8f9a63b86f1824feee7a2__epoch000_seqlen4096_seed42__v2"
    assert path(specs_off, has_loss_mask=False) == baseline
    assert path(specs_off, has_loss_mask=True) == baseline
    assert path(specs_on, has_loss_mask=False) == baseline
    assert path(specs_on, has_loss_mask=True) == (
        baseline + "__truncate_trailing_masked"
    )


def test_epoch_index_path_img_uuid_stability(tmp_path: Path) -> None:
    """Pin the full path produced by _get_epoch_index_path when an image encoder is used.

    Verifies that the encoder UUID is embedded in the path name.
    Any change to _get_epoch_index_path that would silently shift a cache path will be
    caught here.
    """
    FAILURE_MSG = (
        "Epoch index path changed — do NOT just update this value. "
        "Existing caches at the old path will be unreachable."
    )

    class _SimpleEncoder(PretokenizedImageEncoder):
        def uuid(self) -> str:
            return "myuuid123"

        def to_infos_dict(self) -> dict[str, Any]:
            return {}

        def apply_transforms_and_encode_image(
            self, image: Image.Image, metadata: Any, seed: Any
        ) -> Any:  # type: ignore[override]
            raise NotImplementedError

        def precompute_transforms(self, metadata: Any, seed: Any) -> Any:  # type: ignore[override]
            raise NotImplementedError

        def get_transforms(self) -> Any:  # type: ignore[override]
            raise NotImplementedError

    specs = DatasourceSpecs(
        prefix_path="/some/fixed/path/data.bin",
        seed=42,
        seq_len=4096,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        percentage_data=1.0,
        no_shuffling=False,
        no_partial_sequence=False,
    )

    path = _get_epoch_index_path(
        epoch_indexes_root_dir=tmp_path,
        specs=specs,
        epoch_id=0,
        selection_strategy=None,
        image_encoder=_SimpleEncoder(),
        has_loss_mask=False,
    )

    assert (
        path.name
        == "data__9cde6a45ebd8f9a63b86f1824feee7a2__epoch000_seqlen4096_seed42__v2__img_myuuid123"
    ), f"{FAILURE_MSG} got: {path.name}"


def test_scatter_gather_shuffles_sequences(tmp_path: Path) -> None:
    n_sequences = 1000
    original_sequences = [
        [
            DocumentIndexEntry(
                doc_id=i,
                text=TextIndexEntry(offset=0, size=10),
                imgs=[],
            )
        ]
        for i in range(n_sequences)
    ]

    scatter_dir = tmp_path / "scatter"
    scatter_dir.mkdir()
    scatter_rng = np.random.default_rng(42)
    n_buckets = 8

    _scatter_epoch_sequences(
        iter(original_sequences),
        scatter_dir,
        scatter_rng,
        n_buckets,
    )

    gathered_batches = list(
        _gather_epoch_sequences_batched(scatter_dir, scatter_rng, n_buckets, epoch_id=0)
    )
    gathered_serialized = [record for batch in gathered_batches for record in batch]
    gathered_sequences = [
        DocumentIndexEntry.list_from_bytes(data.rstrip(b"\n"))
        for data in gathered_serialized
    ]

    assert len(gathered_sequences) == n_sequences
    assert {seq[0].doc_id for seq in gathered_sequences} == set(range(n_sequences))
    assert [seq[0].doc_id for seq in gathered_sequences] != list(range(n_sequences))


_DUMMY_SPECS = DatasourceSpecs(
    prefix_path="/tmp/source/chunks_00000-00001",
    seed=0,
    seq_len=8,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    pad_id=PAD_ID,
    percentage_data=1.0,
    no_shuffling=False,
    no_partial_sequence=False,
)


@pytest.mark.parametrize(
    "exc",
    [
        MissingEpochIndex(path=Path("/tmp/epoch_index"), specs=_DUMMY_SPECS),
        MissingEpochIndex(
            path=Path("/tmp/epoch_index"), specs=_DUMMY_SPECS, message="custom msg"
        ),
        CorruptedEpochIndex(path=Path("/tmp/epoch_index"), specs=_DUMMY_SPECS),
    ],
    ids=["missing-default-msg", "missing-custom-msg", "corrupted"],
)
def test_epoch_index_exceptions_pickle_roundtrip(
    exc: MissingEpochIndex | CorruptedEpochIndex,
) -> None:
    import pickle

    restored = pickle.loads(pickle.dumps(exc))
    assert type(restored) is type(exc)
    assert restored.path == exc.path
    assert restored.specs == exc.specs
    assert str(restored) == str(exc)


if __name__ == "__main__":
    pytest.main([__file__, "-k", "test_image_serialization"])
