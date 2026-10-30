# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""External VLM dataloader interface.

Providers selected by ``--vlm-dataloader`` return tensor-only batches consumed by
``pretrain_vlm.get_batch``. For packed VLM batches, ``micro_batch_size`` is
currently required to be 1.

Expected keys and dtypes:

.. code-block:: python

    {
        "tokens": torch.int64,
        "position_ids": torch.int64,
        "labels": torch.int64,
        "loss_mask": torch.float32,
        "cu_seqlens": torch.int32,
        "max_seqlen": torch.int32,
        "num_image_tiles": torch.int64,
        "image": torch.float32,
        "imgs_sizes": torch.int32,
        "vision_cu_seqlens": torch.int32,
        "vision_max_seqlen": torch.int32,
    }

``image`` is a tensor, not a Python list. Dynamic-resolution RADIO batches use
``[1, total_patches, 3 * patch_dim * patch_dim]``; fixed-resolution batches use
``[num_images, 3, H, W]`` with ``[1, 1]`` sentinel vision metadata.

``cu_seqlens`` describes the post-LLaVA decoder sequence, after image
placeholder token(s) expand into image embeddings. ``vision_cu_seqlens``
describes the image patch sequence going into the vision encoder.

For ``--dataloader-seq-length 512``, the mock post-c-radio-omni probe showed:

.. code-block:: text

    tokens:              [1, 513]       # image placeholder + 512 text tokens
    image:               [1, 1024, 768] # 512x512, patch_dim=16
    cu_seqlens:          [[0, 511, 768]]
    vision_cu_seqlens:   [[0, 1024]]
    num_image_tiles:     [256]          # post-pixel-shuffle image token count
"""

import os

import torch

from megatron.core import parallel_state
from megatron.core.models.multimodal.llava_model import DEFAULT_IMAGE_TOKEN_INDEX
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.training import get_args, get_tokenizer
from megatron.training.checkpointing import get_checkpoint_name


def add_vlm_data_args(parser):
    group = parser.add_argument_group(title="vision language model data arguments")
    group.add_argument(
        "--vlm-dataloader",
        type=str,
        choices=("mock",),
        default="mock",
        help="External VLM dataloader backend.",
    )
    return parser


def build_vlm_dataset_provider(
    args,
    fallback_provider,
    *,
    vision_model_type,
    class_token_len,
    pixel_shuffle,
    dynamic_resolution,
    dynamic_resolution_max_patches,
    spatial_merge_size=1,
):
    if args.dataloader_type != "external":
        fallback_provider.is_distributed = True
        return fallback_provider
    if args.vlm_dataloader == "mock":
        provider = _build_mock_provider(
            vision_model_type=vision_model_type,
            class_token_len=class_token_len,
            pixel_shuffle=pixel_shuffle,
            spatial_merge_size=spatial_merge_size,
            dynamic_resolution=dynamic_resolution,
            dynamic_resolution_max_patches=dynamic_resolution_max_patches,
        )
    else:
        raise ValueError(f"Unknown --vlm-dataloader={args.vlm_dataloader!r}")
    provider.is_distributed = True
    return provider


def _is_vlm_dataloader_rank():
    if parallel_state.get_tensor_model_parallel_rank() != 0:
        return False
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pp_size == 1:
        return True
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    return pp_rank in (0, pp_size - 1)


class _ExternalDataloader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(dataloader) if dataloader is not None else iter([])

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._dataloader)
            return next(self._iter)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)

    def save_state(self):
        if hasattr(self._dataloader, "save_state_rank"):
            return self._dataloader.save_state_rank()
        if hasattr(self._dataloader, "save_state"):
            return self._dataloader.save_state()
        return {}


def _maybe_restore_external_state(loader, args):
    if args.load is None or not getattr(args, "dataloader_save", None):
        return
    dp_rank = parallel_state.get_data_parallel_rank()
    data_save_name = get_checkpoint_name(
        args.dataloader_save,
        args.iteration,
        pipeline_rank=0,
        basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
    )
    if os.path.exists(data_save_name):
        state = torch.load(data_save_name, map_location="cpu", weights_only=False)
        if hasattr(loader, "restore_state_rank"):
            loader.restore_state_rank(state["dataloader_state_dict"])
        elif hasattr(loader, "set_state"):
            loader.set_state(state["dataloader_state_dict"])


def _mock_vocab_size(args, tokenizer):
    for value in (
        getattr(args, "padded_vocab_size", None),
        getattr(tokenizer, "vocab_size", None),
        getattr(getattr(tokenizer, "_tokenizer", None), "vocab_size", None),
    ):
        if value is not None:
            return max(2, int(value))
    return 32000


def _mock_patch_grid(args, dynamic_resolution_max_patches, pixel_shuffle, spatial_merge_size=1):
    patch_rows = max(1, args.img_h // args.patch_dim)
    patch_cols = max(1, args.img_w // args.patch_dim)
    spatial_merge_size = int(spatial_merge_size)
    if spatial_merge_size < 1:
        raise ValueError(f"spatial_merge_size must be >= 1, got {spatial_merge_size}")
    if pixel_shuffle and spatial_merge_size > 1:
        raise ValueError("Mock VLM data does not support pixel_shuffle with spatial merging")
    max_patches = int(dynamic_resolution_max_patches or 0)
    if max_patches > 0 and patch_rows * patch_cols > max_patches:
        patch_rows = max(1, min(patch_rows, int(max_patches**0.5)))
        patch_cols = max(1, min(patch_cols, max_patches // patch_rows))

    if pixel_shuffle:
        while patch_rows * patch_cols % 4 != 0 and patch_cols > 1:
            patch_cols -= 1
        while patch_rows * patch_cols % 4 != 0 and patch_rows > 1:
            patch_rows -= 1
        if patch_rows * patch_cols % 4 != 0:
            raise ValueError("Mock pixel-shuffle VLM images require at least 4 patches")

    if spatial_merge_size > 1:
        patch_rows -= patch_rows % spatial_merge_size
        patch_cols -= patch_cols % spatial_merge_size
        if patch_rows < spatial_merge_size or patch_cols < spatial_merge_size:
            raise ValueError(
                "Mock spatial-merge VLM images require at least "
                f"{spatial_merge_size}x{spatial_merge_size} patches"
            )

    return patch_rows, patch_cols


class _MockVLMDataLoader:
    def __init__(
        self,
        *,
        length,
        seq_length,
        vocab_size,
        eod_token_id,
        image_token_count,
        img_h,
        img_w,
        patch_dim,
        dynamic_resolution,
        patch_rows,
        patch_cols,
        seed_offset,
    ):
        if seq_length < 4:
            raise ValueError("--dataloader-seq-length must be at least 4 for mock VLM data")
        self._length = max(1, int(length))
        self._seq_length = int(seq_length)
        self._vocab_size = int(vocab_size)
        self._eod_token_id = int(eod_token_id) if eod_token_id is not None else 0
        self._image_token_count = int(image_token_count)
        self._img_h = int(img_h)
        self._img_w = int(img_w)
        self._patch_dim = int(patch_dim)
        self._dynamic_resolution = bool(dynamic_resolution)
        self._patch_rows = int(patch_rows)
        self._patch_cols = int(patch_cols)
        self._seed_offset = int(seed_offset)
        self._iteration = 0
        self._split = max(2, self._seq_length // 2)
        self._num_image_tiles = self._image_token_count if self._dynamic_resolution else 1

    def __iter__(self):
        return self

    def __len__(self):
        return self._length

    def __next__(self):
        sample = self._make_sample(self._iteration)
        self._iteration += 1
        return sample

    def save_state_rank(self):
        return {"iteration": self._iteration}

    def save_state(self):
        return self.save_state_rank()

    def restore_state_rank(self, state):
        self.set_state(state)

    def set_state(self, state):
        self._iteration = int(state.get("iteration", 0))

    def _make_sample(self, iteration):
        tokens = self._tokens(iteration)
        labels = torch.empty_like(tokens)
        labels[:, :-1] = tokens[:, 1:]
        labels[:, -1] = self._eod_token_id
        labels[:, self._split - 1] = self._eod_token_id

        position_ids = torch.cat(
            [
                torch.arange(self._split, dtype=torch.long),
                torch.arange(self._seq_length - self._split, dtype=torch.long),
            ]
        ).unsqueeze(0)

        loss_mask = torch.ones(1, self._seq_length, dtype=torch.float32)
        loss_mask[tokens == DEFAULT_IMAGE_TOKEN_INDEX] = 0
        loss_mask[:, self._split - 1] = 0

        first_packed_len = self._split - 1 + self._image_token_count
        second_packed_len = self._seq_length - self._split
        final_seq_length = first_packed_len + second_packed_len

        sample = {
            "tokens": tokens,
            "position_ids": position_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "cu_seqlens": torch.tensor(
                [[0, first_packed_len, final_seq_length]], dtype=torch.int32
            ),
            "max_seqlen": torch.tensor(
                [[max(first_packed_len, second_packed_len)]], dtype=torch.int32
            ),
            "num_image_tiles": torch.tensor([self._num_image_tiles], dtype=torch.long),
        }
        sample.update(self._image_sample(iteration))
        return sample

    def _tokens(self, iteration):
        offset = self._seed_offset + iteration * self._seq_length
        tokens = (
            (torch.arange(self._seq_length, dtype=torch.long) + offset) % (self._vocab_size - 1)
        ) + 1
        tokens[0] = DEFAULT_IMAGE_TOKEN_INDEX
        return tokens.unsqueeze(0)

    def _image_sample(self, iteration):
        if self._dynamic_resolution:
            patch_count = self._patch_rows * self._patch_cols
            patch_width = 3 * self._patch_dim * self._patch_dim
            image = torch.arange(patch_count * patch_width, dtype=torch.float32)
            image = ((image + iteration) % 251).view(1, patch_count, patch_width) / 250.0
            return {
                "image": image,
                "imgs_sizes": torch.tensor(
                    [[self._patch_rows * self._patch_dim, self._patch_cols * self._patch_dim]],
                    dtype=torch.int32,
                ),
                "vision_cu_seqlens": torch.tensor([[0, patch_count]], dtype=torch.int32),
                "vision_max_seqlen": torch.tensor([patch_count], dtype=torch.int32),
            }

        image = torch.arange(3 * self._img_h * self._img_w, dtype=torch.float32)
        image = ((image + iteration) % 251).view(1, 3, self._img_h, self._img_w) / 250.0
        return {
            "image": image,
            "imgs_sizes": torch.ones(1, 1, dtype=torch.int32),
            "vision_cu_seqlens": torch.ones(1, 1, dtype=torch.int32),
            "vision_max_seqlen": torch.ones(1, dtype=torch.int32),
        }


def _build_mock_provider(
    *,
    vision_model_type,
    class_token_len,
    pixel_shuffle,
    dynamic_resolution,
    dynamic_resolution_max_patches,
    spatial_merge_size=1,
):
    def provider(train_val_test_num_samples):
        args = get_args()
        if not _is_vlm_dataloader_rank():
            return None, None, None
        if not args.mock_data:
            raise ValueError("--vlm-dataloader mock requires --mock-data")
        if args.micro_batch_size != 1:
            raise ValueError("--vlm-dataloader mock currently requires --micro-batch-size 1")
        if dynamic_resolution and not args.disable_vision_class_token:
            raise ValueError(
                "--vlm-dataloader mock dynamic-resolution batches require "
                "--disable-vision-class-token"
            )

        if args.dataloader_seq_length is None:
            raise ValueError("--vlm-dataloader mock requires --dataloader-seq-length")
        seq_length = args.dataloader_seq_length + 1
        tokenizer = get_tokenizer()
        patch_rows, patch_cols = _mock_patch_grid(
            args, dynamic_resolution_max_patches, pixel_shuffle, spatial_merge_size
        )
        if dynamic_resolution:
            patch_count = patch_rows * patch_cols
            if pixel_shuffle:
                image_token_count = patch_count // 4
            else:
                image_token_count = patch_count // (spatial_merge_size * spatial_merge_size)
        else:
            image_token_count = get_num_image_embeddings(
                args.img_h,
                args.img_w,
                args.patch_dim,
                vision_model_type,
                args.disable_vision_class_token,
                class_token_len,
                pixel_shuffle,
                use_tile_tags=False,
                spatial_merge_size=spatial_merge_size,
            )

        def build_loader(num_samples, seed_offset, restore_state=False):
            loader = _MockVLMDataLoader(
                length=num_samples or 1,
                seq_length=seq_length,
                vocab_size=_mock_vocab_size(args, tokenizer),
                eod_token_id=getattr(tokenizer, "eod", None),
                image_token_count=image_token_count,
                img_h=args.img_h,
                img_w=args.img_w,
                patch_dim=args.patch_dim,
                dynamic_resolution=dynamic_resolution,
                patch_rows=patch_rows,
                patch_cols=patch_cols,
                seed_offset=seed_offset,
            )
            if restore_state:
                _maybe_restore_external_state(loader, args)
            return _ExternalDataloader(loader)

        dp_rank = parallel_state.get_data_parallel_rank()
        train_loader = build_loader(
            train_val_test_num_samples[0], dp_rank * 1000003, restore_state=True
        )
        valid_loaders = None
        if args.full_validation or getattr(args, "eval_iters", 0) > 0:
            valid_loaders = [
                build_loader(train_val_test_num_samples[1], dp_rank * 1000003 + 1000000000)
            ]

        return train_loader, valid_loaders, None

    return provider
