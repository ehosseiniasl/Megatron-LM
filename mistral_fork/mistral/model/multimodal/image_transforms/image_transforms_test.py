"""Tests for image transform internal consistency (L1 from testing RFC).

Verifies that:
- precompute_transforms and apply_transforms produce matching output dimensions
- token counts match the image_size_to_num_tokens formula
- the existing transform retains its L2 behavior
- the no-L2-resize mode preserves resizing and normalization without requiring a score
- precompute/apply are deterministic across calls with the same seed

These tests run locally (need torch + cv2 + PIL, no GPU).
"""

import dataclasses

import numpy as np
import pytest
from PIL import Image

from mistral_fork.mistral.data.pretokenized.components.image_transforms import (
    ImageTransformType,
    SpecialImageIDs,
    _NoTransform,
    image_transforms_factory,
)
from mistral_fork.mistral.data.pretokenized.image_transform_metadata import TransformImageMetadata
from mistral_fork.mistral.data.pretokenized.simple_image_encoder import SimpleImageEncoder
from mistral_fork.mistral.model.multimodal.image_transforms.image_transforms import (
    CrawlDefaultNoHighRes,
    GroundingDefault,
    SometimesHighResDefault,
)
from mistral_fork.mistral.model.multimodal.image_transforms.utils import (
    CLIP_DATASET_MEAN,
    CLIP_DATASET_STD,
)
from mistral_fork.mistral_common.tokens.tokenizers.image import image_size_to_num_tokens

SPECIAL_IDS = SpecialImageIDs(img=10, img_break=11, img_end=12)

# ---------------------------------------------------------------------------
# Test metadata fixtures
# ---------------------------------------------------------------------------

CRAWL_METADATA_LOW_L2 = TransformImageMetadata(
    width=1200,
    height=900,
    l2_reconstruction_score={"image_min_size=224,l2_resize_factor=2": 0.005},
)

CRAWL_METADATA_HIGH_L2 = TransformImageMetadata(
    width=1200,
    height=900,
    l2_reconstruction_score={"image_min_size=224,l2_resize_factor=2": 0.02},
)


def _make_test_image(w: int, h: int, pattern: str = "gradient") -> Image.Image:
    """Create synthetic test images."""
    if pattern == "gradient":
        arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        return Image.fromarray(np.stack([arr] * 3, axis=-1))
    elif pattern == "checkerboard":
        arr = np.zeros((h, w), dtype=np.uint8)
        arr[::2, ::2] = 255
        arr[1::2, 1::2] = 255
        return Image.fromarray(np.stack([arr] * 3, axis=-1))
    elif pattern == "white_with_content":
        arr = np.full((h, w, 3), 255, dtype=np.uint8)
        ch, cw = h // 4, w // 4
        arr[ch : 3 * ch, cw : 3 * cw] = 30
        return Image.fromarray(arr)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def _expected_n_tokens(
    width: int,
    height: int,
    max_image_size: int = 1036,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
) -> int:
    """Compute expected token count from dimensions."""
    w_tokens, h_tokens = image_size_to_num_tokens(
        size=(width, height),
        max_image_size=max_image_size,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
    )
    return (w_tokens + 1) * h_tokens


# ---------------------------------------------------------------------------
# NoTransform
# ---------------------------------------------------------------------------


class TestNoTransform:
    def test_precompute_returns_correct_tokens(self) -> None:
        meta = TransformImageMetadata(width=512, height=768)
        transform = _NoTransform(special_ids=SPECIAL_IDS)
        n_toks, out_meta = transform.precompute_transforms(meta, seed=(42,))
        assert n_toks == _expected_n_tokens(512, 768, max_image_size=1024)
        assert out_meta.width == 512
        assert out_meta.height == 768

    def test_apply_returns_unchanged_image(self) -> None:
        meta = TransformImageMetadata(width=64, height=48)
        image = _make_test_image(64, 48)
        transform = _NoTransform(special_ids=SPECIAL_IDS)
        pixels, out_meta = transform.apply_transforms(image, meta, seed=(42,))
        assert pixels.shape == (48, 64, 3)
        assert out_meta.width == 64
        assert out_meta.height == 48


# ---------------------------------------------------------------------------
# L2-enabled crawl and grounding modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform_class", [CrawlDefaultNoHighRes, GroundingDefault])
class TestL2ResizeModes:
    @pytest.mark.parametrize("seed", [(0,), (42,), (123, 456)])
    @pytest.mark.parametrize("meta", [CRAWL_METADATA_LOW_L2, CRAWL_METADATA_HIGH_L2])
    def test_default_matches_original_probability(
        self,
        transform_class: type[CrawlDefaultNoHighRes],
        seed: tuple[int, ...],
        meta: TransformImageMetadata,
    ) -> None:
        """The default remains exactly equivalent to the original 0.9 policy."""
        default_transform = transform_class()
        explicit_transform = transform_class(
            apply_dynamic_proportion=transform_class.APPLY_DYNAMIC_PROPORTION
        )

        default_tokens, default_meta = default_transform.precompute_transforms(meta, seed=seed)
        explicit_tokens, explicit_meta = explicit_transform.precompute_transforms(meta, seed=seed)

        assert default_tokens == explicit_tokens
        assert dataclasses.asdict(default_meta) == dataclasses.asdict(explicit_meta)

    @pytest.mark.parametrize("seed", [(0,), (42,), (123, 456)])
    def test_precompute_apply_dimension_agreement(
        self, transform_class: type[CrawlDefaultNoHighRes], seed: tuple[int, ...]
    ) -> None:
        """precompute and apply must produce the same output dimensions."""
        transform = transform_class()
        meta = CRAWL_METADATA_HIGH_L2

        n_toks, pre_meta = transform.precompute_transforms(meta, seed=seed)
        image = _make_test_image(meta.width, meta.height)
        pixels, apply_meta = transform.apply_transforms(image, meta, seed=seed)

        assert pre_meta.width == apply_meta.width
        assert pre_meta.height == apply_meta.height
        assert pixels.shape[1] == apply_meta.height
        assert pixels.shape[2] == apply_meta.width

    @pytest.mark.parametrize("seed", [(0,), (42,), (123, 456)])
    def test_token_count_matches_dimensions(
        self, transform_class: type[CrawlDefaultNoHighRes], seed: tuple[int, ...]
    ) -> None:
        """Token count must match the image_size_to_num_tokens formula."""
        transform = transform_class()
        meta = CRAWL_METADATA_HIGH_L2

        n_toks, out_meta = transform.precompute_transforms(meta, seed=seed)
        expected = _expected_n_tokens(out_meta.width, out_meta.height)
        assert n_toks == expected

    def test_low_l2_downsamples(
        self, transform_class: type[CrawlDefaultNoHighRes]
    ) -> None:
        """Images with low L2 score should be downsampled."""
        transform = transform_class()
        _, meta_low = transform.precompute_transforms(CRAWL_METADATA_LOW_L2, seed=(0,))
        _, meta_high = transform.precompute_transforms(
            CRAWL_METADATA_HIGH_L2, seed=(0,)
        )
        # Low L2 should produce smaller or equal dimensions
        assert meta_low.width * meta_low.height <= meta_high.width * meta_high.height

    def test_missing_l2_raises(
        self, transform_class: type[CrawlDefaultNoHighRes]
    ) -> None:
        """Missing L2 reconstruction score should raise ValueError."""
        transform = transform_class()
        meta = TransformImageMetadata(width=1200, height=900)
        # Need a seed where the dynamic proportion check passes (p < 0.9)
        # Try multiple seeds to find one where the error triggers
        raised = False
        for seed_val in range(20):
            try:
                transform.precompute_transforms(meta, seed=(seed_val,))
            except ValueError as e:
                assert "l2_reconstruction_score" in str(e)
                raised = True
                break
        assert raised, "Expected ValueError for missing L2 score"

    def test_deterministic(self, transform_class: type[CrawlDefaultNoHighRes]) -> None:
        """Same seed produces same output."""
        transform = transform_class()
        seed = (42, 7)
        n1, m1 = transform.precompute_transforms(CRAWL_METADATA_HIGH_L2, seed=seed)
        n2, m2 = transform.precompute_transforms(CRAWL_METADATA_HIGH_L2, seed=seed)
        assert n1 == n2
        assert m1.width == m2.width
        assert m1.height == m2.height

    @pytest.mark.parametrize(
        ("width", "height"),
        [(64, 64), (512, 768), (2000, 3000), (100, 2000)],
    )
    def test_various_dimensions(
        self, transform_class: type[CrawlDefaultNoHighRes], width: int, height: int
    ) -> None:
        """Test with various image dimensions including extreme aspect ratios."""
        transform = transform_class()
        meta = TransformImageMetadata(
            width=width,
            height=height,
            l2_reconstruction_score={"image_min_size=224,l2_resize_factor=2": 0.02},
        )
        seed = (42,)
        n_toks, pre_meta = transform.precompute_transforms(meta, seed=seed)
        assert n_toks > 0
        assert pre_meta.width > 0
        assert pre_meta.height > 0

        image = _make_test_image(width, height)
        pixels, apply_meta = transform.apply_transforms(image, meta, seed=seed)
        assert pre_meta.width == apply_meta.width
        assert pre_meta.height == apply_meta.height


# ---------------------------------------------------------------------------
# No-L2 resize modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform_class", [CrawlDefaultNoHighRes, GroundingDefault])
class TestNoL2ResizeModes:
    @pytest.mark.parametrize("seed", [(0,), (42,), (123, 456)])
    def test_l2_metadata_does_not_affect_output(
        self, transform_class: type[CrawlDefaultNoHighRes], seed: tuple[int, ...]
    ) -> None:
        """Missing or conflicting L2 maps must produce identical outputs."""
        transform = transform_class(apply_dynamic_proportion=0.0)
        image = _make_test_image(1200, 900)
        l2_maps = [
            {},
            {"image_min_size=224,l2_resize_factor=2": 0.005},
            {"image_min_size=224,l2_resize_factor=2": 0.02},
        ]

        outcomes = []
        for l2_map in l2_maps:
            meta = TransformImageMetadata(width=1200, height=900, l2_reconstruction_score=l2_map)
            n_toks, pre_meta = transform.precompute_transforms(meta, seed=seed)
            pixels, apply_meta = transform.apply_transforms(image, meta, seed=seed)
            assert dataclasses.asdict(pre_meta) == dataclasses.asdict(apply_meta)
            outcomes.append((n_toks, pre_meta, pixels))

        expected_n_toks, expected_meta, expected_pixels = outcomes[0]
        for n_toks, out_meta, pixels in outcomes[1:]:
            assert n_toks == expected_n_toks
            assert (out_meta.width, out_meta.height) == (expected_meta.width, expected_meta.height)
            np.testing.assert_array_equal(pixels, expected_pixels)

    def test_retains_long_side_cap_and_grid_alignment(
        self, transform_class: type[CrawlDefaultNoHighRes]
    ) -> None:
        """Oversized images retain resizing and 28-pixel grid alignment."""
        transform = transform_class(apply_dynamic_proportion=0.0)
        meta = TransformImageMetadata(width=2000, height=1000)

        _, out_meta = transform.precompute_transforms(meta, seed=(42,))

        grid_size = transform.PATCH_SIZE * transform.SPATIAL_MERGE_SIZE
        assert max(out_meta.width, out_meta.height) <= transform.IMAGE_SIZE
        assert out_meta.width % grid_size == 0
        assert out_meta.height % grid_size == 0
        assert out_meta.width / out_meta.height == pytest.approx(2.0, rel=0.03)

    def test_retains_clip_normalization_and_chw_layout(
        self, transform_class: type[CrawlDefaultNoHighRes]
    ) -> None:
        transform = transform_class(apply_dynamic_proportion=0.0)
        meta = TransformImageMetadata(width=56, height=56)
        white_image = Image.fromarray(np.full((56, 56, 3), 255, dtype=np.uint8))

        pixels, out_meta = transform.apply_transforms(white_image, meta, seed=(42,))

        assert pixels.shape == (3, out_meta.height, out_meta.width)
        expected = (1.0 - np.asarray(CLIP_DATASET_MEAN)) / np.asarray(CLIP_DATASET_STD)
        np.testing.assert_allclose(pixels[:, 0, 0], expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("proportion", [-0.1, 1.1, float("nan")])
    def test_rejects_invalid_dynamic_proportion(
        self, transform_class: type[CrawlDefaultNoHighRes], proportion: float
    ) -> None:
        with pytest.raises(ValueError, match="apply_dynamic_proportion"):
            transform_class(apply_dynamic_proportion=proportion)


# ---------------------------------------------------------------------------
# SometimesHighResDefault
# ---------------------------------------------------------------------------


class TestSometimesHighResDefault:
    @pytest.mark.parametrize(
        ("seed", "expected_size", "expected_n_tokens"),
        [
            ((0,), (364, 532), 266),
            ((3,), (1036, 1540), 2090),
            ((25,), (476, 728), 468),
            ((42,), (700, 1036), 962),
        ],
    )
    def test_matches_mistral_metadata_and_token_count(
        self,
        seed: tuple[int, ...],
        expected_size: tuple[int, int],
        expected_n_tokens: int,
    ) -> None:
        transform = SometimesHighResDefault()
        metadata = TransformImageMetadata(width=2000, height=3000)

        n_tokens, output_metadata = transform.precompute_transforms(metadata, seed)

        assert (output_metadata.width, output_metadata.height) == expected_size
        assert n_tokens == expected_n_tokens

    def test_converts_transparent_images_to_rgb_over_white(self) -> None:
        transform = SometimesHighResDefault()
        image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        metadata = TransformImageMetadata(width=32, height=32)

        pixels, output_metadata = transform.apply_transforms(
            image, metadata, seed=(0,)
        )

        assert (output_metadata.width, output_metadata.height) == (56, 56)
        assert pixels.shape == (3, 56, 56)
        expected_white = (
            1.0 - np.array([0.48145466, 0.4578275, 0.40821073])
        ) / np.array(
            [0.26862954, 0.26130258, 0.27577711],
        )
        np.testing.assert_allclose(pixels[:, 0, 0], expected_white, rtol=1e-6)


# ---------------------------------------------------------------------------
# SimpleImageEncoder integration
# ---------------------------------------------------------------------------


class TestSimpleImageEncoder:
    @pytest.mark.parametrize(
        "transform_type",
        [
            ImageTransformType.NO_TRANSFORM,
            ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES,
            ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES_NO_L2_RESIZE,
            ImageTransformType.GROUNDING_DEFAULT,
            ImageTransformType.GROUNDING_DEFAULT_NO_L2_RESIZE,
            ImageTransformType.SOMETIMES_HIGH_RES_DEFAULT,
        ],
    )
    def test_precompute_returns_positive_tokens(
        self, transform_type: ImageTransformType
    ) -> None:
        encoder = SimpleImageEncoder(transform_type, special_ids=SPECIAL_IDS)
        if transform_type in (
            ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES,
            ImageTransformType.GROUNDING_DEFAULT,
        ):
            meta = CRAWL_METADATA_HIGH_L2
        else:
            meta = TransformImageMetadata(width=512, height=768)

        n_toks, out_meta = encoder.precompute_transforms(meta, seed=(42,))
        assert n_toks > 0
        assert out_meta.width > 0
        assert out_meta.height > 0

    @pytest.mark.parametrize(
        "transform_type",
        [
            ImageTransformType.NO_TRANSFORM,
            ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES,
            ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES_NO_L2_RESIZE,
            ImageTransformType.GROUNDING_DEFAULT,
            ImageTransformType.GROUNDING_DEFAULT_NO_L2_RESIZE,
            ImageTransformType.SOMETIMES_HIGH_RES_DEFAULT,
        ],
    )
    def test_apply_returns_correct_tokens_and_image(
        self, transform_type: ImageTransformType
    ) -> None:
        encoder = SimpleImageEncoder(transform_type, special_ids=SPECIAL_IDS)
        if transform_type in (
            ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES,
            ImageTransformType.GROUNDING_DEFAULT,
        ):
            meta = CRAWL_METADATA_HIGH_L2
        else:
            meta = TransformImageMetadata(width=512, height=768)

        seed = (42,)
        n_toks_pre, _ = encoder.precompute_transforms(meta, seed=seed)
        image = _make_test_image(meta.width, meta.height)
        encoding, out_meta = encoder.apply_transforms_and_encode_image(
            image, meta, seed=seed
        )

        assert len(encoding.tokens) == n_toks_pre
        assert encoding.image is not None
        if transform_type == ImageTransformType.NO_TRANSFORM:
            # NoTransform returns HWC numpy array (no resize/normalize)
            assert encoding.image.shape[0] == out_meta.height
            assert encoding.image.shape[1] == out_meta.width
        else:
            # Crawl returns CHW numpy array
            assert encoding.image.shape[1] == out_meta.height
            assert encoding.image.shape[2] == out_meta.width

    def test_high_res_tokens_are_not_capped_to_default_size(self) -> None:
        encoder = SimpleImageEncoder(
            ImageTransformType.SOMETIMES_HIGH_RES_DEFAULT, special_ids=SPECIAL_IDS
        )
        metadata = TransformImageMetadata(width=1200, height=900)
        seed = (3,)

        n_tokens, precomputed_metadata = encoder.precompute_transforms(
            metadata, seed=seed
        )
        image = _make_test_image(metadata.width, metadata.height)
        encoding, applied_metadata = encoder.apply_transforms_and_encode_image(
            image, metadata, seed=seed
        )

        assert (precomputed_metadata.width, precomputed_metadata.height) == (1204, 924)
        assert precomputed_metadata == applied_metadata
        assert n_tokens == 1452
        assert len(encoding.tokens) == 1452

    def test_sometimes_high_res_infos_match_mistral(self) -> None:
        encoder = SimpleImageEncoder(
            ImageTransformType.SOMETIMES_HIGH_RES_DEFAULT, special_ids=SPECIAL_IDS
        )

        assert encoder.to_infos_dict()["image_process_args"] == {
            "centering": "default",
            "color_jitter_args": None,
            "downsampling_args": {
                "always_high_res_charts_plots": False,
                "apply_dynamic_proportion": 0.0,
                "apply_high_res_proportion": 0.1,
                "high_res_charts_plots_threshold": 1.0,
                "high_res_image_size": 1540,
                "l2_resize_factor": 2,
                "l2_resize_threshold": 0.008773181770004912,
                "min_size": 224,
                "spatial_merge_size": 2,
            },
            "enforce_image_first": True,
            "enforce_single_image": True,
            "eval_image_size": None,
            "image_size": 1036,
            "interpolation": "bicubic",
            "max_patches_per_side": None,
            "max_tokens": None,
            "num_channels": 3,
            "ocr_args": None,
            "patch_size": 14,
            "per_source_transform_args_str": None,
            "random_crop_args": None,
            "random_scale_aspect_args": {
                "random_aspect_prob": 0.0,
                "random_aspect_range_max": 1.11,
                "random_aspect_range_min": 0.9,
                "random_scale_prob": 0.5,
                "random_scale_range_max": 1.0,
                "random_scale_range_min": 0.5,
            },
            "schedule_args": None,
            "transform_type": "sometimes_high_res",
            "use_cv2_resize": True,
            "variable_size": True,
        }

    def test_uuid_stability(self) -> None:
        """UUID should be deterministic for same config."""
        e1 = SimpleImageEncoder(
            ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES, special_ids=SPECIAL_IDS
        )
        e2 = SimpleImageEncoder(
            ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES, special_ids=SPECIAL_IDS
        )
        assert e1.uuid() == e2.uuid()

    def test_uuid_differs_across_types(self) -> None:
        """Different transform types should produce different UUIDs."""
        uuids = {
            SimpleImageEncoder(transform_type, special_ids=SPECIAL_IDS).uuid()
            for transform_type in ImageTransformType
        }
        assert len(uuids) == len(ImageTransformType)

    @pytest.mark.parametrize(
        ("l2_transform_type", "no_l2_transform_type"),
        [
            (
                ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES,
                ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES_NO_L2_RESIZE,
            ),
            (
                ImageTransformType.GROUNDING_DEFAULT,
                ImageTransformType.GROUNDING_DEFAULT_NO_L2_RESIZE,
            ),
        ],
    )
    def test_no_l2_resize_infos_disable_only_dynamic_downsampling(
        self,
        l2_transform_type: ImageTransformType,
        no_l2_transform_type: ImageTransformType,
    ) -> None:
        l2_infos = SimpleImageEncoder(
            l2_transform_type, special_ids=SPECIAL_IDS
        ).to_infos_dict()["image_process_args"]
        no_l2_resize_infos = SimpleImageEncoder(
            no_l2_transform_type, special_ids=SPECIAL_IDS
        ).to_infos_dict()["image_process_args"]

        assert l2_infos["downsampling_args"]["apply_dynamic_proportion"] == 0.9
        assert no_l2_resize_infos["downsampling_args"]["apply_dynamic_proportion"] == 0.0
        expected_no_l2_resize_infos = {
            **l2_infos,
            "downsampling_args": {**l2_infos["downsampling_args"], "apply_dynamic_proportion": 0.0},
        }
        assert no_l2_resize_infos == expected_no_l2_resize_infos

    @pytest.mark.parametrize(
        ("transform_type", "expected_dynamic_proportion"),
        [
            (ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES, 0.9),
            (ImageTransformType.CRAWL_DEFAULT_NO_HIGH_RES_NO_L2_RESIZE, 0.0),
            (ImageTransformType.GROUNDING_DEFAULT, 0.9),
            (ImageTransformType.GROUNDING_DEFAULT_NO_L2_RESIZE, 0.0),
            (ImageTransformType.SOMETIMES_HIGH_RES_DEFAULT, 0.0),
        ],
    )
    def test_dynamic_proportion_matches_runtime_and_infos(
        self,
        transform_type: ImageTransformType,
        expected_dynamic_proportion: float,
    ) -> None:
        encoder = SimpleImageEncoder(transform_type, special_ids=SPECIAL_IDS)
        transform = encoder.get_transforms()

        assert isinstance(transform, CrawlDefaultNoHighRes)
        assert transform._apply_dynamic_proportion == expected_dynamic_proportion
        assert (
            encoder.to_infos_dict()["image_process_args"]["downsampling_args"][
                "apply_dynamic_proportion"
            ]
            == expected_dynamic_proportion
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestImageTransformsFactory:
    @pytest.mark.parametrize(
        "transform_type",
        list(ImageTransformType),
    )
    def test_factory_returns_correct_type(
        self, transform_type: ImageTransformType
    ) -> None:
        transform = image_transforms_factory(transform_type, SPECIAL_IDS)
        assert hasattr(transform, "precompute_transforms")
        assert hasattr(transform, "apply_transforms")
