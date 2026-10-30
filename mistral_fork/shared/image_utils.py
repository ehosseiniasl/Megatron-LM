from dataclasses import dataclass, field

from PIL import Image

from mistral_fork.mistral_common.tokens.tokenizers.image import ImageEncoding


@dataclass
class TrainingImageEncoding(ImageEncoding):
    geometries: list = field(default_factory=list)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """Convert a PIL image to RGB. Transparent background becomes white."""
    original_format = image.format
    if image.mode == "RGB":
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    white_bg = Image.new("RGBA", image.size, "WHITE")
    white_bg.paste(image, (0, 0), image)
    output_image = white_bg.convert("RGB")
    if original_format:
        output_image.format = original_format
    return output_image
