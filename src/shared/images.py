"""Validate image bytes without trusting file extensions."""

import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from shared.errors import ToolError
from shared.files import read_limited


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    mime: str
    width: int
    height: int


def load_image(path: Path) -> ImageInput:
    data = read_limited(path, 5_000_000)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ToolError("Only PNG and JPEG images are supported.")
                if image.width * image.height > 12_000_000:
                    raise ToolError("Image exceeds the 12 megapixel limit.")
                mime = "image/png" if image.format == "PNG" else "image/jpeg"
                width, height = image.size
                image.verify()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ToolError("Invalid or unsafe image data.") from exc
    return ImageInput(data, mime, width, height)
