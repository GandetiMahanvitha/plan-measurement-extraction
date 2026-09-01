from pathlib import Path
import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image  and fail  when OpenCV cannot read it."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def build_variants(image: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    """Create grayscale, enlarged, blurred, and thresholded image variants.

    These variants give the OCR passes readable inputs for labels and dimensions.
    """
    factor = float(config["preprocessing"].get("upscale_factor", 2.0))
    kernel = int(config["preprocessing"].get("gaussian_kernel", 3))
    if kernel % 2 == 0:
        kernel += 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(
        gray,
        None,
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_CUBIC,
    )
    blurred = cv2.GaussianBlur(upscaled, (kernel, kernel), 0)
    _, thresholded = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return {
        "original": image,
        "upscaled_gray": upscaled,
        "thresholded": thresholded,
    }
