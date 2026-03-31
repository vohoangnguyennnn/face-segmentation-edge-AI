from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DATASET_ROOT = Path("segmentation_dataset")
IMG_DIR = DATASET_ROOT / "images"
MASK_DIR = DATASET_ROOT / "masks"
LOG_DIR = DATASET_ROOT / "logs"

VALID_LABELS = {"object"}
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BACKGROUND_VALUE = 0
OBJECT_VALUE = 255


@dataclass
class ConversionStats:
    total_images: int = 0
    processed_images: int = 0
    unreadable_images: list[str] = field(default_factory=list)
    missing_json: list[str] = field(default_factory=list)
    invalid_json: list[str] = field(default_factory=list)
    size_mismatches: list[str] = field(default_factory=list)
    invalid_polygons: list[str] = field(default_factory=list)
    skipped_labels: list[str] = field(default_factory=list)
    skipped_shape_types: list[str] = field(default_factory=list)
    empty_masks: list[str] = field(default_factory=list)
    write_failures: list[str] = field(default_factory=list)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure console and file logging for the conversion run."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("labelme_json_to_mask")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / "conversion.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def list_image_files(image_dir: Path) -> list[Path]:
    """Return supported image files sorted by name."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {image_dir}")

    return sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def load_annotation(json_path: Path, logger: logging.Logger, stats: ConversionStats) -> dict | None:
    """Load a Labelme annotation file safely."""
    try:
        with json_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        message = f"{json_path.name}: annotation file is missing."
    except json.JSONDecodeError as exc:
        message = f"{json_path.name}: invalid JSON ({exc})."
    except OSError as exc:
        message = f"{json_path.name}: failed to read annotation ({exc})."

    logger.warning(message)
    stats.invalid_json.append(message)
    return None


def warn_and_store(logger: logging.Logger, bucket: list[str], message: str) -> None:
    """Log a warning and persist the same message in a stats bucket."""
    logger.warning(message)
    bucket.append(message)


def validate_image_size(
    annotation: dict,
    image_height: int,
    image_width: int,
    image_name: str,
    logger: logging.Logger,
    stats: ConversionStats,
) -> None:
    """Warn when the JSON metadata does not match the actual image size."""
    json_height = annotation.get("imageHeight")
    json_width = annotation.get("imageWidth")

    if not isinstance(json_height, int) or not isinstance(json_width, int):
        return

    if json_height != image_height or json_width != image_width:
        message = (
            f"{image_name}: size mismatch, JSON=({json_height}, {json_width}) "
            f"vs image=({image_height}, {image_width})."
        )
        warn_and_store(logger, stats.size_mismatches, message)


def validate_polygon_points(
    points: object,
    image_name: str,
    shape_index: int,
    logger: logging.Logger,
    stats: ConversionStats,
) -> np.ndarray | None:
    """Return a valid polygon array or None when the points are not usable."""
    if not isinstance(points, list):
        message = f"{image_name}: shape {shape_index} skipped, points are not a list."
        warn_and_store(logger, stats.invalid_polygons, message)
        return None

    polygon_points: list[list[float]] = []
    for point_index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            message = (
                f"{image_name}: shape {shape_index} skipped, point {point_index} "
                "is not a valid [x, y] pair."
            )
            warn_and_store(logger, stats.invalid_polygons, message)
            return None

        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            message = (
                f"{image_name}: shape {shape_index} skipped, point {point_index} "
                "contains non-numeric values."
            )
            warn_and_store(logger, stats.invalid_polygons, message)
            return None

        if not np.isfinite(x) or not np.isfinite(y):
            message = (
                f"{image_name}: shape {shape_index} skipped, point {point_index} "
                "contains non-finite values."
            )
            warn_and_store(logger, stats.invalid_polygons, message)
            return None

        polygon_points.append([x, y])

    if len(polygon_points) < 3:
        message = (
            f"{image_name}: shape {shape_index} skipped, polygon has fewer than 3 points."
        )
        warn_and_store(logger, stats.invalid_polygons, message)
        return None

    return np.asarray(polygon_points, dtype=np.float32)


def polygon_to_mask(points: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterize a polygon into a binary mask with values {0, 255}."""
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.rint(points).astype(np.int32)
    cv2.fillPoly(mask, [polygon], OBJECT_VALUE)
    return mask


def build_mask_from_shapes(
    shapes: object,
    image_name: str,
    height: int,
    width: int,
    logger: logging.Logger,
    stats: ConversionStats,
) -> np.ndarray:
    """Merge all valid polygon annotations into a single binary mask."""
    full_mask = np.zeros((height, width), dtype=np.uint8)

    if not isinstance(shapes, list):
        warn_and_store(
            logger,
            stats.invalid_json,
            f"{image_name}: 'shapes' is missing or not a list.",
        )
        return full_mask

    for shape_index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            warn_and_store(
                logger,
                stats.invalid_json,
                f"{image_name}: shape {shape_index} skipped, shape entry is not an object.",
            )
            continue

        label = str(shape.get("label", ""))
        if label not in VALID_LABELS:
            warn_and_store(
                logger,
                stats.skipped_labels,
                f"{image_name}: shape {shape_index} skipped, label '{label}' is not in {sorted(VALID_LABELS)}.",
            )
            continue

        shape_type = shape.get("shape_type", "polygon")
        if shape_type != "polygon":
            warn_and_store(
                logger,
                stats.skipped_shape_types,
                f"{image_name}: shape {shape_index} skipped, shape_type '{shape_type}' is not supported.",
            )
            continue

        points = validate_polygon_points(
            shape.get("points"),
            image_name=image_name,
            shape_index=shape_index,
            logger=logger,
            stats=stats,
        )
        if points is None:
            continue

        shape_mask = polygon_to_mask(points, height, width)
        full_mask = cv2.bitwise_or(full_mask, shape_mask)

    return full_mask


def write_lines(path: Path, lines: Iterable[str]) -> None:
    """Write one log message per line."""
    with path.open("w", encoding="utf-8") as file:
        for line in lines:
            file.write(f"{line}\n")


def save_category_logs(log_dir: Path, stats: ConversionStats) -> None:
    """Persist category-specific logs for downstream dataset QA."""
    write_lines(log_dir / "missing_json.txt", stats.missing_json)
    write_lines(log_dir / "invalid_json.txt", stats.invalid_json)
    write_lines(log_dir / "size_mismatches.txt", stats.size_mismatches)
    write_lines(log_dir / "invalid_polygons.txt", stats.invalid_polygons)
    write_lines(log_dir / "skipped_labels.txt", stats.skipped_labels)
    write_lines(log_dir / "skipped_shape_types.txt", stats.skipped_shape_types)
    write_lines(log_dir / "empty_masks.txt", stats.empty_masks)
    write_lines(log_dir / "unreadable_images.txt", stats.unreadable_images)
    write_lines(log_dir / "write_failures.txt", stats.write_failures)


def process_image(image_path: Path, logger: logging.Logger, stats: ConversionStats) -> None:
    """Process one image/annotation pair and save a binary mask."""
    stats.total_images += 1
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        warn_and_store(
            logger,
            stats.unreadable_images,
            f"{image_path.name}: image could not be read.",
        )
        return

    height, width = image.shape[:2]
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        warn_and_store(
            logger,
            stats.missing_json,
            f"{image_path.name}: missing annotation file {json_path.name}.",
        )
        return

    annotation = load_annotation(json_path, logger, stats)
    if annotation is None:
        return

    validate_image_size(
        annotation,
        image_height=height,
        image_width=width,
        image_name=image_path.name,
        logger=logger,
        stats=stats,
    )

    mask = build_mask_from_shapes(
        shapes=annotation.get("shapes", []),
        image_name=image_path.name,
        height=height,
        width=width,
        logger=logger,
        stats=stats,
    )

    output_path = MASK_DIR / f"{image_path.stem}.png"
    if not cv2.imwrite(str(output_path), mask):
        warn_and_store(
            logger,
            stats.write_failures,
            f"{output_path.name}: failed to write mask.",
        )
        return

    stats.processed_images += 1
    if int(mask.max()) == BACKGROUND_VALUE:
        warn_and_store(
            logger,
            stats.empty_masks,
            f"{output_path.name}: mask is empty (all pixels are 0).",
        )


def main() -> None:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOG_DIR)

    try:
        image_files = list_image_files(IMG_DIR)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error(exc)
        raise SystemExit(1) from exc

    stats = ConversionStats()

    if not image_files:
        logger.warning("No supported image files found.")

    for image_path in image_files:
        process_image(image_path, logger, stats)

    save_category_logs(LOG_DIR, stats)

    print(f"Mask directory: {MASK_DIR}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Total images found: {stats.total_images}")
    print(f"Masks written: {stats.processed_images}")
    print(f"Missing JSON files: {len(stats.missing_json)}")
    print(f"Invalid JSON files: {len(stats.invalid_json)}")
    print(f"Invalid polygons skipped: {len(stats.invalid_polygons)}")
    print(f"Skipped labels: {len(stats.skipped_labels)}")
    print(f"Unsupported shape types skipped: {len(stats.skipped_shape_types)}")
    print(f"Size mismatches: {len(stats.size_mismatches)}")
    print(f"Unreadable images: {len(stats.unreadable_images)}")
    print(f"Mask write failures: {len(stats.write_failures)}")
    print(f"Empty masks: {len(stats.empty_masks)}")


if __name__ == "__main__":
    main()
