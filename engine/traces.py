# -*- coding: utf-8 -*-
# Traces: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import re
import unicodedata
import numpy as np
from PIL import Image, ExifTags
from scipy import ndimage, fft
from engine.storage import relative_path, digest
NAMES = ["laplacian_variance", "spectral_tail_slope", "wavelet_noise_mad", "gradient_anisotropy", "radial_chromatic_difference", "noise_intensity_slope", "exposure_skew", "saliency_offset", "thirds_distance", "border_asymmetry", "saliency_concentration", "edge_density", "jpeg_quantization_mean", "dct_gap_ratio", "resampling_peak", "resolution_ladder_distance", "grid_energy", "exif_completeness", "software_tag_present", "crop_border_energy", "text_length", "character_diversity", "hashtag_density", "symbol_density", "exclamation_density", "uppercase_ratio"]
FAMILIES = {"capture": list(range(7)), "composition": list(range(7, 12)), "transmission": list(range(12, 20)), "narrative": list(range(20, 26))}
@dataclass(frozen=True)
class TraceConfig:
    max_side: int = 768
    tile_grid: int = 4
    min_side: int = 32
    def validate(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.max_side, self.tile_grid, self.min_side)):
            raise ValueError("Image and tile dimensions must be integers")
        if self.min_side < 16 or self.max_side < self.min_side or not 2 <= self.tile_grid <= 8:
            raise ValueError("Invalid image or tile dimensions")
def safe_skew(values: np.ndarray) -> float:
    centered = values - values.mean()
    scale = np.sqrt(np.mean(centered ** 2))
    return float(np.mean(centered ** 3) / max(scale ** 3, 1e-12))
def radial_slope(gray: np.ndarray) -> float:
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * window))) ** 2
    yy, xx = np.indices(gray.shape)
    radius = np.sqrt(((yy - gray.shape[0] / 2) / gray.shape[0]) ** 2 + ((xx - gray.shape[1] / 2) / gray.shape[1]) ** 2)
    edges = np.linspace(0.12, 0.48, 13)
    means = [np.mean(power[(radius >= left) & (radius < right)]) for left, right in zip(edges[:-1], edges[1:])]
    return float(np.polyfit(np.log((edges[:-1] + edges[1:]) / 2), np.log(np.maximum(means, 1e-12)), 1)[0])
def noise_slope(gray: np.ndarray, residual: np.ndarray) -> float:
    samples = []
    for y in range(0, gray.shape[0] - 7, 8):
        for x in range(0, gray.shape[1] - 7, 8):
            patch = gray[y:y + 8, x:x + 8]
            samples.append((patch.mean(), residual[y:y + 8, x:x + 8].var(), patch.var()))
    if len(samples) < 4:
        return 0.0
    values = np.asarray(samples)
    values = values[values[:, 2] <= np.quantile(values[:, 2], 0.5)]
    if len(values) < 3 or np.ptp(values[:, 0]) < 1e-8:
        return 0.0
    return float(np.polyfit(values[:, 0], values[:, 1], 1)[0])
def pixel_traces(rgb: np.ndarray) -> np.ndarray:
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    residual = gray - ndimage.median_filter(gray, size=3, mode="reflect")
    # Estimate noise from the finest Haar diagonal band.
    even = gray[:gray.shape[0] // 2 * 2, :gray.shape[1] // 2 * 2]
    diagonal_wavelet = (even[0::2, 0::2] - even[1::2, 0::2] - even[0::2, 1::2] + even[1::2, 1::2]) / 2
    wavelet_mad = np.median(np.abs(diagonal_wavelet - np.median(diagonal_wavelet))) / 0.67448975
    tensor = np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
    eigenvalues = np.linalg.eigvalsh(tensor)
    anisotropy = (eigenvalues[-1] - eigenvalues[0]) / max(eigenvalues.sum(), 1e-12)
    yy, xx = np.indices(gray.shape)
    radial = np.sqrt(((yy + 0.5) / gray.shape[0] - 0.5) ** 2 + ((xx + 0.5) / gray.shape[1] - 0.5) ** 2)
    chromatic = np.abs(ndimage.gaussian_filter(rgb[:, :, 0], 1) - ndimage.gaussian_filter(rgb[:, :, 2], 1))
    chromatic_covariance = np.mean((radial - radial.mean()) * (chromatic - chromatic.mean()))
    saliency = ndimage.gaussian_filter(magnitude, 2) + 1e-12
    weights = saliency / saliency.sum()
    center = np.array([np.sum(weights * (xx + 0.5) / gray.shape[1]), np.sum(weights * (yy + 0.5) / gray.shape[0])])
    thirds = np.array([[x, y] for x in (1 / 3, 2 / 3) for y in (1 / 3, 2 / 3)])
    border = max(1, min(gray.shape) // 16)
    sides = np.array([magnitude[:border].mean(), magnitude[-border:].mean(), magnitude[:, :border].mean(), magnitude[:, -border:].mean()])
    blocks = gray[:gray.shape[0] // 8 * 8, :gray.shape[1] // 8 * 8]
    blocks = blocks.reshape(blocks.shape[0] // 8, 8, blocks.shape[1] // 8, 8).transpose(0, 2, 1, 3)
    coefficients = fft.dctn(blocks - 0.5, axes=(-2, -1), norm="ortho")
    ac = coefficients[..., 1:, 1:].reshape(-1)
    histogram, _ = np.histogram(ac, bins=41, range=(-0.5, 0.5))
    gap = np.mean(histogram[1:-1] < np.minimum(histogram[:-2], histogram[2:]) * 0.25)
    profile = np.r_[np.mean(np.abs(np.diff(gray, n=2, axis=0)), axis=1), np.mean(np.abs(np.diff(gray, n=2, axis=1)), axis=0)]
    spectrum = np.abs(np.fft.rfft(profile - profile.mean()))[1:]
    resampling = float(spectrum.max() / max(spectrum.sum(), 1e-12)) if spectrum.size else 0.0
    horizontal = np.abs(np.diff(gray, axis=0))
    vertical = np.abs(np.diff(gray, axis=1))
    horizontal_grid = horizontal[7::8]
    vertical_grid = vertical[:, 7::8]
    grid = 0.5 * ((horizontal_grid.mean() if horizontal_grid.size else 0) / max(horizontal.mean(), 1e-12) + (vertical_grid.mean() if vertical_grid.size else 0) / max(vertical.mean(), 1e-12))
    values = [ndimage.laplace(gray).var(), radial_slope(gray), wavelet_mad, anisotropy, chromatic_covariance, noise_slope(gray, residual), safe_skew(gray), np.linalg.norm(center - 0.5), np.min(np.linalg.norm(thirds - center, axis=1)), np.std(sides) / max(np.mean(sides), 1e-12), np.sum(weights ** 2) * gray.size, np.mean(magnitude > magnitude.mean() + magnitude.std()), 0.0, gap, resampling, 0.0, grid, 0.0, 0.0, sides.mean()]
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
def narrative_traces(text: str) -> np.ndarray:
    length = max(len(text), 1)
    letters = [char for char in text if char.isalpha()]
    symbols = sum(unicodedata.category(char).startswith("S") for char in text)
    return np.asarray([len(text), len(set(text)) / length, len(re.findall(r"#[^\s#]+", text)) / length, symbols / length, sum(char in ("!", "\uFF01") for char in text) / length, sum(char.isupper() for char in letters) / max(len(letters), 1)], dtype=float)
def extract_one(row: dict, config: TraceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    config.validate()
    if not row.get("image"):
        values = np.r_[np.zeros(20), narrative_traces(row.get("text", ""))]
        observed = np.r_[np.zeros(20, dtype=bool), np.ones(6, dtype=bool)]
        patches = np.zeros((config.tile_grid ** 2, len(NAMES)))
        return values, observed, patches, dict(row, has_image=False, physical_input_status="missing_image", trace_version="flawprint-traces-v1")
    path = relative_path(row["image"])
    content_hash = digest(path)
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) > 1:
            raise ValueError("Animated or multipage inputs require an explicit still-image conversion")
        original_width, original_height = image.size
        if min(image.size) < config.min_side:
            raise ValueError(f"Image is too small: {row['id']}")
        image_format = str(image.format or "unknown")
        exif = image.getexif()
        tags = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
        quantization = getattr(image, "quantization", None)
        copy = image.convert("RGB")
        copy.thumbnail((config.max_side, config.max_side), Image.Resampling.LANCZOS)
        rgb = np.asarray(copy, dtype=float) / 255.0
    if min(rgb.shape[:2]) < config.tile_grid * 8:
        raise ValueError("Image cannot support the configured tile grid")
    values = np.r_[pixel_traces(rgb), narrative_traces(row.get("text", ""))]
    observed = np.ones(len(NAMES), dtype=bool)
    observed[12] = bool(quantization)
    values[12] = float(np.mean([number for table in (quantization or {}).values() for number in table])) if quantization else 0.0
    ladders = np.array([224, 256, 320, 480, 512, 640, 720, 768, 1080, 1280, 1920, 2048, 4096])
    values[15] = np.min(np.abs(np.log(max(original_width, original_height) / ladders)))
    values[17] = sum(key in tags for key in ("Make", "Model", "DateTime", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength")) / 7
    values[18] = float(bool(tags.get("Software")))
    patches = []
    for y in np.array_split(np.arange(rgb.shape[0]), config.tile_grid):
        for x in np.array_split(np.arange(rgb.shape[1]), config.tile_grid):
            patches.append(np.r_[pixel_traces(rgb[np.ix_(y, x)]), np.zeros(6)])
    if digest(path) != content_hash:
        raise ValueError("An image changed during extraction")
    metadata = dict(row, has_image=True, width=original_width, height=original_height, format=image_format, content_sha256=content_hash, trace_version="flawprint-traces-v1", resized_for_pixel_traces=max(original_width, original_height) > config.max_side, pixel_policy="RGB encoded raster without EXIF rotation or alpha compositing")
    return values, observed, np.asarray(patches), metadata
def extract(rows: list[dict], config: TraceConfig, workers: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    if not rows or isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Provide samples and a positive worker count")
    config.validate()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        result = list(executor.map(lambda row: extract_one(row, config), rows))
    values, observed, patches, metadata = zip(*result)
    return np.asarray(values), np.asarray(observed), np.asarray(patches), list(metadata)
