# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import argparse
import inspect
import numpy as np
from PIL import Image
from PIL.Image import DecompressionBombError
from engine import embeddings
from engine.storage import read_rows, write_rows, relative_path, resolved, same_path, protect_outputs, digest
from engine.data import normalize
from engine.provenance import source_digest

def alignment(records: list[dict], directory: str, batch_size: int = 64, device: str = "cpu", max_length: int = 128, text_processing: str = "identity") -> list[dict]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    embeddings.context(max_length)
    rows = normalize(records)
    if not rows or any(not row["image"] or not row["text"].strip() for row in rows):
        raise ValueError("CLIP alignment requires a local image and nonempty text for every sample")
    texts = [embeddings.clean_text(row["text"], text_processing) for row in rows]
    if any(not text.strip() for text in texts):
        raise ValueError("Text preprocessing removed an alignment input")
    device = embeddings.runtime(device)
    model_path = relative_path(directory)
    fingerprint, implementation = embeddings.checkpoint_digest(model_path), source_digest()
    images_sha256 = {}
    for row in rows:
        path = relative_path(row["image"]).as_posix()
        if path not in images_sha256:
            images_sha256[path] = digest(path)
        if row.get("content_sha256") and row["content_sha256"] != images_sha256[path]:
            raise ValueError("The manifest image checksum is stale")
    import torch
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(model_path.as_posix(), local_files_only=True).to(device).eval()
    if model.config.model_type not in ("clip", "chinese_clip"):
        raise ValueError("Alignment requires an OpenAI CLIP or Chinese-CLIP checkpoint")
    processor = AutoProcessor.from_pretrained(model_path.as_posix(), local_files_only=True)
    limit = embeddings.context(max_length, getattr(processor.tokenizer, "model_max_length", None), getattr(model.config.text_config, "max_position_embeddings", None))
    accepted = inspect.signature(model.get_text_features).parameters
    provenance = {"encoder_sha256": fingerprint, "source_sha256": implementation, "max_text_length": max_length, "text_context": limit, "text_processing": text_processing, "text_processing_sha256": embeddings.processing_hash(text_processing)}
    output = []
    for start in range(0, len(rows), batch_size):
        batch, images = rows[start:start + batch_size], []
        for row in batch:
            with Image.open(relative_path(row["image"])) as image:
                if getattr(image, "n_frames", 1) > 1 and row.get("image_frame") != 0:
                    raise ValueError("Animated or multipage inputs require an explicit first-frame selection")
                image.seek(0)
                images.append(image.convert("RGB"))
        inputs = processor(text=texts[start:start + len(batch)], images=images, return_tensors="pt", padding=True, truncation=True, max_length=limit)
        text_inputs = {key: value.to(device) for key, value in inputs.items() if key in accepted and key != "pixel_values"}
        with torch.inference_mode():
            # Use the same native projections as the deep feature extractor.
            vectors = {"clip_text": embeddings.cliptext(model, text_inputs), "clip_image": model.get_image_features(pixel_values=inputs["pixel_values"].to(device))}
        for name, value in vectors.items():
            value = getattr(value, "pooler_output", value)
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.ndim != 2 or value.shape[0] != len(batch) or value.shape[1] < 1 or not torch.isfinite(value).all():
                raise ValueError("CLIP alignment requires finite aligned projection vectors")
            vectors[name] = value.detach().double().cpu().numpy()
        from engine.learning import agreement
        similarities = agreement(vectors, np.ones((len(batch), 2), dtype=bool))
        if not similarities[:, 1].all():
            raise ValueError("CLIP alignment requires nonzero text and image projections")
        output.extend({"id": row["id"], "logit": -float(value), "alignment_cosine": float(value), "method": "clip_alignment_control", "score_scope": "Negative alignment is a control score, not a calibrated fake-news probability", "content_sha256": images_sha256[row["image"]], "image_frame": row.get("image_frame"), **provenance} for row, value in zip(batch, similarities[:, 0]))
    if embeddings.checkpoint_digest(model_path) != fingerprint or source_digest() != implementation:
        raise ValueError("The CLIP checkpoint or implementation changed during alignment")
    if any(digest(path) != checksum for path, checksum in images_sha256.items()):
        raise ValueError("An image changed during alignment")
    return output

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute optional local CLIP alignment control scores.")
    parser.add_argument("manifest")
    parser.add_argument("--model", default="./models/clip")
    parser.add_argument("--output", default="./data/clip_predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--text-processing", choices=("identity", "eann"), default="identity")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        manifest = relative_path(args.manifest)
        before = digest(manifest)
        rows = normalize(read_rows(manifest))
        model_path = relative_path(args.model)
        if not model_path.is_dir():
            raise ValueError("The local CLIP model directory is missing")
        if same_path(args.output, manifest):
            raise ValueError("The output path cannot overwrite the manifest")
        if resolved(args.output).is_relative_to(resolved(model_path)):
            raise ValueError("Alignment output must remain outside the encoder directory")
        protect_outputs([args.output], [row["image"] for row in rows])
        output = alignment(rows, model_path, args.batch_size, args.device, args.max_length, args.text_processing)
        if digest(manifest) != before:
            raise ValueError("The manifest changed during alignment")
        write_rows(args.output, output)
    except ImportError as error:
        parser.error('Install deep dependencies with python -m pip install "torch>=2.2" "transformers>=4.40,<6"')
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, DecompressionBombError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
