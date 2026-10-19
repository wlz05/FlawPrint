# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import argparse
from PIL import Image
from engine.storage import read_rows, write_rows, relative_path, same_path
from engine.data import normalize
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute optional local CLIP alignment control scores.")
    parser.add_argument("manifest")
    parser.add_argument("--model", default="./models/clip")
    parser.add_argument("--output", default="./data/clip_predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    rows = normalize(read_rows(args.manifest))
    if not rows or any(not row["image"] or not row["text"].strip() for row in rows):
        parser.error("CLIP alignment requires a local image and nonempty text for every sample")
    model_path = relative_path(args.model)
    if not model_path.is_dir():
        parser.error("The local CLIP model directory is missing")
    if same_path(args.output, args.manifest):
        parser.error("The output path cannot overwrite the manifest")
    if any(not relative_path(row["image"]).is_file() for row in rows):
        parser.error("An input image is missing; run main.py validate with check-images")
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(str(model_path), local_files_only=True).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(str(model_path), local_files_only=True)
    output = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        images = []
        for row in batch:
            with Image.open(relative_path(row["image"])) as image:
                images.append(image.convert("RGB"))
        inputs = processor(text=[row.get("text", "") for row in batch], images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            result = model(**inputs)
            image_features = torch.nn.functional.normalize(result.image_embeds, dim=-1)
            text_features = torch.nn.functional.normalize(result.text_embeds, dim=-1)
            similarities = (image_features * text_features).sum(-1).cpu().numpy()
        output.extend({"id": row["id"], "logit": float(-similarity), "alignment_cosine": float(similarity), "method": "clip_alignment_control", "score_scope": "Negative alignment is a control score, not a calibrated fake-news probability"} for row, similarity in zip(batch, similarities))
    write_rows(args.output, output)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
