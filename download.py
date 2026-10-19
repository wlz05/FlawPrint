# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import io
import requests
from pypdf import PdfReader
from engine.storage import read_json, write_json, relative_path, digest
def fetch(paper: dict, output: Path, timeout: float = 60.0) -> dict:
    target = output / relative_path(paper["file"])
    record = dict(paper)
    try:
        if not target.exists():
            content = None
            failure = None
            for url in [paper["pdf"], *paper.get("fallback_pdf", [])]:
                try:
                    response = requests.get(url, timeout=timeout, headers={"User-Agent": "FlawPrintResearch/1.0"})
                    response.raise_for_status()
                    content = response.content
                    record["download_url"] = url
                    break
                except requests.RequestException as error:
                    failure = error
            if content is None:
                raise failure
            if not content.startswith(b"%PDF-"):
                raise ValueError("The server did not return a PDF")
            reader = PdfReader(io.BytesIO(content))
            if len(reader.pages) < 1:
                raise ValueError("The PDF has no pages")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        reader = PdfReader(target)
        text = (reader.pages[0].extract_text() or "").strip()
        record.update(status="downloaded", bytes=target.stat().st_size, pages=len(reader.pages), sha256=digest(target), first_page_excerpt=" ".join(text.split()[:20]), checked_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    return record
def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and validate official baseline papers.")
    parser.add_argument("--sources", default="./configs/sources.json")
    parser.add_argument("--output", default="./references")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--only", action="append")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error("Workers and timeout must be positive")
    sources = read_json(args.sources)["papers"]
    papers = [paper for paper in sources if not args.only or paper["id"] in args.only]
    if not papers:
        parser.error("No matching source identifiers")
    output = relative_path(args.output)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(lambda paper: fetch(paper, output, args.timeout), papers))
    manifest = output / "download_manifest.json"
    previous = read_json(manifest)["papers"] if manifest.exists() else []
    lookup = {record["id"]: record for record in previous + records}
    write_json(manifest, {"papers": [lookup.get(source["id"], dict(source, status="pending")) for source in sources]})
    for record in records:
        print(f"{record['id']}: {record['status']}")
    if any(record["status"] != "downloaded" for record in records):
        raise SystemExit(1)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
