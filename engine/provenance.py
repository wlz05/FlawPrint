# -*- coding: utf-8 -*-
# Manuscript and Source Audit: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import ast
import difflib
import hashlib
import io
from pathlib import Path
import re
import tokenize
from engine.storage import relative_path, read_json, write_json, same_path
def document_path(manuscript: str | None = None) -> Path:
    if manuscript is not None:
        return relative_path(manuscript)
    internal = relative_path("../writing/FlawPrint.md")
    return internal if internal.is_file() else relative_path("./README.md")
def snapshot_path(manuscript: str | None = None, snapshot: str | None = None) -> Path:
    if snapshot is not None:
        return relative_path(snapshot)
    internal = relative_path("../writing/FlawPrint.md")
    return relative_path("../procedure/materials/implementation/manuscript_snapshot.md") if same_path(document_path(manuscript), internal) else relative_path("./outputs/audit_snapshot.md")
def manuscript_status(manuscript: str | None = None, snapshot: str | None = None, accept: bool = False) -> dict:
    path, saved = document_path(manuscript), snapshot_path(manuscript, snapshot)
    if same_path(path, saved) or same_path(path, saved.with_suffix(".json")):
        raise ValueError("The document and snapshot artifacts must be distinct")
    content = path.read_bytes()
    current = content.decode("utf-8-sig")
    previous = saved.read_bytes().decode("utf-8-sig") if saved.exists() else ""
    result = {"manuscript": path.as_posix(), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "formulas": sorted(set(re.findall(r"\\tag\{(F\d+)\}", current)), key=lambda value: int(value[1:])), "headings": [line for line in current.splitlines() if line.startswith("###")], "changed": current != previous, "diff": "\n".join(difflib.unified_diff(previous.splitlines(), current.splitlines(), fromfile="accepted_manuscript", tofile="current_manuscript", lineterm="")) if previous else "Initial manuscript snapshot"}
    if accept:
        if path.read_bytes() != content:
            raise RuntimeError("The manuscript changed during review; audit it again before acceptance")
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(content)
        write_json(saved.with_suffix(".json"), {key: value for key, value in result.items() if key != "diff"})
    return result
def source_files(root: str = ".") -> list[Path]:
    return sorted(path for path in relative_path(root).rglob("*.py") if not set(path.parts).intersection((".git", ".venv", "venv", "node_modules", "__pycache__")))
def source_digest(root: str = ".") -> str:
    checksum = hashlib.sha256()
    for path in source_files(root):
        checksum.update(path.relative_to(relative_path(root)).as_posix().encode("utf-8") + b"\x00")
        checksum.update(path.read_bytes() + b"\x00")
    return checksum.hexdigest()
def manuscript_title(manuscript: str | None = None) -> str:
    for line in document_path(manuscript).read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            if not title or not title.isascii():
                raise ValueError("The manuscript requires an English ASCII top-level title")
            return title
    raise ValueError("The manuscript has no top-level title")
def conventions(root: str = ".", manuscript: str | None = None) -> dict:
    files = source_files(root)
    issues, lines, comments = [], 0, 0
    title = manuscript_title(manuscript)
    for path in files:
        if not re.fullmatch(r"[a-z]+\.py", path.name):
            issues.append({"file": path.as_posix(), "rule": "lowercase-word-filename"})
        source = path.read_text(encoding="utf-8")
        lines += len(source.splitlines())
        content = source.splitlines()
        valid_header = bool(len(content) >= 2 and re.fullmatch(r"# [A-Za-z][A-Za-z ]*: " + re.escape(title), content[1]))
        if not content or content[0] != "# -*- coding: utf-8 -*-":
            issues.append({"file": path.as_posix(), "rule": "utf8-header"})
        if len(content) < 4 or not valid_header or content[2] != "":
            issues.append({"file": path.as_posix(), "rule": "title-header"})
        for number, line in enumerate(source.splitlines(), 1):
            if not line.strip() and number != 3:
                issues.append({"file": path.as_posix(), "line": number, "rule": "blank-line"})
            if not line.isascii():
                issues.append({"file": path.as_posix(), "line": number, "rule": "non-english-source"})
        try:
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.match(r"^[A-Za-z]:[\\/]", node.value):
                    issues.append({"file": path.as_posix(), "line": node.lineno, "rule": "absolute-path-literal"})
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.COMMENT:
                    comments += 1
                    if not (token.start[0] == 2 and valid_header) and (len(token.string) > 100 or len(re.findall(r"[.!?](?:\s|$)", token.string)) > 1):
                        issues.append({"file": path.as_posix(), "line": token.start[0], "rule": "comment-length"})
        except (SyntaxError, tokenize.TokenError) as error:
            issues.append({"file": path.as_posix(), "rule": "syntax", "reason": str(error)})
    return {"python_files": len(files), "code_lines": lines, "comments": comments, "header_title": title, "encoding": "utf-8", "header_blank_line": 3, "issues": issues, "passed": not issues}
