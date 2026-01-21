#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_project.py
- Generates:
  1) PROJECT_TREE.txt  : readable folder/file tree (ASCII)
  2) PROJECT_SNAPSHOT.txt : concatenated source snapshot with file markers
  3) PROJECT_TREE.dot  : Graphviz DOT graph (optional "picture")
  4) PROJECT_TREE.png  : if Graphviz 'dot' is installed and --png is used

Usage (from project root):
  python export_project.py
  python export_project.py --depth 6 --png
  python export_project.py --include-ext .py .toml .md .yaml .yml .json
  python export_project.py --max-file-bytes 400000 --max-total-bytes 8000000

Notes:
- Excludes common virtualenv/cache/build directories by default.
- Tries to avoid binary files; snapshots text-like extensions by default.
- Redaction: you should still remove/replace secrets before sharing outputs.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    "node_modules",
}

DEFAULT_EXCLUDE_FILES = {
    "PROJECT_SNAPSHOT.txt",
    "PROJECT_TREE.txt",
    "PROJECT_TREE.dot",
    "PROJECT_TREE.png",
    "PROJECT_TREE.md",
    "gtp_promt_sample.txt",
}

DEFAULT_INCLUDE_EXT = {
    ".py",
    ".pyi",
    ".toml",
    ".cfg",
    ".ini",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".tsv",
    ".sql",
    ".env.example",
    ".env",  # careful: do NOT snapshot actual .env by default
}

# Typical sensitive filenames we skip even if extension matches
SENSITIVE_NAME_PATTERNS = [
    r"^\.env$",
    r"^\.env\..+",
    r".*secret.*",
    r".*secrets.*",
    r".*credential.*",
    r".*credentials.*",
    r".*token.*",
    r".*apikey.*",
    r".*api[_-]?key.*",
    r".*password.*",
    r".*passwd.*",
    r".*private.*key.*",
]
SENSITIVE_NAME_RE = re.compile("|".join(SENSITIVE_NAME_PATTERNS), re.IGNORECASE)


@dataclass
class Limits:
    max_file_bytes: int
    max_total_bytes: int
    max_nodes_dot: int


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_excluded(path: Path, root: Path, exclude_dirs: set, exclude_files: set) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    if parts & exclude_dirs:
        return True
    if path.name in exclude_files:
        return True
    return False


def looks_sensitive(path: Path) -> bool:
    return bool(SENSITIVE_NAME_RE.match(path.name))


def is_probably_text_file(path: Path) -> bool:
    """
    Heuristic: read a small chunk and check for NUL bytes.
    """
    try:
        with path.open("rb") as f:
            chunk = f.read(4096)
        if b"\x00" in chunk:
            return False
        return True
    except Exception:
        return False


def gather_files(
    root: Path,
    include_ext: set,
    exclude_dirs: set,
    exclude_files: set,
) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if is_excluded(p, root, exclude_dirs, exclude_files):
            continue
        # Skip sensitive named files by default
        if looks_sensitive(p):
            continue

        # Extension logic:
        # - Allow exact include (including .env.example)
        # - Otherwise, require suffix in include_ext
        if p.name in include_ext:
            files.append(p)
            continue
        if p.suffix.lower() in include_ext:
            files.append(p)
            continue

    files.sort(key=lambda x: str(x).lower())
    return files


def build_tree_map(root: Path, files: List[Path], depth: int) -> Dict:
    """
    Build a nested dict representing directories/files.
    depth: max depth relative to root; <=0 means unlimited.
    """
    tree: Dict = {}

    for f in files:
        rel = f.relative_to(root)
        parts = rel.parts
        if depth > 0 and len(parts) > depth:
            # keep only up to depth; represent remainder collapsed
            parts = parts[:depth] + ("…",)
        node = tree
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            if is_last:
                node.setdefault("__files__", []).append(part)
            else:
                node = node.setdefault(part, {})
    return tree


def render_tree_ascii(tree: Dict, prefix: str = "") -> str:
    """
    Render nested dict to an ASCII tree.
    """
    lines: List[str] = []

    # directories (keys except __files__)
    dir_keys = sorted([k for k in tree.keys() if k != "__files__"], key=str.lower)
    files = sorted(tree.get("__files__", []), key=str.lower)

    entries = dir_keys + files
    for idx, name in enumerate(entries):
        is_last = idx == len(entries) - 1
        branch = "└── " if is_last else "├── "
        lines.append(prefix + branch + name)

        if name in tree:  # directory
            extension = "    " if is_last else "│   "
            lines.extend(render_tree_ascii(tree[name], prefix + extension).splitlines())

    return "\n".join(lines)


def write_tree_outputs(
    root: Path, files: List[Path], depth: int, make_md: bool = False
) -> Tuple[Path, Optional[Path]]:
    tree_map = build_tree_map(root, files, depth)
    ascii_tree = render_tree_ascii(tree_map)

    out_txt = root / "PROJECT_TREE.txt"
    header = [
        f"# Project tree (generated: {now_stamp()})",
        f"# Root: {root}",
        f"# Max depth: {'unlimited' if depth <= 0 else depth}",
        f"# Files included in scan: {len(files)}",
        "",
    ]

    out_txt.write_text("\n".join(header) + ascii_tree + "\n", encoding="utf-8")

    out_md = None
    if make_md:
        out_md = root / "PROJECT_TREE.md"
        md = "\n".join(header) + "```text\n" + ascii_tree + "\n```\n"
        out_md.write_text(md, encoding="utf-8")

    return out_txt, out_md


def write_snapshot(root: Path, files: List[Path], limits: Limits) -> Path:
    out = root / "PROJECT_SNAPSHOT.txt"
    total_written = 0

    with out.open("w", encoding="utf-8") as f:
        f.write(f"# Project snapshot (generated: {now_stamp()})\n")
        f.write(f"# Root: {root}\n")
        f.write(f"# Included files: {len(files)}\n")
        f.write(
            f"# Limits: max_file_bytes={limits.max_file_bytes}, max_total_bytes={limits.max_total_bytes}\n\n"
        )

        for p in files:
            rel = p.relative_to(root)

            # stop if total exceeds limit
            if total_written >= limits.max_total_bytes:
                f.write("\n" + "=" * 80 + "\n")
                f.write("SNAPSHOT TRUNCATED: max_total_bytes limit reached.\n")
                f.write("=" * 80 + "\n")
                break

            # skip if likely binary
            if not is_probably_text_file(p):
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"FILE: {rel}\n")
                f.write("=" * 80 + "\n")
                f.write("<<SKIPPED: looks like a binary/non-text file>>\n\n")
                continue

            size = p.stat().st_size
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"FILE: {rel}\n")
            f.write(f"SIZE: {size} bytes\n")
            f.write("=" * 80 + "\n")

            if size > limits.max_file_bytes:
                f.write(
                    f"<<SKIPPED: file exceeds max_file_bytes ({limits.max_file_bytes})>>\n\n"
                )
                continue

            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                f.write(f"<<ERROR reading file: {e}>>\n\n")
                continue

            # write content and update total
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")
            total_written += len(content.encode("utf-8", errors="replace"))

    return out


def make_dot_graph(root: Path, files: List[Path], depth: int, max_nodes: int) -> Path:
    """
    Create a Graphviz DOT representation of the folder structure.
    NOTE: For very large projects, DOT can be huge; max_nodes limits output.
    """
    out_dot = root / "PROJECT_TREE.dot"

    def node_id(path_str: str) -> str:
        # safe identifier
        return "n" + re.sub(r"[^a-zA-Z0-9_]", "_", path_str)

    nodes: List[str] = []
    edges: List[str] = []
    seen = set()

    # Add root
    root_label = root.name if root.name else str(root)
    root_id = node_id(".")
    nodes.append(f'{root_id} [label="{root_label}", shape=folder, fontsize=12];')
    seen.add(".")

    count = 1

    for p in files:
        rel = p.relative_to(root)
        parts = rel.parts
        if depth > 0 and len(parts) > depth:
            parts = parts[:depth] + ("…",)

        # Build all intermediate directories
        acc = []
        parent_key = "."
        parent_id = root_id

        for i, part in enumerate(parts):
            acc.append(part)
            key = "/".join(acc)
            is_last = i == len(parts) - 1

            if key not in seen:
                if count >= max_nodes:
                    break
                label = part
                shape = "note" if is_last else "folder"
                nodes.append(
                    f'{node_id(key)} [label="{label}", shape={shape}, fontsize=10];'
                )
                seen.add(key)
                count += 1

            # edge from parent to current
            edge = (parent_key, key)
            if edge not in seen:
                edges.append(f"{node_id(parent_key)} -> {node_id(key)};")
                seen.add(edge)

            parent_key = key
            parent_id = node_id(key)

        if count >= max_nodes:
            break

    with out_dot.open("w", encoding="utf-8") as f:
        f.write("digraph ProjectTree {\n")
        f.write("  rankdir=LR;\n")
        f.write('  node [fontname="Arial"];\n')
        f.write("  " + "\n  ".join(nodes) + "\n")
        f.write("  " + "\n  ".join(edges) + "\n")
        if count >= max_nodes:
            f.write(f'  {root_id} -> {node_id("__TRUNCATED__")};\n')
            f.write(
                f'  {node_id("__TRUNCATED__")} [label="TRUNCATED (max_nodes reached)", shape=box, color=red];\n'
            )
        f.write("}\n")

    return out_dot


def try_render_png(dot_file: Path) -> Optional[Path]:
    """
    If Graphviz is installed, create a PNG from DOT.
    """
    dot_exe = shutil.which("dot")
    if not dot_exe:
        return None
    out_png = dot_file.with_suffix(".png")
    try:
        subprocess.run(
            [dot_exe, "-Tpng", str(dot_file), "-o", str(out_png)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return out_png
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate project tree and snapshot from a project root."
    )
    p.add_argument(
        "--root", default=".", help="Project root folder (default: current directory)."
    )
    p.add_argument(
        "--depth", type=int, default=0, help="Max depth for tree/dot (0 = unlimited)."
    )
    p.add_argument(
        "--include-ext",
        nargs="*",
        default=None,
        help="Extensions to include (e.g. .py .toml .md). If omitted, defaults are used.",
    )
    p.add_argument(
        "--exclude-dir",
        nargs="*",
        default=None,
        help="Additional directory names to exclude (e.g. data logs).",
    )
    p.add_argument(
        "--exclude-file",
        nargs="*",
        default=None,
        help="Additional filenames to exclude.",
    )
    p.add_argument(
        "--max-file-bytes",
        type=int,
        default=300_000,
        help="Max file size to include in snapshot (bytes). Larger files are skipped.",
    )
    p.add_argument(
        "--max-total-bytes",
        type=int,
        default=6_000_000,
        help="Max total snapshot size (bytes). Snapshot truncates after reaching this.",
    )
    p.add_argument(
        "--dot", action="store_true", help="Also generate PROJECT_TREE.dot (Graphviz)."
    )
    p.add_argument(
        "--png",
        action="store_true",
        help="If Graphviz is available, render PROJECT_TREE.png too.",
    )
    p.add_argument(
        "--max-nodes-dot",
        type=int,
        default=2500,
        help="Max nodes in DOT output to avoid huge graphs.",
    )
    p.add_argument(
        "--md",
        action="store_true",
        help="Also generate PROJECT_TREE.md (Markdown version of the tree).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    include_ext = set(DEFAULT_INCLUDE_EXT)
    if args.include_ext is not None and len(args.include_ext) > 0:
        # normalize and accept both ".py" and "py"
        include_ext = set()
        for x in args.include_ext:
            x = x.strip()
            if not x:
                continue
            if not x.startswith(".") and x != ".env.example":
                x = "." + x
            include_ext.add(x)

    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)
    if args.exclude_dir:
        exclude_dirs.update({d.strip() for d in args.exclude_dir if d.strip()})

    exclude_files = set(DEFAULT_EXCLUDE_FILES)
    if args.exclude_file:
        exclude_files.update({f.strip() for f in args.exclude_file if f.strip()})

    limits = Limits(
        max_file_bytes=int(args.max_file_bytes),
        max_total_bytes=int(args.max_total_bytes),
        max_nodes_dot=int(args.max_nodes_dot),
    )

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Generating tree + snapshot @ {now_stamp()}")
    print(f"[INFO] Include extensions: {sorted(include_ext)}")
    print(f"[INFO] Exclude dirs: {sorted(exclude_dirs)}")

    files = gather_files(root, include_ext, exclude_dirs, exclude_files)
    print(f"[INFO] Files gathered: {len(files)}")

    tree_txt, tree_md = write_tree_outputs(
        root, files, depth=int(args.depth), make_md=bool(args.md)
    )
    if tree_md:
        print(f"[OK] Tree written: {tree_txt.name}, {tree_md.name}")
    else:
        print(f"[OK] Tree written: {tree_txt.name}")

    snapshot = write_snapshot(root, files, limits)
    print(f"[OK] Snapshot written: {snapshot.name}")

    dot_file = None
    png_file = None
    if args.dot or args.png:
        dot_file = make_dot_graph(
            root, files, depth=int(args.depth), max_nodes=limits.max_nodes_dot
        )
        print(f"[OK] DOT written: {dot_file.name}")
    if args.png:
        if dot_file is None:
            dot_file = root / "PROJECT_TREE.dot"
        png_file = try_render_png(dot_file)
        if png_file:
            print(f"[OK] PNG rendered: {png_file.name}")
        else:
            print(
                "[WARN] Could not render PNG. Graphviz 'dot' not found or rendering failed."
            )
            print(
                "       Install Graphviz or use the .dot file elsewhere (e.g., web graphviz viewer)."
            )

    print("[DONE] Outputs created in project root:")
    print(f"       - {tree_txt.name}")
    if tree_md:
        print(f"       - {tree_md.name}")
    print(f"       - {snapshot.name}")
    if dot_file:
        print(f"       - {dot_file.name}")
    if png_file:
        print(f"       - {png_file.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
