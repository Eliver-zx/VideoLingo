#!/usr/bin/env python3
"""Sync modular HTML views into index.html."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
INDEX_OUT = ROOT / "index.html"

HTML_FILES = [
    "view-workbench.html",
    "view-burn.html",
    "view-glossary.html",
    "view-settings.html",
]

FAVICON_DATA_URI = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 28 28'%3E"
    "%3Crect x='1' y='1' width='26' height='26' rx='6' fill='%231f1b18'/%3E"
    "%3Crect x='5.5' y='11' width='2.2' height='6' rx='1.1' fill='%23c4501b'/%3E"
    "%3Crect x='9.3' y='7.5' width='2.2' height='13' rx='1.1' fill='%23c4501b'/%3E"
    "%3Crect x='13.1' y='10' width='2.2' height='8' rx='1.1' fill='%23c4501b'/%3E"
    "%3Crect x='17.4' y='11.4' width='5.6' height='2.2' rx='1.1' fill='%23ffffff'/%3E"
    "%3Crect x='17.4' y='15.2' width='3.8' height='2.2' rx='1.1' fill='%23ffffff' fill-opacity='0.6'/%3E"
    "%3C/svg%3E"
)


def build_index_html() -> str:
    body_combined = "\n\n  ".join(
        (ROOT / f).read_text(encoding="utf-8").strip() for f in HTML_FILES
    )
    return f"""<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VideoLingo — 本地多任务字幕与配音工作台</title>
  <link rel="icon" type="image/svg+xml" href="{FAVICON_DATA_URI}">
  <link rel="stylesheet" href="./css/base.css">
  <link rel="stylesheet" href="./css/layout-rail.css">
  <link rel="stylesheet" href="./css/intake-modal.css">
  <link rel="stylesheet" href="./css/stream-grid.css">
  <link rel="stylesheet" href="./css/terms-console.css">
  <link rel="stylesheet" href="./css/burn-glossary.css">
  <link rel="stylesheet" href="./css/settings-dialog.css">
</head>
<body>
  {body_combined}
  <script type="module" src="./js/main.js"></script>
</body>
</html>
"""


if __name__ == "__main__":
    INDEX_OUT.write_text(build_index_html(), encoding="utf-8")
    print(f"Synced {INDEX_OUT}")
