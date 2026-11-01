#!/usr/bin/env python3
"""Sync modular HTML views into index.html."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

html_files = ["view-workbench.html", "view-glossary.html", "view-settings.html"]
body_combined = "\n\n  ".join(
    (ROOT / f).read_text(encoding="utf-8").strip() for f in html_files
)

index_doc = f"""<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VideoLingo — 本地多任务字幕与配音后期工作台</title>
  <link rel="stylesheet" href="./base.css">
  <link rel="stylesheet" href="./workbench.css">
  <link rel="stylesheet" href="./views.css">
</head>
<body>
  {body_combined}
  <script src="./data.js"></script>
  <script src="./app.js"></script>
</body>
</html>
"""
(ROOT / "index.html").write_text(index_doc, encoding="utf-8")
print(f"Synced {ROOT / 'index.html'}")
