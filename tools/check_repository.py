"""Check Python syntax and repository-local Markdown links without model imports."""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
for source in ROOT.rglob('*.py'):
    if any(part in {'.git', '.venv', '__pycache__'} for part in source.parts):
        continue
    ast.parse(source.read_text(encoding='utf-8'), filename=str(source.relative_to(ROOT)))
for document in ROOT.rglob('*.md'):
    for link in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
        if link.startswith(('http://', 'https://', 'mailto:', '#')):
            continue
        path = link.split('#', 1)[0]
        if not (document.parent / path).is_file():
            raise SystemExit(f'Broken local link: {document.relative_to(ROOT)} -> {link}')
print('Python syntax and repository-local documentation links passed.')
