"""Audit publishable text files and optionally create an allowlisted ZIP."""
import argparse
import ast
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','.venv','venv','__pycache__','.pytest_cache','data','results','private','weights','build','dist'}
SUFFIXES={'.py','.md','.json','.toml','.txt','.yml','.yaml'}
NAMES={'.gitignore','.gitattributes','LICENSE','NOTICE'}
FIGURE_ASSETS={
    'assets/ais_draft_uncertainty.jpg',
    'assets/vseim_framework.jpg',
    'assets/movr_rhsg_architecture.jpg',
}


def audit():
    findings=[];files=[]
    patterns={
        'personal_absolute_path':re.compile(r'(?:[A-Za-z]:[\\/](?:Users|anaconda|Desktop)[\\/]|/(?:home|Users)/)[^\s\"\']+',re.I),
        'github_token':re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
        'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        'assigned_secret':re.compile(r'(?:api_key|access_token|secret_key)\s*[:=]\s*[\"\'][^\"\']{12,}[\"\']',re.I),
    }
    for p in sorted(ROOT.rglob('*')):
        relative=p.relative_to(ROOT)
        if any(part in EXCLUDED or part.endswith('.egg-info') for part in relative.parts):continue
        if p.is_symlink():findings.append((str(relative),'symlink'));continue
        if not p.is_file():continue
        if p.name.endswith('.local.json') or p.name.startswith('.env'):
            findings.append((str(relative),'private_local_configuration'));continue
        if relative.as_posix() in FIGURE_ASSETS:
            from PIL import Image
            try:
                with Image.open(p) as figure:
                    if figure.format != 'JPEG':raise ValueError('Expected JPEG figure')
                    figure.verify()
            except (OSError,ValueError):findings.append((str(relative),'invalid_figure_asset'))
            files.append(p);continue
        if p.suffix not in SUFFIXES and p.name not in NAMES:
            findings.append((str(relative),'nonallowlisted_file'));continue
        text=p.read_text(encoding='utf-8-sig')
        for name,pattern in patterns.items():
            if pattern.search(text):findings.append((str(relative),name))
        if p.suffix=='.py':
            try:ast.parse(text)
            except SyntaxError:findings.append((str(relative),'python_syntax_error'))
        if p.suffix=='.json':
            try:json.loads(text)
            except ValueError:findings.append((str(relative),'invalid_json'))
        files.append(p)
    return files,findings


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path);args=ap.parse_args()
    files,findings=audit()
    if findings:
        for path,issue in findings:print(f'{issue}: {path}')
        return 1
    print(f'PASS: {len(files)} publishable files; no configured privacy patterns found')
    if args.zip:
        target=args.zip.resolve()
        if target.is_relative_to(ROOT):raise ValueError('Place the release archive outside the repository directory')
        if target.exists():raise FileExistsError(target)
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,Path('vseim-research')/p.relative_to(ROOT))
        print(target)
    return 0


if __name__=='__main__':raise SystemExit(main())
