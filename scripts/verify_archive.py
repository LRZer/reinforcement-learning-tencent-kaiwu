"""Verify original-file hashes and parse all archived Python/TOML sources."""
import ast
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'data/metadata/original_files.json').read_text(encoding='utf-8'))
    preserved = 0
    for item in manifest['files']:
        if not item['repository_path']:
            continue
        p = ROOT/item['repository_path']
        assert p.is_file(), f'Missing archived file: {p}'
        assert len(p.read_bytes()) == item['bytes'], f'Size mismatch: {p}'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], f'Hash mismatch: {p}'
        preserved += 1
    sources = list(ROOT.glob('agent_*/**/*.py')) + list(ROOT.glob('conf/*.py'))
    for p in sources:
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    configs = list(ROOT.glob('agent_*/**/*.toml')) + list(ROOT.glob('conf/*.toml'))
    for p in configs:
        tomllib.loads(p.read_text(encoding='utf-8-sig'))
    print(json.dumps({'original_files': len(manifest['files']), 'preserved_files_verified': preserved,
                      'excluded_generated_or_signing_files': len(manifest['files'])-preserved,
                      'python_sources_parsed': len(sources), 'toml_configs_parsed': len(configs)}, indent=2))


if __name__ == '__main__':
    main()
