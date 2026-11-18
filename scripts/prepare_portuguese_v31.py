#!/usr/bin/env python3
"""Assemble the conversa-pt-fluent-v3.1 training dataset.

Starts from the v3 dataset (``public_pt_v1``) and adds the public text fetched
by ``fetch_public_pt_corpus.py`` into the same ``style/`` tree. Duplicate files
are moved out of ``style/`` so the symbolical models do not observe the same
text twice. The merged manifest keeps attribution and license notes.
"""
import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

COPY_FILES = ['dialogue_train.jsonl', 'dialogue_eval.jsonl',
              'dialogue_probes.jsonl', 'facts.conllu']
SOURCE_GROUPS = {
    'wikipedia_pt': 'Wikipedia em português (CC BY-SA 4.0)',
    'gutenberg': 'Project Gutenberg (domínio público nos EUA)',
    'pdfs': 'PDFs educacionais públicos',
    'v3_base': 'corpus conversacional do conversa-pt-fluent-v3',
    'vocabulary_seed': 'amostra balanceada dos grupos acima (semente de vocabulário/contexto)',
}
SEED_NAME = '00_vocabulary_seed.txt'


def build_vocabulary_seed(style):
    """Deterministic balanced excerpt so the capped codebook sees every register."""
    seed_path = style / SEED_NAME
    if seed_path.exists():
        seed_path.unlink()
    groups, targets = defaultdict(list), {'v3_base': 7, 'wikipedia_pt': 120,
                                          'gutenberg': 30, 'pdfs': 1}
    words = {'v3_base': 3000, 'wikipedia_pt': 400, 'gutenberg': 800, 'pdfs': 1200}
    for path in sorted(style.rglob('*.txt')):
        relative = path.relative_to(style)
        groups[group_of(relative)].append(path)
    parts = []
    for group in ('v3_base', 'wikipedia_pt', 'gutenberg', 'pdfs'):
        paths = groups.get(group, [])
        target = targets[group]
        if not paths:
            continue
        if len(paths) > target:
            indexes = [round(i * (len(paths) - 1) / max(1, target - 1))
                       for i in range(target)]
            paths = [paths[i] for i in dict.fromkeys(indexes)]
        for path in paths:
            tokens = path.read_text(encoding='utf-8', errors='replace').split()
            count = words[group]
            start = len(tokens) // 3 if group == 'gutenberg' else 0
            excerpt = ' '.join(tokens[start:start + count])
            if excerpt:
                parts.append(excerpt)
    seed_path.write_text('\n\n'.join(parts) + '\n', encoding='utf-8')
    return seed_path


def link_or_copy(source, target):
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return True


def group_of(relative):
    if relative.name == SEED_NAME:
        return 'vocabulary_seed'
    first = relative.parts[0] if len(relative.parts) > 1 else ''
    if first in ('wikipedia_pt', 'gutenberg', 'pdfs'):
        return first
    return 'v3_base'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True,
                        help='dataset used to train conversa-pt-fluent-v3')
    parser.add_argument('--out', type=Path, required=True,
                        help='dataset with newly fetched sources in style/ and raw/')
    args = parser.parse_args()
    out_style = args.out / 'style'
    out_style.mkdir(parents=True, exist_ok=True)

    copied = {'auxiliary': [], 'base_style': 0}
    for name in COPY_FILES:
        if link_or_copy(args.base / name, args.out / name):
            copied['auxiliary'].append(name)
    for source in sorted((args.base / 'style').rglob('*.txt')):
        relative = source.relative_to(args.base / 'style')
        if link_or_copy(source, out_style / relative):
            copied['base_style'] += 1

    seed_path = build_vocabulary_seed(out_style)
    inventory, duplicates = [], []
    hashes = {}
    for path in sorted(out_style.rglob('*.txt')):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        relative = path.relative_to(out_style)
        if digest in hashes:
            duplicate = args.out / 'duplicates' / relative
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, duplicate)
            duplicates.append({'file': str(relative), 'duplicate_of': hashes[digest],
                               'moved_to': str(duplicate.relative_to(args.out))})
            continue
        hashes[digest] = str(relative)
        inventory.append({'file': str(relative), 'group': group_of(relative),
                          'characters': len(data), 'sha256': digest})

    totals = defaultdict(lambda: {'files': 0, 'characters': 0})
    for row in inventory:
        totals[row['group']]['files'] += 1
        totals[row['group']]['characters'] += row['characters']

    base_manifest = json.loads((args.base / 'manifest.json').read_text(encoding='utf-8'))
    source_manifest_path = args.out / 'sources_manifest.json'
    source_manifest = (json.loads(source_manifest_path.read_text(encoding='utf-8'))
                       if source_manifest_path.exists() else {'sources': {}})

    manifest = dict(base_manifest)
    manifest['created_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['version'] = 'conversa-pt-fluent-v3.1'
    manifest['dataset_scope'] = ('Portuguese conversation and public-domain/CC text: '
                                 'dialogue intents from v3 plus Wikipedia, Gutenberg books '
                                 'and public educational PDFs for vocabulary, style and context')
    manifest['style_groups'] = {group: description for group, description
                                in SOURCE_GROUPS.items()}
    manifest['style_inventory'] = inventory
    manifest['style_totals'] = dict(totals)
    manifest['style_totals_all'] = {
        'files': len(inventory),
        'characters': sum(row['characters'] for row in inventory),
    }
    manifest['added_from_v3_base'] = copied
    manifest['duplicates_excluded'] = duplicates
    manifest['public_text_sources'] = source_manifest.get('sources', {})
    manifest['public_text_failures'] = {
        key: value for key, value in source_manifest.items() if key.endswith('_failures')}
    manifest['attribution'] = {
        'wikipedia_pt': 'Texto da Wikipédia em português, licença CC BY-SA 4.0; '
                        'títulos, revisões e URLs no manifesto.',
        'gutenberg': 'Obras de domínio público nos Estados Unidos, obtidas do espelho '
                     'pglaf do Project Gutenberg; a marca Project Gutenberg não é reivindicada.',
        'pdfs': 'Materiais educacionais publicamente acessíveis; direitos de '
                'redistribuição não foram afirmados pelo projeto.',
    }
    manifest['style_characters'] = manifest['style_totals_all']['characters']
    (args.out / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(json.dumps({'files': len(inventory), 'characters': sum(
        row['characters'] for row in inventory), 'groups': dict(totals),
        'duplicates': len(duplicates)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
