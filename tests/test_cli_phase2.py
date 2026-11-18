#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

cli = str(Path(sys.argv[1]).resolve())

def run(*args, input=None):
    result = subprocess.run([cli, *map(str, args)], input=input, text=True, capture_output=True)
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result.stdout

with tempfile.TemporaryDirectory(prefix='cognicore_cli_') as directory:
    root = Path(directory)
    repo = root / 'repo'
    repo.mkdir()
    (repo / 'mini.py').write_text('def identity(value: str) -> str:\n    return value\n\ndef count(value: str) -> int:\n    return len(value)\n')
    model = root / 'model'
    book = root / 'book.txt'
    book.write_text('O calor derrete gelo.\nO fogo não pode gerar frio.\n')
    run('ingest-books', '--model', model, '--input', book)
    unsupported = subprocess.run([cli, 'chat', '--model', str(model), '--realizer', 'neural'],
                                 input='sair\n', text=True, capture_output=True)
    assert unsupported.returncode != 0 and 'symbolic' in unsupported.stderr
    run('ingest-code', '--model', model, '--repo', repo)
    output = run('code', '--model', model, '--workspace', repo, '--lang', 'py', '--task', 'identity e count')
    script = repo / 'generated.py'
    script.write_text(output + '\nassert synthesized_pipeline("hello") == 5\n')
    subprocess.run([sys.executable, str(script)], check=True)
    pairs = root / 'pairs.jsonl'
    run('export-realizer-dataset', '--model', model, '--out', pairs)
    rows = [json.loads(row) for row in pairs.read_text().splitlines()]
    assert len(rows) == 2 and any('<NEG: true>' in row['plan'] for row in rows)
    assert any('não pode gerar' in row['text'] for row in rows)
    result = run('chat', '--model', model, '--workspace', repo, '--lang', 'py', '--debug',
                 input='O atrito gera calor.\nPor que o atrito causa gelo?\nGere uma função para combinar identity e count\nsair\n')
    for route in ['FACT_UPDATE', 'CAUSAL_REASONING', 'CODE_GENERATE_OR_PATCH']:
        assert route in result, result
    assert 'identity(input)' in result and 'count(step1)' in result
    history = [json.loads(line) for line in (repo / '.cognicore/conversation.jsonl').read_text().splitlines()]
    assert len(history) == 3
    resumed = run('chat', '--model', model, '--workspace', repo, '--debug', input='Por que o atrito causa gelo?\nsair\n')
    assert 'atrito -[gerar]-> calor' in resumed and 'gelo' in resumed, resumed

    compiler = shutil.which('c++')
    if compiler:
        library = repo / 'mini.cpp'
        library.write_text('#include <string>\nstd::string read_text(const std::string& path) { return path; }\nstd::size_t size_text(const std::string& text) { return text.size(); }\n')
        cpp = run('code', '--model', model, '--workspace', repo, '--lang', 'cpp', '--task', 'read_text e size_text')
        generated = repo / 'generated.cpp'
        generated.write_text(cpp + '\nint main() { return synthesized_pipeline("hello") == 5 ? 0 : 1; }\n')
        binary = root / 'generated'
        subprocess.run([compiler, '-std=c++20', str(generated), str(library), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
