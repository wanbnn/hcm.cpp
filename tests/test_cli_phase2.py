#!/usr/bin/env python3
import ast
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

# Architecture regression: no neural training implementation or weight artifact.
project = Path(__file__).resolve().parents[1]
for script in (project / 'scripts').glob('*.py'):
    for node in ast.walk(ast.parse(script.read_text())):
        imported = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                    [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
        assert not any(name.split('.')[0] in {'torch', 'transformers', 'tensorflow'} for name in imported), script
assert not (project / 'scripts/train_micro_decoder.py').exists()
assert not list((project / 'src').rglob('*neural_mouth*'))
assert not list((project / 'include').rglob('*neural_mouth*'))

with tempfile.TemporaryDirectory(prefix='cognicore_cli_') as directory:
    root = Path(directory)
    repo = root / 'repo'
    repo.mkdir()
    (repo / 'mini.py').write_text('def identity(value: str) -> str:\n    return value\n\ndef count(value: str) -> int:\n    return len(value)\n')
    model = root / 'model'
    book = root / 'book.txt'
    book.write_text('O calor derrete gelo.\nO fogo não pode gerar frio.\n')
    fiction = root / 'fiction.txt'
    fiction.write_text('O dragão gera magia. A fada produz encantos.\n')
    style_model = root / 'style_model'
    style_log = run('ingest-books', '--model', style_model, '--input', fiction, '--style-only', '--max-vocab', 12, '--context-stride', 2)
    assert 'Triplas extraidas: 0' in style_log
    style_stats = run('stats', '--model', style_model)
    assert '0 nos, 0 arestas' in style_stats
    dialogue = root / 'escaped.jsonl'
    dialogue.write_text(json.dumps({'text': 'A chuva gera água.\n"Olá"', 'intent': 'ANSWER_FACT'}, ensure_ascii=True) + '\n')
    run('train-dialogue', '--model', style_model, '--input', dialogue, '--epochs', 1, '--clauses', 2, '--balance')
    evaluated = json.loads(run('evaluate-dialogue', '--model', style_model, '--input', dialogue))
    assert evaluated['samples'] == 1 and 0 <= evaluated['accuracy'] <= 1
    run('train-realizer', '--model', style_model)
    assert (style_model / 'fluency.bin').is_file()
    unsupported = subprocess.run([cli, 'chat', '--model', str(model), '--realizer', 'neural'],
                                 input='sair\n', text=True, capture_output=True)
    assert unsupported.returncode != 0
    run('ingest-books', '--model', model, '--input', book)
    unsupported = subprocess.run([cli, 'chat', '--model', str(model), '--realizer', 'neural'],
                                 input='sair\n', text=True, capture_output=True)
    assert unsupported.returncode != 0 and 'strict' in unsupported.stderr
    run('ingest-code', '--model', model, '--repo', repo)
    output = run('code', '--model', model, '--workspace', repo, '--lang', 'py', '--task', 'identity e count')
    script = repo / 'generated.py'
    script.write_text(output + '\nassert synthesized_pipeline("hello") == 5\n')
    subprocess.run([sys.executable, str(script)], check=True)
    run('train-realizer', '--model', model)
    assert (model / 'fluency.bin').is_file()
    assert (model / 'graph_mmap.bin').is_file()
    calibration = json.loads(run('evaluate-fluency', '--model', model, '--input', book))
    assert calibration['bytes'] > 0 and calibration['bits_per_byte'] > 0
    assert calibration['trained'] is True
    unsupported_export = subprocess.run([cli, 'export-realizer-dataset', '--model', str(model)], capture_output=True)
    assert unsupported_export.returncode != 0
    result = run('chat', '--model', model, '--workspace', repo, '--lang', 'py', '--debug',
                 input='O atrito gera calor.\nPor que o atrito causa gelo?\nGere uma função para combinar identity e count\nsair\n')
    for route in ['FACT_UPDATE', 'CAUSAL_REASONING', 'CODE_GENERATE_OR_PATCH']:
        assert route in result, result
    assert 'identity(input)' in result and 'count(step1)' in result
    history = [json.loads(line) for line in (repo / '.cognicore/conversation.jsonl').read_text().splitlines()]
    assert len(history) == 3
    resumed = run('chat', '--model', model, '--workspace', repo, '--debug', input='Por que o atrito causa gelo?\nsair\n')
    assert 'atrito --gerar--> calor' in resumed and 'gelo' in resumed, resumed

    compiler = shutil.which('c++')
    if compiler:
        library = repo / 'mini.cpp'
        library.write_text('#include <string>\nstd::string read_text(const std::string& path) { return path; }\nstd::size_t size_text(const std::string& text) { return text.size(); }\nstd::size_t twice(std::size_t n) { return n * 2; }\n')
        cpp = run('code', '--model', model, '--workspace', repo, '--lang', 'cpp', '--task', 'read_text e size_text e twice')
        generated = repo / 'generated.cpp'
        generated.write_text(cpp + '\nint main() { return synthesized_pipeline("hello") == 10 ? 0 : 1; }\n')
        binary = root / 'generated'
        subprocess.run([compiler, '-std=c++20', str(generated), str(library), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
