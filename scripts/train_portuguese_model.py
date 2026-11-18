#!/usr/bin/env python3
"""Reproducible symbolic corpus ingestion, integer training, consolidation and evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cli', type=Path, default=Path('build_rocm/cognicore'))
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--clauses', type=int, default=20)
    parser.add_argument('--max-vocab', type=int, default=16000)
    parser.add_argument('--context-stride', type=int, default=4)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--resume-policy', action='store_true')
    args = parser.parse_args()
    args.cli = args.cli.resolve()
    if not args.cli.is_file(): parser.error('build the CLI first')
    if args.model.exists() and any(args.model.iterdir()) and not args.resume_policy:
        parser.error('model directory is not empty; use a new path or --resume-policy')
    args.model.mkdir(parents=True, exist_ok=True)
    logs = args.model / 'training_logs'
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='4')
    stages = []

    def run(name, *arguments, gpu=False):
        command = [str(args.cli), *map(str, arguments)] + (['--use-gpu'] if gpu and args.use_gpu else [])
        start = time.monotonic()
        print(f'[{name}] starting', flush=True)
        with (logs / (name + '.log')).open('w') as log:
            result = subprocess.run(command, text=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        text = (logs / (name + '.log')).read_text()
        print(text[-3000:], flush=True)
        if result.returncode: raise RuntimeError(f'{name} failed: {result.returncode}')
        stages.append({'stage': name, 'seconds': round(time.monotonic() - start, 3), 'command': command})

    if not args.resume_policy:
        run('style', 'ingest-books', '--input', args.data / 'style', '--model', args.model,
            '--style-only', '--max-vocab', str(args.max_vocab),
            '--context-stride', str(args.context_stride))
        run('facts', 'ingest-books', '--input', args.data / 'facts.conllu', '--model', args.model,
            '--parser', 'conllu', gpu=True)
    run('policy', 'train-dialogue', '--input', args.data / 'dialogue_train.jsonl', '--model', args.model,
        '--epochs', args.epochs, '--clauses', args.clauses, '--balance', gpu=True)
    run('fluency', 'train-realizer', '--model', args.model, gpu=True)
    run('consolidate', 'consolidate', '--model', args.model)
    run('heldout', 'evaluate-dialogue', '--input', args.data / 'dialogue_eval.jsonl', '--model', args.model,
        '--out', args.model / 'heldout_metrics.json')
    run('probes', 'evaluate-dialogue', '--input', args.data / 'dialogue_probes.jsonl', '--model', args.model,
        '--out', args.model / 'probe_metrics.json')
    run('stats', 'stats', '--model', args.model)
    shutil.copy2(args.data / 'manifest.json', args.model / 'corpus_manifest.json')
    report = {'architecture': 'symbolic; no neural network or backpropagation',
              'stages': stages, 'epochs': args.epochs, 'clauses_per_class': args.clauses,
              'max_vocab': args.max_vocab, 'context_stride': args.context_stride,
              'gpu_requested': args.use_gpu, 'binary': str(args.cli)}
    (args.model / 'training_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
