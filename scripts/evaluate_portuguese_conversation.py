#!/usr/bin/env python3
"""Evaluate a real CPU chat session against a disposable copy of the trained model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli',type=Path,default=Path('build/cognicore'))
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    inputs=['Olá!', 'Quem é você?', 'Tudo bem?', 'O que você pode fazer?',
            'Obrigado pela ajuda.', 'Por que o atrito pode derreter gelo?', 'O que é a SDM?',
            'O vento esfria a água.', 'O que eu disse?', 'O que o vento faz?',
            'O que você sabe sobre teletransporte?', 'Tchau!']
    with tempfile.TemporaryDirectory(prefix='cognicore_eval_pt_') as temp:
        copy=Path(temp)/'model'
        shutil.copytree(args.model,copy,ignore=shutil.ignore_patterns('training_logs','*.log','*.json','*.md'))
        command=[str(args.cli.resolve()),'chat','--model',str(copy),'--debug']
        rss_file=Path(temp)/'rss.txt'
        if Path('/usr/bin/time').exists():command=['/usr/bin/time','-f','%M','-o',str(rss_file)]+command
        start=time.monotonic()
        result=subprocess.run(command,input='\n'.join(inputs+['sair','']),capture_output=True,text=True,
                              env=dict(os.environ,OMP_NUM_THREADS='4'),check=True)
        elapsed=time.monotonic()-start
        answers=[line.split('CogniCore> ',1)[1] for line in result.stdout.splitlines() if 'CogniCore> ' in line]
        assert len(answers)==len(inputs),(len(answers),result.stdout)
        checks={
            'identity': 'cognicore' in answers[1].casefold(),
            'thanks': any(word in answers[4].casefold() for word in ['de nada','por nada']),
            'multihop': 'gera' in answers[5] and 'derrete' in answers[5],
            'one_shot': 'esfria' in answers[9] and 'água' in answers[9],
            'episodic_memory': answers[8]==inputs[7],
            'farewell': 'até' in answers[11].casefold(),
            'unknown_topic': 'teletransporte' in answers[10].casefold() and 'suficientes' in answers[10].casefold(),
        }
        rss=int(rss_file.read_text().strip())*1024 if rss_file.exists() else None
        report={'inputs':inputs,'answers':answers,'checks':checks,'passed':sum(checks.values()),
                'total_checks':len(checks),'session_seconds_including_persistence':elapsed,'peak_rss_bytes':rss,
                'transcript':result.stdout,'cpu_only_binary':str(args.cli.resolve())}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='transcript'},ensure_ascii=False),flush=True)
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
