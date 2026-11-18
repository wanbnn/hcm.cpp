#!/usr/bin/env python3
"""CogniCore symbolic benchmarks.

Categories are inspired by public benchmark formats (GPQA-style science,
HLE-style hard reasoning, GSM8K/MathVista-style arithmetic, HumanEval-style
code generation and SWE-bench-style patching). They are small, project-owned
diagnostic sets, not the original public benchmarks and not a claim of
comparable scores.

No neural model is used. All answers come from VSA/SDM/Tsetlin/graph/DOP.
"""
import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCIENCE = [
    ("O que o atrito gera?", ["calor"]),
    ("O que derrete o gelo?", ["calor"]),
    ("O gelo é o quê?", ["sólido", "solido"]),
    ("O que a fotossíntese produz?", ["glicose"]),
    ("O que a gravidade provoca?", ["queda"]),
]

HARD_SEQUENCE = [
    ("Se todos os motores aquecem então a bomba falha.", None),
    ("Se todos os motores aquecem, o que acontece?", ["bomba"]),
    ("O calor, que derrete gelo, aquece a água.", None),
    ("O que o calor aquece?", ["água", "agua"]),
    ("Quando o motor aquece, a bomba falha.", None),
    ("Se o motor aquece, o que acontece?", ["bomba"]),
]

MATH = [
    ("Quanto é 12 + 8?", ["20"]),
    ("Quanto é 7 * 6?", ["42"]),
    ("João tem 10 figurinhas e compra mais 5. Quantas tem?", ["15"]),
    ("Maria tinha 20 balas e perdeu 6. Quantas balas restam?", ["14"]),
    ("Cada livro custa 15 reais e comprei 3. Quanto é o total?", ["45"]),
]

FLUENCY = [
    ("Por que o atrito causa gelo?", ["atrito", "gelo"]),
    ("Explique o que acontece quando a chuva causa umidade.", ["umidade"]),
    ("O que o calor faz com o gelo?", ["gelo"]),
]

CODE_TASKS = [
    ("identity e count", "hello", 5),
    ("double_value e increment", 3, 7),
]


def normalize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def matches(answer, expected):
    text = normalize(answer)
    return any(normalize(value) in text for value in expected)


def run(command, *, cwd=ROOT, input_text=None, timeout=120):
    started = time.monotonic()
    result = subprocess.run(
        [str(value) for value in command],
        cwd=str(cwd),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=dict(os.environ, OMP_NUM_THREADS="4"),
    )
    elapsed = (time.monotonic() - started) * 1000.0
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({command}): {result.stdout}\n{result.stderr}")
    return result.stdout, elapsed


def extract_answers(stdout):
    answers = []
    for line in stdout.splitlines():
        marker = "CogniCore> "
        if marker in line:
            answers.append(line.split(marker, 1)[1].strip())
    return answers


def chat_sequence(cli, model, prompts, realizer="fluent"):
    input_text = "\n".join(list(prompts) + ["sair"]) + "\n"
    stdout, elapsed = run([cli, "chat", "--model", model, "--realizer", realizer], input_text=input_text)
    answers = extract_answers(stdout)
    if len(answers) < len(prompts):
        answers += [""] * (len(prompts) - len(answers))
    return answers, elapsed


def category_result(name, details, elapsed):
    correct = sum(1 for item in details if item["correct"])
    return {
        "category": name,
        "total": len(details),
        "correct": correct,
        "accuracy": correct / len(details) if details else 0.0,
        "latency_ms": round(elapsed, 2),
        "details": details,
    }


def prepare_model(cli, source_model, out_dir):
    model = out_dir / "bench_model"
    if model.exists():
        shutil.rmtree(model)
    shutil.copytree(source_model, model)
    run([cli, "ingest-books", "--input", ROOT / "benchmarks/data/science_facts.txt", "--model", model])
    run([cli, "ingest-books", "--input", ROOT / "benchmarks/data/long_form_facts.txt", "--model", model])
    stdout, _ = run([cli, "train-realizer", "--model", model])
    return model, stdout


def prepare_workspace(out_dir):
    workspace = out_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "mathlib.py").write_text(
        "def identity(value: str) -> str:\n"
        "    return value\n\n"
        "def count(value: str) -> int:\n"
        "    return len(value)\n\n"
        "def double_value(value: int) -> int:\n"
        "    return value * 2\n\n"
        "def increment(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def load_items(path: str):\n"
        "    return [1, 2, 3]\n\n"
        "def process(path: str):\n"
        "    return load_items(path)\n"
    )
    return workspace


def run_science(cli, model):
    details, elapsed = [], 0.0
    for prompt, expected in SCIENCE:
        answers, cost = chat_sequence(cli, model, [prompt])
        elapsed += cost
        answer = answers[-1]
        details.append({"prompt": prompt, "answer": answer, "expected": expected,
                        "correct": matches(answer, expected)})
    return category_result("science_mcq_gpqa_style", details, elapsed)


def run_hard(cli, model):
    prompts = [prompt for prompt, _ in HARD_SEQUENCE]
    answers, elapsed = chat_sequence(cli, model, prompts)
    details = []
    for (prompt, expected), answer in zip(HARD_SEQUENCE, answers):
        if expected is None:
            continue
        details.append({"prompt": prompt, "answer": answer, "expected": expected,
                        "correct": matches(answer, expected)})
    return category_result("hard_reasoning_hle_style", details, elapsed)


def run_math(cli, model):
    prompts = [prompt for prompt, _ in MATH]
    answers, elapsed = chat_sequence(cli, model, prompts)
    details = []
    for (prompt, expected), answer in zip(MATH, answers):
        details.append({"prompt": prompt, "answer": answer, "expected": expected,
                        "correct": matches(answer, expected)})
    return category_result("math_gsm8k_style", details, elapsed)


def run_code(cli, model, workspace, out_dir):
    details, elapsed = [], 0.0
    for task, sample_input, expected in CODE_TASKS:
        stdout, cost = run([cli, "code", "--model", model, "--workspace", workspace,
                            "--lang", "py", "--task", task])
        elapsed += cost
        generated = workspace / ("generated_" + str(len(details)) + ".py")
        generated.write_text(stdout + f"\nassert synthesized_pipeline({sample_input!r}) == {expected}\n")
        try:
            _, execution = run([sys.executable, generated.name], cwd=workspace, timeout=30)
            elapsed += execution
            correct = True
            error = ""
        except Exception as exc:  # noqa: BLE001
            correct = False
            error = str(exc)
        details.append({"task": task, "code": stdout, "correct": correct, "error": error})
    return category_result("code_generation_humaneval_style", details, elapsed)


def run_patch(cli, model, workspace, out_dir):
    started = time.monotonic()
    stdout, cost = run([cli, "code", "--model", model, "--workspace", workspace,
                        "--lang", "py", "--task", "patch process"])
    patched = workspace / "patched.py"
    patched.write_text(stdout + "\nassert patched_process('x') == [1, 2, 3]\n")
    correct = False
    error = ""
    try:
        run([sys.executable, "patched.py"], cwd=workspace, timeout=30)
        correct = True
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    elapsed = (time.monotonic() - started) * 1000.0
    return category_result("patch_swebench_style", [
        {"task": "patch process", "code": stdout, "correct": correct, "error": error}
    ], elapsed)


def run_fluency(cli, model, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    details, elapsed = [], 0.0
    candidates = []
    for index, (prompt, expected) in enumerate(FLUENCY):
        strict_answers, strict_cost = chat_sequence(cli, model, [prompt], "strict")
        fluent_answers, fluent_cost = chat_sequence(cli, model, [prompt], "fluent")
        elapsed += strict_cost + fluent_cost
        strict = strict_answers[-1]
        fluent = fluent_answers[-1]
        coverage = matches(fluent, expected)
        details.append({"prompt": prompt, "strict": strict, "fluent": fluent,
                        "expected": expected, "fluent_coverage": coverage,
                        "novel_vs_strict": normalize(strict) != normalize(fluent),
                        "correct": coverage})
        candidates.append({"id": f"fluency_{index}", "prompt": prompt,
                           "system_a": strict, "system_b": fluent})
    sheet_candidates = out_dir / "fluency_candidates.jsonl"
    sheet = out_dir / "fluency_rating_sheet.jsonl"
    sheet_candidates.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in candidates) + "\n")
    run([sys.executable, ROOT / "scripts/evaluate_fluency_human.py", "generate",
         "--candidates", sheet_candidates, "--out", sheet])
    result = category_result("fluency_strict_vs_fluent", details, elapsed)
    result["human_rating_sheet"] = str(sheet)
    result["note"] = ("The sheet is empty by design. Human ratings were not fabricated; "
                      "scores come only from returned raters.")
    return result


LONG_TOPICS = [
    ("calor", ["calor", "gelo"]),
    ("chuva", ["chuva", "umidade"]),
    ("memória", ["memória", "memoria", "grafo"]),
]


def run_long_form(cli, model, out_dir):
    details, elapsed = [], 0.0
    candidates = []
    for index, (topic, expected) in enumerate(LONG_TOPICS):
        fluent, fluent_cost = run([cli, "write", "--model", model, "--topic", topic,
                                   "--sections", "5", "--paragraphs", "2", "--realizer", "fluent"])
        strict, strict_cost = run([cli, "write", "--model", model, "--topic", topic,
                                   "--sections", "5", "--paragraphs", "2", "--realizer", "strict"])
        elapsed += fluent_cost + strict_cost
        sections = fluent.count("\n## ")
        paragraphs = len([line for line in fluent.splitlines() if line and not line.startswith('#')])
        sentences = [part for part in fluent.replace('\n', ' ').split('. ') if part.strip()]
        unique = len(set(sentences))
        coverage = matches(fluent, expected)
        correct = (fluent.startswith("# ") and sections >= 2 and paragraphs >= 3 and coverage)
        details.append({
            "topic": topic,
            "chars": len(fluent),
            "sections": sections,
            "paragraphs": paragraphs,
            "sentences": len(sentences),
            "unique_sentence_ratio": round(unique / len(sentences), 4) if sentences else 0.0,
            "expected": expected,
            "coverage": coverage,
            "correct": correct,
            "text": fluent,
        })
        candidates.append({"id": f"long_form_{index}", "prompt": topic,
                           "system_a": strict, "system_b": fluent})
    candidates_path = out_dir / "long_form_candidates.jsonl"
    sheet_path = out_dir / "long_form_rating_sheet.jsonl"
    candidates_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in candidates) + "\n")
    run([sys.executable, ROOT / "scripts/evaluate_fluency_human.py", "generate",
         "--candidates", candidates_path, "--out", sheet_path])
    result = category_result("long_form_generation", details, elapsed)
    result["human_rating_sheet"] = str(sheet_path)
    result["note"] = "Long-form checks are automatic (coverage/structure/repetition); human ratings are not fabricated."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=ROOT / "build/cognicore")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks/results")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    args.cli = args.cli.resolve()
    args.model = args.model.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="cognicore_bench_", dir=str(args.out)))
    try:
        model, train_log = prepare_model(args.cli, args.model, work)
        workspace = prepare_workspace(work)
        categories = {}
        categories["science"] = run_science(args.cli, model)
        categories["hard_reasoning"] = run_hard(args.cli, model)
        categories["math"] = run_math(args.cli, model)
        categories["code"] = run_code(args.cli, model, workspace, work)
        categories["patch"] = run_patch(args.cli, model, workspace, work)
        categories["fluency"] = run_fluency(args.cli, model, args.out)
        categories["long_form"] = run_long_form(args.cli, model, args.out)

        calibration, _ = run([args.cli, "evaluate-fluency", "--model", model,
                              "--input", ROOT / "benchmarks/data/heldout_pt.txt"])
        calibration = json.loads(calibration)
        peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        report = {
            "architecture": "symbolic VSA + Kanerva SDM + Tsetlin + causal graph + DOP; no neural weights",
            "source_model": str(args.model),
            "benchmark_model": str(model),
            "categories": categories,
            "calibration": calibration,
            "peak_child_rss_kb": peak_rss,
            "train_log_tail": train_log.strip().splitlines()[-3:],
            "notes": [
                "Project-owned diagnostic sets, not the original GPQA/HLE/GSM8K/HumanEval/SWE-bench.",
                "The science set is ingested from benchmarks/data/science_facts.txt.",
                "PPM calibration uses a small hand-written probe, not a representative held-out corpus.",
                "No human fluency ratings are included; use the generated rating sheet with real raters.",
            ],
        }
        report_path = args.out / "benchmark_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({
            "report": str(report_path),
            "accuracies": {name: value["accuracy"] for name, value in categories.items()},
            "calibration_bits_per_byte": calibration["bits_per_byte"],
            "calibration_perplexity": calibration["perplexity"],
            "peak_child_rss_kb": peak_rss,
        }, indent=2, ensure_ascii=False))
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
