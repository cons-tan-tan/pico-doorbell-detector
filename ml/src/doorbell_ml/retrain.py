import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

from .data import file_digest, provenance, read_manifests, write_json


def options(values):
    arguments = []
    for key, value in values.items():
        arguments.append("--" + key.replace("_", "-"))
        items = value if isinstance(value, list) else [value]
        if not items or any(type(item) not in (str, int, float) for item in items):
            raise ValueError(f"Invalid recipe option: {key}")
        arguments.extend(str(item) for item in items)
    return arguments


def recipe_commands(recipe: Path, output: Path):
    from .__main__ import parser

    config = tomllib.loads(recipe.read_text())
    if config.get("version") != 1 or set(config) != {
        "version",
        "manifests",
        "train",
        "base",
        "mine",
        "refine",
    }:
        raise ValueError("Expected a version 1 training recipe")
    forbidden = {
        "manifest",
        "output",
        "initial_model",
        "teacher_model",
        "hard_negatives",
        "notification_validation",
        "decision_binary",
        "model",
    }
    for section in ("train", "base", "refine", "mine"):
        if forbidden.intersection(config[section]):
            raise ValueError(f"{section}: recipe cannot depend on previous runs")
    manifests = [(recipe.parent / value).resolve() for value in config["manifests"]]
    if not manifests:
        raise ValueError("Recipe needs training manifests")
    inputs = ["--manifest", *map(str, manifests)]
    base, candidate, hard = (
        output / "base",
        output / "candidate",
        output / "hard-negatives.json",
    )
    commands = [
        [
            "train",
            *inputs,
            *options(config["train"] | config["base"]),
            "--output",
            str(base),
        ],
        [
            "mine",
            *inputs,
            *options(config["mine"]),
            "--model",
            str(base),
            "--output",
            str(hard),
        ],
        [
            "train",
            *inputs,
            *options(config["train"] | config["refine"]),
            "--initial-model",
            str(base),
            "--hard-negatives",
            str(hard),
            "--output",
            str(candidate),
        ],
    ]
    cli = parser()
    for command in commands:
        try:
            cli.parse_args(command)
        except SystemExit as error:
            raise ValueError("Invalid training recipe options") from error
    return manifests, commands


def retrain(recipe: Path, output: Path):
    recipe, output = recipe.resolve(), output.resolve()
    manifests, commands = recipe_commands(recipe, output)
    records = read_manifests(manifests)
    root = Path(__file__).resolve().parents[2]
    environment_files = {
        f"ml/{name}": root / name
        for name in ("uv.lock", "pyproject.toml", ".python-version")
    } | {"flake.lock": root.parent / "flake.lock"}
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(recipe, output / "recipe.toml")
    package = Path(__file__).parent
    report = {
        "status": "running",
        "deployment_status": "candidate_only; requires notification evaluation and device checks",
        "initialization": "random weights; hard negatives mined from this run's base model",
        "recipe_sha256": file_digest(recipe),
        "manifests": {str(path): file_digest(path) for path in manifests},
        "sources": provenance(records),
        "code_sha256": {
            path.name: file_digest(path) for path in sorted(package.glob("*.py"))
        },
        "environment_sha256": {
            name: file_digest(path)
            for name, path in environment_files.items()
            if path.is_file()
        },
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in (
                "tensorflow",
                "numpy",
                "scipy",
                "soundfile",
                "scaper",
                "ai-edge-litert",
            )
        },
        "commands": commands,
    }
    write_json(output / "run.json", report)
    environment = os.environ | {
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "PYTHONHASHSEED": "0",
    }
    try:
        for name, command in zip(("base", "mine", "refine"), commands, strict=True):
            print(f"Starting {name}; log: {output / (name + '.log')}", flush=True)
            with (output / f"{name}.log").open("x") as log:
                subprocess.run(
                    [sys.executable, "-m", "doorbell_ml", *command],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
        report["status"] = "complete"
        report["candidate_sha256"] = file_digest(output / "candidate/model.int8.tflite")
    except subprocess.CalledProcessError:
        report["status"] = "failed"
        raise
    finally:
        (output / "run.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    print(f"Candidate ready for notification evaluation: {output / 'candidate'}")
    return report
