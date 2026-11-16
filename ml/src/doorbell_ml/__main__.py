import argparse
import os
import subprocess
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    from .sources import ESC50_CATEGORIES

    project = Path(__file__).resolve().parents[2]
    data = project.parent / "data"
    root = argparse.ArgumentParser(
        description="Train and evaluate the Pico doorbell audio classifier."
    )
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-data", help="Fetch and verify the pinned training corpus"
    )
    prepare.add_argument("--data", type=Path, default=data)
    prepare.add_argument("--offline", action="store_true")
    retrain = commands.add_parser(
        "retrain",
        help="Train from initialization, mine negatives, and refine a candidate",
    )
    retrain.add_argument("--recipe", type=Path, default=project / "training.toml")
    retrain.add_argument("--output", type=Path, required=True)
    deploy = commands.add_parser(
        "export-firmware",
        help="Package a model and its validated decision settings for firmware",
    )
    deploy.add_argument("--model", type=Path, required=True)
    deploy.add_argument("--decision", type=Path, required=True)
    deploy.add_argument("--output", type=Path, required=True)
    checkpoint = commands.add_parser(
        "export-checkpoint",
        help="Quantize a saved validation checkpoint for firmware evaluation",
    )
    checkpoint.add_argument("--model", type=Path, required=True)
    checkpoint.add_argument("--epoch", type=int, required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify-runtime",
        help="Compare TFLite Micro and LiteRT reference kernels on identical int8 inputs",
    )
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument("--examples", type=Path, required=True)
    verify.add_argument("--binary", type=Path, required=True)
    verify.add_argument("--output", type=Path)
    train = commands.add_parser(
        "train", help="Train a small CNN and export a fully int8 TFLite model"
    )
    train.add_argument(
        "--manifest", type=Path, nargs="+", default=[data / "recordings.csv"]
    )
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--examples-per-class", type=int, default=256)
    train.add_argument("--validation-per-class", type=int, default=128)
    train.add_argument("--synthetic-fraction", type=float, default=0.25)
    train.add_argument("--background-multiplier", type=int, default=2)
    train.add_argument("--hard-negatives", type=Path)
    train.add_argument("--hard-negative-fraction", type=float, default=0.5)
    train.add_argument("--time-shift-seconds", type=float, default=0.5)
    train.add_argument("--soundscape-fraction", type=float, default=0.25)
    train.add_argument("--min-snr-db", type=float, default=-5)
    train.add_argument("--time-mask-frames", type=int, default=8)
    train.add_argument("--frequency-mask-bins", type=int, default=1)
    train.add_argument("--mask-probability", type=float, default=0.25)
    train.add_argument("--filter-db", type=float, default=3.0)
    train.add_argument("--filter-probability", type=float, default=0.0)
    train.add_argument("--notification-validation", type=Path)
    train.add_argument("--decision-binary", type=Path)
    train.add_argument("--selection-interval", type=int, default=4)
    train.add_argument("--initial-model", type=Path)
    train.add_argument("--teacher-model", type=Path)
    train.add_argument("--distillation-weight", type=float, default=0.25)
    train.add_argument("--distillation-temperature", type=float, default=2.0)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--max-model-bytes", type=int, default=65536)
    train.add_argument("--channels", type=int, nargs=3, default=(8, 12, 16))
    train.add_argument("--compression", choices=("log", "pcen"), default="log")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=("cpu", "gpu"), default="cpu")

    cache = commands.add_parser(
        "cache-validation",
        help="Freeze validation features for notification-based checkpoint selection",
    )
    cache.add_argument("--report", type=Path, required=True)
    cache.add_argument("--trace", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--compression", choices=("log", "pcen"), default="log")

    select = commands.add_parser(
        "select-policy", help="Select one policy across matched validation corpora"
    )
    select.add_argument("--reports", type=Path, nargs="+", required=True)
    select.add_argument("--references", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, required=True)

    teacher = commands.add_parser(
        "train-teacher",
        help="Fit a PC-only classifier on frozen YAMNet temporal embeddings",
    )
    teacher.add_argument("--manifest", type=Path, nargs="+", required=True)
    teacher.add_argument("--pretrained", type=Path, required=True)
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument("--notification-validation", type=Path, required=True)
    teacher.add_argument("--decision-binary", type=Path, required=True)
    teacher.add_argument("--hard-negatives", type=Path)
    teacher.add_argument("--pools", type=int, default=4)
    teacher.add_argument("--examples-per-class", type=int, default=512)
    teacher.add_argument("--epochs", type=int, default=40)
    teacher.add_argument("--seed", type=int, default=42)

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate int8 inference on held-out recordings"
    )
    evaluate.add_argument("--manifest", type=Path, nargs="+", required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--threshold", type=float, default=0.8)
    evaluate.add_argument("--hop-seconds", type=float, default=0.256)
    evaluate.add_argument("--cooldown-seconds", type=float, default=3.0)
    evaluate.add_argument("--consecutive", type=int, default=1)
    evaluate.add_argument("--output", type=Path)

    mine = commands.add_parser(
        "mine", help="Rank recorded training backgrounds by chime score"
    )
    mine.add_argument("--manifest", type=Path, nargs="+", required=True)
    mine.add_argument("--model", type=Path, required=True)
    mine.add_argument("--output", type=Path, required=True)
    mine.add_argument("--count", type=int, default=64)
    mine.add_argument("--hop-seconds", type=float, default=0.256)

    benchmark = commands.add_parser(
        "benchmark",
        help="Measure held-out background alarms and known-chime robustness",
    )
    benchmark.add_argument("--manifest", type=Path, nargs="+", required=True)
    benchmark.add_argument("--model", type=Path, required=True)
    benchmark.add_argument("--split", choices=("val", "test"), required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--seed", type=int, default=20261115)
    benchmark.add_argument("--episodes-per-condition", type=int, default=20)
    benchmark.add_argument(
        "--snr-db", type=float, nargs="+", default=[-5, 0, 5, 10, 20]
    )
    benchmark.add_argument("--hop-seconds", type=float, default=0.256)
    benchmark.add_argument("--cooldown-seconds", type=float, default=3.0)
    benchmark.add_argument("--calibrate", action="store_true")
    benchmark.add_argument("--policy", type=Path)
    benchmark.add_argument("--minimum-recall", type=float, default=0.95)
    benchmark.add_argument("--minimum-snr-db", type=float, default=5)

    compare = commands.add_parser(
        "compare",
        help="Compare Rust rules, firmware ML, and a two-stage detector on PC",
    )
    compare.add_argument("--manifest", type=Path, nargs="+", required=True)
    compare.add_argument("--model", type=Path, required=True)
    compare.add_argument("--replay-binary", type=Path, required=True)
    compare.add_argument("--split", choices=("val", "test"), required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--policy", type=Path)
    compare.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    )
    compare.add_argument("--episodes-per-condition", type=int, default=20)
    compare.add_argument("--snr-db", type=float, nargs="+", default=[-5, 0, 5, 10, 20])
    compare.add_argument("--minimum-snr-db", type=float, default=5)
    compare.add_argument("--seed", type=int, default=20261115)

    policy = commands.add_parser(
        "evaluate-policy",
        help="Evaluate temporal confirmation and pattern-assisted ML on separate acoustic conditions",
    )
    policy.add_argument("--manifest", type=Path, nargs="+", required=True)
    policy.add_argument("--model", type=Path, required=True)
    policy.add_argument("--replay-binary", type=Path, required=True)
    policy.add_argument("--decision-binary", type=Path, required=True)
    policy.add_argument("--split", choices=("val", "test"), required=True)
    policy.add_argument("--output", type=Path, required=True)
    policy.add_argument("--cache", type=Path)
    policy.add_argument("--policy", type=Path)
    policy.add_argument("--reference-report", type=Path)
    policy.add_argument("--fresh-test-backgrounds", action="store_true")
    policy.add_argument("--episodes-per-condition", type=int, default=20)
    policy.add_argument("--snr-db", type=float, nargs="+", default=[-5, 0, 5, 10, 20])
    policy.add_argument("--seed", type=int, default=20261116)

    stream = commands.add_parser(
        "evaluate-stream",
        help="Evaluate notifications without resetting between Scaper scenes",
    )
    stream.add_argument("--manifest", type=Path, nargs="+", required=True)
    stream.add_argument("--model", type=Path, required=True)
    stream.add_argument("--replay-binary", type=Path, required=True)
    stream.add_argument("--decision-binary", type=Path, required=True)
    stream.add_argument("--policy", type=Path, required=True)
    stream.add_argument("--split", choices=("val", "test"), required=True)
    stream.add_argument("--output", type=Path, required=True)
    stream.add_argument("--blocks", type=int, default=100)
    stream.add_argument("--seed", type=int, default=20261117)
    stream.add_argument("--fresh-test-backgrounds", action="store_true")

    synth = commands.add_parser(
        "synth", help="Render MIDI instrument sequences as training negatives"
    )
    soundfont = os.environ.get("DOORBELL_SOUNDFONT")
    synth.add_argument(
        "--soundfont", type=Path, default=Path(soundfont) if soundfont else None
    )
    synth.add_argument(
        "--soundfont-license", default="unspecified; consult SoundFont license"
    )
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--count", type=int, default=48)
    synth.add_argument("--seed", type=int, default=42)

    esc = commands.add_parser(
        "import-esc50", help="Import selected ESC-50 negatives, preserving source folds"
    )
    esc.add_argument("--root", type=Path, required=True)
    esc.add_argument("--output", type=Path, required=True)
    esc.add_argument("--categories", nargs="+", default=ESC50_CATEGORIES)
    esc.add_argument("--all-categories", action="store_true")
    esc.add_argument("--esc10-only", action="store_true")
    fsd = commands.add_parser(
        "import-fsd50k",
        help="Import downloaded FSD50K backgrounds with uploader-disjoint custom splits",
    )
    fsd.add_argument("--root", type=Path, required=True)
    fsd.add_argument("--output", type=Path, required=True)
    fsd.add_argument("--exclude-manifest", type=Path, nargs="+")
    fsd.add_argument("--limits", type=int, nargs=3, default=(1500, 400, 600))
    fsd.add_argument("--seed", type=int, default=20261116)
    return root


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    try:
        if args.command == "prepare-data":
            from .prepare import prepare_data

            prepare_data(args.data, offline=args.offline)
        elif args.command == "retrain":
            from .retrain import retrain

            retrain(args.recipe, args.output)
        elif args.command == "export-firmware":
            from .deployment import export_firmware

            export_firmware(args.model, args.decision, args.output)
        elif args.command == "train":
            from .training import train

            train(args)
        elif args.command == "export-checkpoint":
            from .training import export_checkpoint

            export_checkpoint(args)
        elif args.command == "verify-runtime":
            from .deployment import verify_runtime

            verify_runtime(args)
        elif args.command == "cache-validation":
            from .selection import cache_validation

            cache_validation(args)
        elif args.command == "select-policy":
            from .selection import select_policy

            select_policy(args)
        elif args.command == "train-teacher":
            from .distillation import train_teacher

            train_teacher(args)
        elif args.command == "evaluate":
            from .evaluate import evaluate

            evaluate(args)
        elif args.command == "mine":
            from .evaluate import mine

            mine(args)
        elif args.command == "benchmark":
            from .benchmark import benchmark

            benchmark(args)
        elif args.command == "compare":
            from .hybrid import compare

            compare(args)
        elif args.command == "evaluate-policy":
            from .policy_eval import evaluate_policy

            evaluate_policy(args)
        elif args.command == "evaluate-stream":
            from .stream_eval import evaluate_stream

            evaluate_stream(args)
        elif args.command == "synth":
            from .synth import synthesize

            synthesize(args)
        elif args.command == "import-esc50":
            from .sources import import_esc50

            import_esc50(args)
        elif args.command == "import-fsd50k":
            from .sources import import_fsd50k

            import_fsd50k(args)
    except (TypeError, ValueError, FileExistsError, FileNotFoundError) as error:
        cli.error(str(error))
    except subprocess.CalledProcessError as error:
        detail = error.stderr
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        cli.error(f"{error.cmd[0]} failed: {detail}")


if __name__ == "__main__":
    main()
