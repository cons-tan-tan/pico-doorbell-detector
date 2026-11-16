import hashlib
import json

import numpy as np

from .audio import Frontend
from .data import (
    LABELS,
    check_independent,
    file_digest,
    provenance,
    read_manifests,
    write_json,
)
from .metrics import notification_metrics
from .policy_eval import validate_policy
from .replay import run_decisions, score_clip
from .soundscapes import SoundscapeCorpus


def make_stream(records, split, seed, blocks, config):
    rng = np.random.default_rng(seed)
    backgrounds = [r for r in records if r.split == split and r.label == "background"]
    positives = [r for r in records if r.split == "train" and r.label != "background"]
    if not backgrounds or {r.label for r in positives} != set(LABELS[1:]) or blocks < 4:
        raise ValueError(
            "Need both known chimes, held-out backgrounds and at least four blocks"
        )
    chunks, annotations, used = [], [], set()
    with SoundscapeCorpus(backgrounds + positives, config) as corpus:
        order = rng.permutation(len(backgrounds))
        for index in range(blocks):
            background = backgrounds[int(order[index % len(order)])]
            used.add(background)
            events = []
            if index % 2:
                label = LABELS[1 + (index // 2) % 2]
                choices = [r for r in positives if r.label == label]
                record = choices[int(rng.integers(len(choices)))]
                snr = float((-5, 0, 5, 10, 20)[(index // 4) % 5])
                events.append((record, float(rng.uniform(2, 3)), snr))
            audio, labels = corpus.render(background, events, rng)
            fade = np.linspace(0, 1, round(0.01 * config.sample_rate))
            audio[: len(fade)] *= fade
            audio[-len(fade) :] *= fade[::-1]
            chunks.append(audio)
            annotations.extend(
                {
                    **event,
                    "onset": event["onset"] + index * 12,
                    "offset": event["offset"] + index * 12,
                }
                for event in labels
            )
    return np.concatenate(chunks), annotations, sorted(used, key=lambda r: str(r.path))


def evaluate_stream(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    fresh = getattr(args, "fresh_test_backgrounds", False)
    if fresh and args.split != "test":
        raise ValueError("Fresh-background declaration applies only to test evaluation")
    records = read_manifests(args.manifest)
    metadata = json.loads((args.model / "metadata.json").read_text())
    model_sha = file_digest(args.model / "model.int8.tflite")
    frozen = json.loads(args.policy.read_text())
    validate_policy(
        frozen,
        model_sha,
        file_digest(args.replay_binary),
        file_digest(args.decision_binary),
    )
    config = Frontend(**metadata["frontend"])
    backgrounds = [
        r for r in records if r.split == args.split and r.label == "background"
    ]
    check_independent(
        backgrounds,
        metadata,
        args.split,
        frozen["sources"] if args.split == "test" else (),
    )
    audio, events, used = make_stream(
        records, args.split, args.seed, args.blocks, config
    )
    row = score_clip(args.replay_binary, audio, config, metadata, model_sha)
    alarms = run_decisions(args.decision_binary, [row], frozen["policy"])[0]
    report = {
        "split": args.split,
        "model_sha256": model_sha,
        "policy": frozen["policy"],
        "protocol": {
            "generator": "scaper",
            "seed": args.seed,
            "blocks": args.blocks,
            "audio_sha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
            "state_reset": "once_at_start",
            "positive_sources_independent": False,
            "test_backgrounds_previously_inspected": args.split == "test" and not fresh,
            "background_sources": provenance(used),
            "positive_sources": provenance(
                [r for r in records if r.split == "train" and r.label != "background"]
            ),
            "latency": "audio timeline; excludes compute and notification transport",
            "snr_definition": "Scaper integrated loudness",
        },
        "metrics": notification_metrics(
            events, alarms, len(audio) / config.sample_rate
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(
        json.dumps(
            {k: v for k, v in report["metrics"].items() if not k.endswith("details")},
            indent=2,
        )
    )
    return report
