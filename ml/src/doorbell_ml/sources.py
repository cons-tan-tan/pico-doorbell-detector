import csv
import hashlib
import json
import os

ESC50_CATEGORIES = (
    "dog",
    "rain",
    "water_drops",
    "pouring_water",
    "toilet_flush",
    "crying_baby",
    "coughing",
    "footsteps",
    "door_wood_knock",
    "door_wood_creaks",
    "can_opening",
    "washing_machine",
    "vacuum_cleaner",
    "clock_alarm",
    "clock_tick",
    "glass_breaking",
    "keyboard_typing",
)


def import_esc50(args):
    from .data import write_manifest

    root = args.root.resolve()
    if not (root / "LICENSE").is_file():
        raise ValueError(
            "Keep the ESC-50 LICENSE file alongside the audio and metadata"
        )
    rows = []
    with (root / "meta/esc50.csv").open(newline="") as handle:
        sources = list(csv.DictReader(handle))
    source_folds = {}
    for source in sources:
        key = source["src_file"]
        source_folds[key] = max(source_folds.get(key, 0), int(source["fold"]))
    for source in sources:
        if (
            not getattr(args, "all_categories", False)
            and source["category"] not in args.categories
        ):
            continue
        if args.esc10_only and source["esc10"].lower() != "true":
            continue
        path = root / "audio" / source["filename"]
        if not path.is_file():
            raise ValueError(f"Missing ESC-50 clip: {path}")
        fold = source_folds[source["src_file"]]
        split = "train" if fold <= 3 else "val" if fold == 4 else "test"
        rows.append(
            {
                "path": os.path.relpath(path, args.output.resolve().parent),
                "label": "background",
                "split": split,
                "group": f"freesound:{source['src_file']}",
                "source": f"ESC-50/{source['filename']}; https://freesound.org/s/{source['src_file']}/",
                "license": f"{'CC-BY-3.0 (ESC-10)' if args.esc10_only else 'CC-BY-NC-3.0 (ESC-50)'}; clip attribution: {root / 'LICENSE'}",
                "kind": "recorded",
            }
        )
    if not rows:
        raise ValueError("No ESC-50 clips matched the selected categories")
    write_manifest(args.output, rows)
    print(
        f"Imported {len(rows)} background clips with source-disjoint splits: {args.output}"
    )
    return args.output


def select_fsd50k(records, metadata, excluded_ids, limits, seed):
    if len(limits) != 3 or any(type(value) is not int or value < 0 for value in limits):
        raise ValueError("Specify nonnegative train, validation and test clip limits")
    excluded_uploaders = {
        metadata[key]["uploader"] for key in excluded_ids if key in metadata
    }
    limits = dict(zip(("train", "val", "test"), limits, strict=True))
    counts = dict.fromkeys(limits, 0)
    ordered = sorted(
        records,
        key=lambda row: hashlib.sha256(f"{seed}:{row['fname']}".encode()).hexdigest(),
    )
    selected = []
    for row in ordered:
        key = row["fname"]
        info = metadata[key]
        uploader = info["uploader"]
        if not uploader or not info["license"]:
            raise ValueError("FSD50K clip lacks uploader or license")
        description = (info["title"] + " " + " ".join(info["tags"])).lower()
        if (
            key in excluded_ids
            or uploader in excluded_uploaders
            or "Doorbell" in row["labels"].split(",")
            or "doorbell" in description
        ):
            continue
        bucket = (
            int(hashlib.sha256(f"doorbell:{uploader}".encode()).hexdigest()[:8], 16)
            % 10
        )
        split = "test" if bucket < 2 else "val" if bucket < 4 else "train"
        if counts[split] < limits[split]:
            selected.append(row | {"split": split})
            counts[split] += 1
    return selected


def import_fsd50k(args):
    from .data import file_digest, read_manifests, write_manifest

    root = args.root.resolve()
    metadata = {}
    for split in ("dev", "eval"):
        metadata |= json.loads(
            (root / f"FSD50K.metadata/{split}_clips_info_FSD50K.json").read_text()
        )
    existing = read_manifests(args.exclude_manifest) if args.exclude_manifest else []
    excluded = {
        record.group.removeprefix("freesound:")
        for record in existing
        if record.group.startswith("freesound:")
    }
    with (root / "FSD50K.ground_truth/eval.csv").open(newline="") as handle:
        records = [
            row
            for row in csv.DictReader(handle)
            if (root / "FSD50K.eval_audio" / f"{row['fname']}.wav").is_file()
        ]
    selected = select_fsd50k(records, metadata, excluded, args.limits, args.seed)
    known_hashes = {file_digest(record.path) for record in existing}
    rows = []
    for row in sorted(
        selected, key=lambda row: ("train", "val", "test").index(row["split"])
    ):
        key = row["fname"]
        info = metadata[key]
        path = root / "FSD50K.eval_audio" / f"{key}.wav"
        digest = file_digest(path)
        if digest in known_hashes:
            continue
        known_hashes.add(digest)
        rows.append(
            {
                "path": os.path.relpath(path, args.output.resolve().parent),
                "label": "background",
                "split": row["split"],
                "group": f"fsd-uploader:{info['uploader']}",
                "source": f"FSD50K/{key}; https://freesound.org/s/{key}/; uploader={info['uploader']}; labels={row['labels']}; custom uploader-disjoint split of official eval",
                "license": info["license"],
                "kind": "recorded",
            }
        )
    if not rows:
        raise ValueError("No downloaded FSD50K backgrounds matched the selection")
    write_manifest(args.output, rows)
    print(
        f"Imported {len(rows)} FSD50K clips with uploader-disjoint custom splits: {args.output}"
    )
    return args.output
