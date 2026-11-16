import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from .audio import load_audio
from .data import file_digest, write_manifest

PROGRAMS = (0, 4, 10, 11, 24, 25, 26, 27, 28, 29, 80, 81)


def make_midi(path: Path, rng: np.random.Generator, program: int) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(int(rng.integers(80, 181))))
    )
    track.append(mido.Message("program_change", program=program, time=0))
    for _ in range(int(rng.integers(4, 9))):
        root = (
            int(rng.choice((75, 76, 77, 79, 80, 81)))
            if rng.random() < 0.5
            else int(rng.integers(45, 85))
        )
        intervals = (0, 4, 7) if rng.random() < 0.35 else (0,)
        for interval in intervals:
            track.append(
                mido.Message(
                    "note_on",
                    note=root + interval,
                    velocity=int(rng.integers(40, 121)),
                    time=0,
                )
            )
        duration = int(rng.integers(100, 361))
        for index, interval in enumerate(intervals):
            track.append(
                mido.Message(
                    "note_off",
                    note=root + interval,
                    velocity=0,
                    time=duration if index == 0 else 0,
                )
            )
    track.append(mido.MetaMessage("end_of_track", time=480))
    midi.save(path)


def synthesize(args) -> Path:
    if args.count < 1:
        raise ValueError("count must be positive")
    executable = shutil.which("fluidsynth")
    if executable is None:
        raise ValueError("FluidSynth is required; run this command inside nix develop")
    if args.soundfont is None or not args.soundfont.is_file():
        raise ValueError("Provide --soundfont or enter nix develop to use FluidR3")
    font_hash = file_digest(args.soundfont)
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    rows, generated = [], []
    for index in range(args.count):
        program = PROGRAMS[index % len(PROGRAMS)]
        name = f"instrument-{index:05d}"
        midi_path = args.output / f"{name}.mid"
        make_midi(midi_path, rng, program)
        with tempfile.TemporaryDirectory() as temporary:
            rendered = Path(temporary) / "render.wav"
            subprocess.run(
                [
                    executable,
                    "-ni",
                    "-q",
                    "-R",
                    "0",
                    "-C",
                    "0",
                    "-T",
                    "wav",
                    "-r",
                    "16000",
                    "-F",
                    str(rendered),
                    str(args.soundfont),
                    str(midi_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            samples = load_audio(rendered, 8000).copy()
            load_audio.cache_clear()
        active = np.flatnonzero(np.abs(samples) > 1e-5)
        if not len(active):
            raise ValueError(f"SoundFont produced silence for MIDI program {program}")
        samples = samples[: min(len(samples), int(active[-1]) + 1600)]
        sf.write(args.output / f"{name}.wav", samples, 8000, subtype="PCM_16")
        rows.append(
            {
                "path": f"{name}.wav",
                "label": "background",
                "split": "train",
                "group": f"midi:{font_hash}:program:{program}",
                "source": f"FluidSynth; {args.soundfont.name}; program={program}; seed={args.seed}",
                "license": args.soundfont_license,
                "kind": "synthetic",
            }
        )
        generated.append(
            {"file": f"{name}.wav", "midi": midi_path.name, "program": program}
        )
    manifest = args.output / "manifest.csv"
    write_manifest(manifest, rows)
    (args.output / "synthesis.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "soundfont": str(args.soundfont.resolve()),
                "soundfont_sha256": font_hash,
                "soundfont_license": args.soundfont_license,
                "files": generated,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Rendered {args.count} training negatives; manifest: {manifest}")
    return manifest
