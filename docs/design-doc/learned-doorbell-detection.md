# Learned doorbell detection

Status: running on Pico 2 W; installed-room accuracy remains unverified.
This document covers recognition and evaluation. Hardware and execution belong in
the [firmware design](rust-firmware.md).

## Goal and constraints

Notify reliably when either of two known chimes rings. The earlier tone-only
detector picked up chimes but also reacted to unrelated household sounds.

- Prioritize false alarms, missed chimes, and correct kind over small latency gains.
- Send one timely alert; exact phrase boundaries and repeated alerts are unnecessary.
- Use short recordings of the real chimes. Generate labels from individual
  repetitions and backgrounds, without manually annotating long recordings.

## Detection decision

Train a supervised CNN on the PC and deploy integer inference over log-mel features
on Pico. Detection stays local and independent of the notification service.
Use broader background coverage and a wider model while processing retains headroom.

Retain the tone-pattern rules as supporting evidence:

| Path | Acceptance rule | Reason |
| --- | --- | --- |
| Pattern-assisted ML | A recent complete tone pattern and a moderate ML score agree on the kind. | Recover weaker predictions without accepting frequency proximity alone. |
| ML alone | Repeated high scores support the same kind. | Detect chimes that the rules miss. |

Tune thresholds, pattern lifetime, and confirmation together. Overlapping windows
provide context, but confirmation adds delay and can miss short events. Expire old
patterns so they cannot combine with unrelated later predictions.

Recalibrate these settings when recordings change: better window scores alone do
not establish better notifications. Brief wrong-kind predictions at a chime's
opening must not determine its alert before enough same-kind evidence accumulates.

Start a fixed cooldown after an accepted alert. This suppresses repeated phrases
and also merges separate button presses within that period.

Treat the model, frontend, labels, quantization, and decision policy as one deployment
unit: changing preprocessing can change decisions even with identical weights.

### Low-frequency interference

Strong low-frequency interference can hide a chime from the rules because their
tone strength is relative to the frame's total energy. Apply a stateful high-pass
filter to the rule input only when low frequencies dominate. Unconditional
filtering also strengthened misleading patterns in music, so compare filtered
and original AC energy before choosing the rule input. Matching ML evidence
remains necessary; filtering does not bypass the notification policy.

Keep the filter continuous across frames and recognized patterns, resetting it
after capture gaps. Leave the model's input unchanged to preserve its trained
frontend. Filtering cannot recover clipped samples or establish whether the
interference is acoustic or electrical; inspect capture levels separately.

## Training data

### Sources

| Source | Role | Constraint |
| --- | --- | --- |
| Short chime recordings | Examples of the installed chimes. | Variants of one take are not independent recordings. |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | Real backgrounds, including confusing bells and sirens. | Retain attribution and its noncommercial licence terms. |
| [FSD50K](https://zenodo.org/records/4060432) | Broader recorded background coverage. | Retain clip licences; use a custom uploader-based split, not the official benchmark split. |
| MIDI instruments | Controlled musical confusions and interference. | Supplement real recordings; synthetic timbres provide limited coverage. |

### Example generation

- **Scaper:** insert complete chimes into backgrounds and derive event times
  automatically. Combine stream windows with event-centred examples; exclude
  ambiguous opening and trailing windows from class supervision.
- **Variation:** vary timing, level, and interference. Use mild SpecAugment masks
  during training; broad frequency masks can erase the identifying tones.
- **Hard negatives:** emphasize the training windows that the model mistakes for
  chimes while retaining broad background coverage.

### Captured recordings

Keep training-ready mono WAVs by recording session, with source hashes, sample
intervals, channel choices, and recording conditions. These files are the fixed
inputs for retraining. Phone originals and decoded masters are disposable after
checking the prepared waveforms; reconstructing the original recording is not
a requirement. Cut complete attacks and decays before generating training
windows; a repeated recording is not one long chime event.

Inspect stereo channels before downmixing. Phase differences in phone recordings
can cancel an identifying tone when channels are averaged. Use separate mono
channel crops when this occurs, keeping both channels in the same session group.
If one channel has much weaker chime evidence, select the clearer channel and
record the reason. Resample once to the model's sample rate and store float WAVs
to avoid another quantization step. Preserve earlier takes as
different recording conditions rather than replacing them with the cleanest take.
Sample positive sessions equally, then choose a clip within the session, so extra
repetitions or channels cannot overwhelm the older recording conditions.

### Reproducible preparation and training

Version the prepared local recordings and rendered MIDI negatives alongside
their manifests. Public audio remains a disposable cache: freeze the selected
files, splits, licences, upstream revision, and content hashes, then verify
every restored file. Corpus membership must not depend on what happened to be
downloaded before an interrupted run. Keep unlabelled device captures outside
the supervised manifests.

Maintain one from-scratch recipe in source control. Train an initialized model,
mine hard negatives from its training backgrounds, then refine that model.
Neither data preparation nor this recipe may depend on ignored experiment
scripts, cached features, or historical checkpoints. Preserve data, settings,
code, and environment identities with each run; reject changed inputs and
accidental reuse of an output directory. Fix random seeds and use deterministic
CPU operations, while allowing numerical differences across software or hardware
changes. Byte-identical historical weights are not the retraining contract.

The fresh recipe produces candidates. It uses validation loss as an initial
checkpoint shortlist; notification-level comparisons below still determine
adoption. This recipe consolidates the retained methods without recreating every
historical fine-tuning step. Archive compact measurements and adoption decisions;
discard duplicate models, feature caches, binaries, and one-off scripts once
their required behavior is in the maintained tools.

### Split and leakage policy

Split by original recording or session before augmentation. Keep related clips
together, carrying source provenance through fine-tuning and teacher training.
For FSD50K, group by uploader and exclude ambiguous doorbells, known overlapping
source IDs and uploaders, and exact duplicate files.

Mine hard negatives only from training data. Keep stochastic feature augmentation
out of evaluation and inference. Quantization calibration and numeric references
use unmasked features derived from their stored waveform, which may itself be a mixture.

## Model selection and evaluation

### Selection sequence

1. **Shortlist checkpoints by notification errors.** Floating-point predictions
   approximate the output quantization grid; window-level loss alone is insufficient.
   Use matched controls and repeat promising conditions with different random seeds.
2. **Evaluate actual integer inference.** Replay the Rust frontend, C++ inference,
   and Rust decisions on the PC. Compare multiple inputs with integer reference
   kernels: accelerated desktop kernels can produce different quantized outputs.
3. **Choose one policy across validation corpora.** Compare with the previous
   detector on identical audio. Require no extra normal-condition false alarms,
   missed chimes, or incorrect first kinds in any corpus before ranking gains.
   Check detection for each chime kind as well; Type 2 gains cannot hide Type 1 losses.
4. **Check continuous validation, then freeze.** Evaluate without resetting between
   events. Freeze model and policy before final testing on held-out sources.

### Measurement rules

| Concern | Rule |
| --- | --- |
| Notification accuracy | Count missed events, background alarms, first-kind errors, and duplicates separately. Match each notification once. |
| Acoustic robustness | Retain complete chimes and report normal mixtures separately from pitch and room-response stress. |
| Streaming equivalence | Use the firmware prediction grid and sample timeline; reject incomplete traces. Include confirmation waits; report startup, compute, and transport delays separately. |
| Data reuse | Count shared background exposure once across suites. Augmented variants remain correlated. |
| Repeated recordings | Reset between isolated rings for sensitivity comparisons. In continuous replay, repetitions inside the notification cooldown are intentionally suppressed. |
| Reproducibility | Bind caches, traces, and policies to audio, waveform conversion, frontend, conditions, and applicable model and executable identities. Reject stale combinations. |

Keep final-test regressions in the adoption record. Report losses by corpus and
chime kind even when aggregate results improve. If test results inform a revision,
reclassify those sources as development data and preserve their original evaluation
records. Reserve new sources for testing and freeze the revised detector before
inspecting their predictions. Fresh backgrounds do not make reused positive chime
recordings independent.

The bundled public partitions named `test` have already informed development
and serve as regression data. Their names preserve the historical split;
independent final testing requires new sources.

## Alternatives tried

These decisions concern the tested settings. The implementations remain available
for revisiting when the data or evidence changes.

| Method | Intended benefit | Why it was not selected |
| --- | --- | --- |
| FilterAugment | Cover microphone coloration through band-gain changes. | Mild augmentation did not improve the matched control. |
| PCEN | Adapt compression to persistent background energy. | Some detection gains came with extra false alarms and wrong kinds. |
| YAMNet teacher and distillation | Transfer broader audio knowledge into the small CNN. | No better notification trade-off than supervised training. |
| Stronger training interference | Recognize chimes at lower signal-to-noise ratios. | No improvement in combined validation. |

Two constraints matter when revisiting these methods:

- **PCEN:** preserve continuous background history in training and replay; reset
  after audio gaps. Resetting every window changes the deployed frontend's behavior.
- **Distillation:** preserve temporal information and generate teacher targets from
  the student's actual augmented waveform. Inherit known source provenance; unknown
  pretraining overlap remains an evaluation limitation. The teacher stays PC-only.

## Evidence limits

- **Known-chime mixtures** measure robustness to held-out backgrounds, not accuracy
  on new chime recordings. Short public clips do not establish a household alarm rate.
- **Device checks** establish numerical agreement, runtime headroom, capture, and
  notification operation under the observed conditions.
- **Phone playback** checks the microphone-to-notification path. Uncontrolled level,
  speaker, and distance do not establish sensitivity faults or an operating range.
- **Additional phone takes** can expose weaknesses before training. Once used for
  fine-tuning, their crops measure fitting and robustness, not independent accuracy.
  Phone channels do not replace evaluation through the installed mono microphone.

Installed-room accuracy requires representative chime and background recordings
through the final microphone and placement. Keep settings, source identities,
measurements, and adoption evidence with generated artifacts rather than duplicating
them here.
