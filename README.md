# Maritime Vessel Re-ID World-Model Experiment

Vision-only Phase 1 experiment testing whether **JEPA-style predictive video
representations** improve video-based vessel re-identification, with the
headline test being **cross-camera, delayed re-acquisition after full
disappearance** (the "disappearance-gap" protocol), against conventional CNN
Re-ID, VLA-derived vision features, motion-only dead-reckoning, and
tracker-based association baselines.

See `maritime_reid_worldmodel_vla_experiment_spec.md` for the full
specification. This repository implements it.

## Research question

> Do frozen V-JEPA 2.1 video representations, with a lightweight Re-ID probe,
> outperform conventional image Re-ID, VLA-derived vision representations,
> motion-only dead-reckoning, and tracker-based association on cross-camera
> vessel Re-ID — particularly for re-acquisition after full disappearance across
> a time gap?

**Non-claims (explicit):** this is a vision-only benchmark. It does not claim to
validate a full AIS/radar/satellite world model, does not prove fog penetration,
and does not evaluate VLAs in their intended robotics-control setting.

## Repository layout

```text
configs/                 YAML configs: data / models / experiments / gap
data/                    manifests/ (tracklet manifests), gap_trials/ (DGRA trials)
scripts/                 CLI entry points (spec §21)
src/
  data/                  manifests, splits, sampling, gap trials, distractor pools, adapters
  models/                TrackletEncoder protocol, shared Re-ID head, arm adapters, baselines
  training/              losses, trainer, callbacks
  evaluation/            reid metrics, robustness, bootstrap, degradation curves, reports
  utils/                 registry, reproducibility, config loading, logging
tests/                   pytest suite for the deterministic core logic
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .            # base requirements (torch, numpy, yaml, ...)
pip install -r requirements/vjepa.txt    # for V-JEPA arms
pip install -r requirements/openvla.txt  # for the VLA arm (heavy; see file)
```

## Data

Place restricted datasets under `data/raw/` (do not commit raw data). Then:

```bash
python scripts/prepare_dataset.py --config-name viv_reid
```

Dataset adapters are the only piece that knows exact upstream folder names;
everything downstream reads the normalized JSONL manifest (spec §4.3).

## Build the disappearance-gap trials

```bash
python scripts/build_gap_trials.py --config-name viv_reid_dgra \
    --manifest data/manifests/viv_reid.jsonl
```

Produces `data/gap_trials/viv_reid_dgra.jsonl` — deterministic
query→gallery re-acquisition trials across short/medium/long gaps, with
distractor pools, maneuver labels, and AIS-withheld subsets.

## Standard run sequence (spec §21)

```bash
python scripts/prepare_dataset.py --config-name viv_reid
python scripts/build_gap_trials.py --config-name viv_reid_dgra
python scripts/extract_features.py --config-name vivreid_cnn
python scripts/extract_features.py --config-name vivreid_vjepa_encoder
python scripts/extract_features.py --config-name vivreid_openvla_vision
python scripts/train_probe.py --config-name vivreid_cnn
python scripts/train_probe.py --config-name vivreid_vjepa_encoder
python scripts/train_probe.py --config-name vivreid_openvla_vision
python scripts/run_baselines.py --config-name vivreid_vjepa_encoder   # Arms F, G, H
python scripts/evaluate.py --config-name vivreid_vjepa_encoder
python scripts/run_stress_suite.py --config-name vivreid_all_arms
python scripts/aggregate_results.py --runs outputs/vivreid_*
```

## How to add a new encoder adapter

1. Implement `TrackletEncoder` (see `src/models/interfaces.py`).
2. Register it:

   ```python
   from models import encoder_registry
   encoder_registry.register("my_encoder", MyEncoder)
   ```

3. Add a `configs/models/my_encoder.yaml` and an experiment config.

## How to add a new gap baseline

Implement `GapBaseline` (`src/models/baselines/base.py`) — a `name` plus
`rank(trial, gallery_features) -> list[str]` returning ranked gallery tracklet
ids. Wire it into `scripts/run_baselines.py`.

## Reproducibility

Every run writes `resolved_config.yaml`, `git_state.json`,
`environment.txt`, manifest hashes, and seeds into its output directory
(`outputs/<run_id>/`). Feature caches use content-addressed keys
(`utils.reproducibility.content_addressed_key`).

## Status

Implemented (tested, 149 passing):

- **Data**: manifests + per-frame bboxes/timestamps, split hygiene, temporal
  sampling, DGRA gap trials, distractor pools, AIS ping/trajectory manifests,
  intermittent-observation sampler (frame-skipping + block patch masks).
- **Adapters**: ViV-ReID (config-driven layout), **FVessel** (video+AIS+GT,
  camera meta, MMSI mapping via fusion IoU), **MVTD** (GOT-10k, absence/cover
  labels -> occlusion/truncation).
- **Harness**: sensor-blackout episodes (10/30/60/120s), AIS withholding with
  jitter/dropout, co-present distractor pools, reappearance ground truth.
- **Metrics**: re-acquisition Top-1/Top-5, IDSW, IDF1, HOTA, Haversine/pixel
  drift, paired bootstrap, degradation curves/slopes (arbitrary gap bins).
- **Arms**: real frozen encoders — CNN resnet50 (Arm A), DINOv2 + SigLIP via
  HF transformers (Arm B), V-JEPA (Arm C, `blocked_by_api` until the official
  package is pinned), Arm D kinematic layer (constant-velocity Kalman +
  predictive search-window proposal).
- **Pipeline**: `extract_features.py` runs end-to-end with content-addressed
  feature caching; shared Re-ID head, ID CE + batch-hard triplet losses.

Stubs (fail loudly, land in later commits): VesselReID adapter, OpenVLA live
feature extraction, V-JEPA predictor future-latency, ByteTrack/BoT-SORT and
AIS-upper-bound baselines, probe trainer, stress-suite and report runs.
