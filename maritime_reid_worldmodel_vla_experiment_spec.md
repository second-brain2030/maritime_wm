# AI Coding Specification: Maritime Vessel Re-ID Experiment

> **Revision note.** This document merges the original Phase-1 Re-ID specification with the
> "strengths of JEPA, dive deeper" analysis. The central change: the headline test is no longer
> same-frame occlusion (which classical trackers already handle) but **cross-camera, delayed
> re-acquisition after full disappearance** — a "predict across a visibility gap, then verify"
> task that is structurally the same as the task where JEPA shows its largest verified gains
> (action anticipation, forward latent prediction). Everything in the original spec is preserved;
> new hard-test material is marked **[HARD TEST]**.

## 1. Purpose

Build a reproducible research codebase that tests whether a JEPA-style predictive video representation improves **video-based vessel re-identification (Re-ID)** — specifically the ability to **re-identify a vessel after it has fully disappeared from view and reappears later, in a different camera, under different conditions, among visually similar distractors**.

The experiment must compare four representation families under one controlled Re-ID pipeline:

1. **Conventional supervised Re-ID baseline** (CNN / image Re-ID).
2. **V-JEPA 2.1 visual representation** (frozen backbone + lightweight temporal Re-ID probe).
3. **V-JEPA 2.1 with temporal prediction features** (encoder features plus predictor-derived future latent features).
4. **VLA-derived visual representation** (OpenVLA/Prismatic vision encoder features), included as a competitive representation baseline.

In addition, the hard test requires **external controls that are not representation arms**:

5. **Motion-only dead-reckoning** (Kalman/particle filter + nearest-position matching) — defines what appearance-free prediction achieves.
6. **Conventional tracker with re-ID head** (ByteTrack/BoT-SORT + appearance embedding) — the current maritime practical standard.
7. **AIS-fused association baseline** (with AIS available) — a separately reported upper-bound reference, *not* part of the fair vision-only comparison.

This is a vision-only Phase 1 experiment. It does **not** claim to validate the complete AIS/radar/satellite world model. Its purpose is to establish whether predictive video representations offer a measurable advantage before building multimodal fusion.

### 1.1 Why the hard test must be a disappearance-gap test [HARD TEST]

The JEPA strengths analysis motivates the difficulty calibration: JEPA's dramatic, benchmark-verified gains (e.g., 39.7%→40.8% Recall@5 on action anticipation benchmarks, +6.1 points on Something-Something v2) come from tasks with one structural property — **the model must generate a belief about something not currently visible, using only latent/temporal context, and that belief is checked against ground truth that arrives later or from a different vantage point** ([Meta AI V-JEPA 2 blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/); [V-JEPA 2 paper, arXiv 2506.09985](https://arxiv.org/abs/2506.09985)).

Same-frame occlusion tests (occluder over part of the ship in one continuous shot) do **not** have this property. Classical trackers with memory (Kalman filters, ByteTrack-style track buffers) already handle brief same-scene occlusion — which is why the occlusion-benchmark literature reports incremental 2–5 point gains. If our test only re-creates that setup, prior methods will look competitive and we will not get the "dramatically better" signal we are after.

The hard test is therefore **cross-camera, delayed re-acquisition after full disappearance**:

| Design choice | Easy version (avoid) | Hard version (use) |
| :-- | :-- | :-- |
| Occlusion type | Partial occluder over part of the ship in one continuous shot | Full disappearance: vessel exits frame or is 100% blocked for 10–120+ seconds |
| Reappearance | Same camera, same shot, seconds later | Different camera, different angle/range/lighting, minutes later |
| Ground-truth check | Any plausible detection near the last known point counts | Exact identity match against a pool of 5–20 visually similar vessels present simultaneously |
| Motion during gap | Straight line at constant speed (trivial for Kalman) | Heading/speed change, or gap long enough that dead-reckoning drifts past the ambiguity radius |
| Distractors | One vessel in the scene | Multiple near-identical vessels (same shipping line, same hull class) moving through the same waterway concurrently |
| Sensor state during gap | AIS/radar available as a ground-truth crutch | AIS withheld/corrupted for a subset, forcing vision-only re-identification |

## 2. Research Question

Primary question:

> Do frozen V-JEPA 2.1 video representations, with a lightweight Re-ID probe, outperform conventional image Re-ID, VLA-derived vision representations, motion-only dead-reckoning, and tracker-based association on cross-camera vessel Re-ID — particularly for **re-acquisition after full disappearance across a time gap**, where identity must be recovered from pre-disappearance evidence plus predictive latent state?

Secondary questions:

- Does adding V-JEPA predictor-derived future-latent features improve over encoder-only V-JEPA features?
- **Does the predictive-representation advantage widen as the disappearance gap grows** (baselines collapse toward chance at long gaps while V-JEPA matching stays above chance), mirroring the pattern in action anticipation where JEPA's edge is largest at the longest prediction horizons? **[HARD TEST]**
- Is any gain concentrated in difficult subgroups (maneuver-during-gap, cross-camera, high-similarity distractor pools) rather than clean, near-duplicate views?
- Can the re-identification system remain above chance when **AIS is withheld**, i.e., when no kinematic ground-truth crutch is available? **[HARD TEST]**
- Does temporal aggregation beat frame-only features at the same compute budget?
- Can model confidence be calibrated well enough to support a later local-to-global maritime tracking system?

## 3. Scope and Non-Goals

### In scope

- Dataset ingestion, validation, and split handling for video vessel Re-ID.
- Tracklet-level feature extraction and matching.
- **Disappearance-gap trial construction: cross-camera / cross-time query–gallery pairing, distractor pools, gap-duration bins, AIS-withheld subsets.** **[HARD TEST]**
- Frozen-backbone linear/attentive probes.
- V-JEPA, VLA-derived visual, and conventional Re-ID baselines.
- **Motion-only dead-reckoning and tracker-with-re-ID external controls.** **[HARD TEST]**
- Standard Re-ID metrics and difficult-condition slices, plus **gap-degradation curves**.
- Reproducibility, experiment tracking, and result reporting.
- Optional synthetic occlusion and sparse-observation stress tests (as diagnostics, see §8.4).

### Explicitly out of scope for Phase 1

- Training a new foundation/world model from scratch.
- Autonomous vessel or drone control.
- Collision-avoidance decisions.
- Inferring facts during total sensor blackout (the test measures identity re-acquisition from *pre-gap* evidence; it never claims to track a vessel with zero observations ever).
- Claims that a camera system sees through fog.
- AIS, radar, SAR/satellite, or chart fusion **as model input in the fair comparison** (AIS appears only as a withheld/available ablation and as the separate upper-bound arm H).
- Generative language assistant functionality.

## 4. Dataset Plan

### 4.1 Primary benchmark: ViV-ReID [HARD TEST anchor]

Use **ViV-ReID** as the primary benchmark, subject to license/access approval. It is a video-based, cross-port vessel Re-ID benchmark with 480 vessel identities, 20 camera views, 7,165 tracklets, and approximately 1.14 million frames. It is directly relevant because it evaluates identity association across video tracklets and camera viewpoints rather than only static image retrieval.

**Dataset decision (rationale).** The hard-test protocol requires (a) an identity pool large enough to build distractor pools of 5–20 *visually similar* vessels with a chance baseline of 1/10 or worse, and (b) genuine cross-camera tracklet structure to create natural disappearance gaps. ViV-ReID provides both (480 identities, 20 cameras). The MVTD dataset proposed in the JEPA analysis (182 sequences, ~150,000 frames) has only **four vessel classes**, which cannot support a 5–20 vessel distractor pool; it is retained as an optional Tier-3 external-validation tier (§4.4), not as the primary anchor. FVessel's contribution is its **AIS–video pairing methodology** for identity ground truth, which the extension tier reuses.

Expected local layout:

```text
data/raw/viv-reid/
  train/
  query/
  gallery/
  metadata/
  splits/
```

Create an adapter so exact upstream folder names can be mapped by configuration rather than hard-coded.

### 4.2 Secondary benchmark: VesselReID

Support **VesselReID** as an image-based supplementary benchmark, subject to data access. The dataset contains 30,587 images of 1,248 vessels captured across times, viewpoints, and weather conditions. Use it only for image-level transfer and ablation, not as the main video test.

### 4.3 Dataset manifest

For every sample/tracklet, normalize metadata into this schema:

```json
{
  "tracklet_id": "string",
  "vessel_id": "string",
  "camera_id": "string",
  "split": "train|query|gallery",
  "frame_paths": ["string"],
  "timestamp_start": "optional ISO-8601",
  "timestamp_end": "optional ISO-8601",
  "orientation": "optional string",
  "vessel_type": "optional string",
  "weather": "optional string",
  "occlusion_level": "none|partial|severe|unknown",
  "truncation_level": "none|partial|severe|unknown",
  "quality_score": "optional float",
  "source_dataset": "viv_reid|vesselreid|mvtd|fvessel|custom"
}
```

Do not infer protected or missing metadata. Store `unknown` when source annotations do not support a field.

### 4.4 Gap-trials manifest [HARD TEST]

The hard test is defined over **trials**, each pairing a pre-gap query observation with a post-gap gallery observation of the same vessel plus a distractor pool. Trials are generated by a deterministic builder and stored separately:

```json
{
  "trial_id": "string",
  "query_tracklet_id": "string",
  "gallery_tracklet_id": "string",
  "vessel_id": "string",
  "gap_seconds": "optional float",
  "gap_duration_source": "timestamp|frame_count|unknown",
  "gap_type": "natural_cross_camera|synthetic_within_tracklet",
  "maneuver_during_gap": "straight|maneuver|unknown",
  "distractor_pool_id": "string",
  "pool_size": 10,
  "ais_available_at_test": false,
  "split": "test"
}
```

Rules:

- **Natural trials** use two separate tracklets of the same vessel identity across cameras; `gap_seconds` is the time between the query tracklet's last frame and the gallery tracklet's first frame. If the dataset has no timestamps, estimate from frame count × nominal frame rate and record `gap_duration_source: frame_count`; report this approximation in every result.
- **Synthetic trials** hold out a contiguous block of a single tracklet (e.g., the middle 50%): query = frames before the block, gallery = frames after the block, and the removed block *is* the disappearance. This guarantees coverage of every gap bin even where natural cross-camera gaps are sparse.
- **Distractor pools** are built once, deterministically (fixed seed), from gallery tracklets that are (i) co-present in the same temporal window as the target when timestamps exist, or same-camera/same-vessel-class otherwise, and (ii) ranked by similarity to the target under a single **neutral reference embedding** (a fixed, frozen DINOv2 or VesselReID-pretrained OSNet, recorded in the manifest). The same pools are used for every arm; pool membership is *not* chosen per-arm and never uses V-JEPA or OpenVLA features.
- Chance baseline per trial is exactly `1 / pool_size` for Top-1. Pools must be ≥ 5 and ≤ 20 vessels.

### 4.5 Optional extension tier: MVTD + FVessel-style AIS ground truth [HARD TEST]

As Tier 3 only, optionally validate on the Maritime Visual Tracking Dataset (MVTD) combined with FVessel's AIS–video pairing methodology:

- Use AIS to establish true identity and gap durations across separated capture events, then **strip AIS from the model inputs** exactly as in the primary protocol.
- Because MVTD has only four vessel classes, it **cannot** host the full distractor-pool protocol; treat it as a real-world sanity check of gap-degradation behavior (does the slope pattern replicate on true camera footage with genuine AIS-derived gaps?), not as the primary evidence.
- If ViV-ReID access fails, do not silently substitute: mark the dataset arm `dataset_blocked` (same convention as Arm C's `blocked_by_api`) and report what ran on VesselReID/MVTD instead.

## 5. Fair Comparison Contract

All **representation arms** (A–E below) must use:

- Identical official train/query/gallery splits.
- Identical input sampling policy per experiment.
- Identical crop resolution where the encoder allows it; record unavoidable encoder-specific resizing.
- Identical temporal aggregator architecture and trainable parameter budget where possible.
- Identical Re-ID losses, optimizer family, epoch budget, early-stopping rule, and random seeds.
- Identical gallery protocol, distance metric, and optional re-ranking policy.
- No access to labels from query/gallery identities during representation pretraining or probe selection.
- **Identical gap-trials, distractor pools, and AIS-withheld subsets for all arms.** [HARD TEST]

### External-control scope [HARD TEST]

- Arms **F** (dead-reckoning) and **G** (tracker + re-ID) are external controls, not representation arms: they use whatever inputs their method defines (positions; track association), and are *not* bound by the shared-head budget. They are mandatory because they define the bar that "prior practical methods" set.
- Arm **H** (AIS-fused association) runs with AIS available and is reported **separately** as an upper-bound reference. It is not part of the fair vision-only comparison and must never be quoted as "beaten" by a vision-only arm without that caveat.

### Important interpretation rule

OpenVLA is trained for robot action prediction, not vessel Re-ID. This experiment evaluates its **visual representation path**, not its native action-generation capability. OpenVLA uses a fused DINOv2 + SigLIP visual encoder upstream of a language/action model.

Report this comparison as:

> V-JEPA predictive-video representation versus VLA-derived vision representation for vessel Re-ID.

Do not claim that this evaluates a VLA in its intended robotics-control setting.

## 6. Model Arms

### A. Conventional Re-ID baseline

Implement a strong, reproducible image/video Re-ID baseline:

- Backbone: `resnet50_ibn_a` or `osnet_x1_0` configurable.
- Initialization: ImageNet pretrained permitted; document source/version.
- Frame encoder output: L2-normalized embedding.
- Temporal aggregation: mean pooling initially; temporal attention as a shared ablation.
- Loss: cross-entropy identity classification + batch-hard triplet loss.

This is the minimum competitive baseline. Do not compare V-JEPA only with an untrained or weak CNN.

### B. V-JEPA 2.1 encoder-only probe

- Load an official V-JEPA 2.1 checkpoint through Hugging Face `AutoModel` or the official implementation.
- Freeze all V-JEPA parameters for the primary experiment.
- Sample a fixed-length clip from every tracklet.
- Extract temporal/spatial tokens and pool them with the common temporal aggregator.
- Train only the aggregator, projection layer, classifier, and metric-learning head.

V-JEPA 2 official checkpoints are available through Hugging Face and support feature extraction through `AutoModel`.

### C. V-JEPA 2.1 encoder + predictor probe

This arm tests the world-model-specific hypothesis — it is the arm most directly aligned with the disappearance-gap task.

- Use observed clip tokens from the frozen encoder.
- Mask a held-out future segment or final portion of the clip using the repository-supported predictor interface.
- Extract predictor-produced latent tokens for that future segment.
- Concatenate or cross-attend observed and predicted latent tokens.
- Apply the same trainable parameter budget as Arm B as closely as possible.

**[HARD TEST] Gap-conditioned predictor usage.** For gap trials, the observed query clip ends exactly at the disappearance point *T*. Arm C may (in addition to the standard usage above) run the predictor over the gap interval to produce latent tokens for `[T, T + Δ]` (Δ configurable, default = the trial's gap bin upper bound, capped by predictor horizon) and match those predicted tokens against gallery latents at reappearance. This makes Arm C's mechanism literally "predict the latent state of the not-yet-visible future." Log `predictor_horizon_used` per trial. The trainable parameter budget for the head remains identical to Arm B; only the *frozen* predictor is invoked.

Primary hypothesis:

> If predictive latent state is useful for vessel identity persistence, Arm C should outperform Arm B most strongly in sparse, occluded, and long-gap tracklets.

If the public V-JEPA interface does not expose predictor features reliably, mark Arm C as `blocked_by_api` and run Arms A, B, and D without silently substituting a different implementation.

### D. VLA-derived visual-representation probe

Implement two modes, clearly labelled:

1. **OpenVLA fused vision features**: Extract DINOv2 + SigLIP/Prismatic visual tokens before the language-model projection, if supported by the selected checkpoint/API.
2. **Fallback vision-only VLA components**: Concatenate the exact DINOv2 and SigLIP components documented for the selected OpenVLA family when direct fused-token extraction is impractical.

- Freeze all VLA components.
- Do not provide language prompts in the primary fair comparison, because Re-ID is not a language-conditioned task.
- Use the common temporal aggregator/projection/Re-ID loss.
- Log whether direct OpenVLA features or component-level fallback was used.

OpenVLA's documented visual pathway fuses DINOv2 and SigLIP features before projection into a Llama 2 language-model space.

### E. Diagnostic representation controls

These controls improve interpretation:

- **E1. DINOv2 alone — mandatory for the hard test.** A static-image embedding + cosine similarity is one of the baselines the hard test must beat ([JEPA strengths analysis](https://ginwind.github.io/VLA-JEPA/)); it represents "appearance memory without temporal prediction."
- **E2. SigLIP alone — mandatory for the hard test**, for the same reason.
- E3. VLA fused DINOv2 + SigLIP vision features (shared with Arm D mode 1/2).
- E4. Image-level V-JEPA frame features, if supported.
- E5. Oracle temporal aggregation using all frames, only as an upper-bound diagnostic.

These distinguish the contribution of video prediction from simply using a stronger visual backbone. E1/E2 are *mandatory* because the disappearance-gap protocol explicitly requires beating static-image Re-ID embeddings.

### F. Motion-only dead-reckoning control [HARD TEST]

- Kalman (or particle) filter over the query tracklet's last observed positions and velocities.
- Rank gallery candidates by nearest-position matching at the gallery tracklet's first-frame position (with uncertainty radius from the filter).
- No appearance input at all. This control defines the "appearance-free" ceiling and validates that long-gap trials are genuinely hard for kinematic extrapolation alone.
- Report per-gap-bin Top-1 against the same distractor pools; it should **collapse toward chance (≤ 2× chance) at the long-gap bin** — if it does not, the trials are too easy (see §8.2 validity gates).

### G. Conventional tracker with re-ID head control [HARD TEST]

- ByteTrack or BoT-SORT-style association **with its default appearance embedding** (documented per tracker; e.g., OSNet or the tracker's shipped embedding — never V-JEPA features, so this stays an independent practical standard).
- Run on the same gap trials; the tracker's association buffers define how far its memory can carry identity across the disappearance gap.
- This is the current maritime practical standard and a required baseline to beat.

### H. AIS-fused association upper bound (separate) [HARD TEST]

- DeepSORVF-style fusion or a simple AIS-identity + kinematic association using AIS labels available at test time.
- Run **only** on the `ais_available_at_test: true` trials, reported in a separate table.
- Purpose: establish the practical ceiling and calibrate how much of the vision-only gap is worth closing. Never merged into the fair comparison metrics.

## 7. Tracklet Sampling and Preprocessing

### Canonical temporal sampling

Default configuration:

```yaml
frames_per_tracklet: 16
sample_mode: uniform
input_resolution: 224
short_tracklet_policy: repeat_last
long_tracklet_policy: uniform_subsample
```

Add configurable sampling regimes:

- `uniform`: frames distributed across full tracklet.
- `recent`: last contiguous window; tests near-term evidence.
- `sparse`: every kth frame / temporal-gap simulation.
- `prefix_only`: first 25%, 50%, or 75% of a tracklet; test future identity retrieval from incomplete observation.
- `occlusion_stress`: synthetic masks applied only at evaluation time (diagnostic; see §8.4).

### 7.1 Disappearance-gap protocol configuration [HARD TEST]

```yaml
gap_protocol:
  enabled: true
  gap_bins_seconds:            # used when timestamps exist
    short: [10, 30]
    medium: [60, 120]
    long: [300, inf]
  gap_bins_frames:             # frame-count fallback when timestamps are unknown
    short: [25, 90]
    medium: [150, 360]
    long: [900, inf]
  pool_sizes: [5, 10, 20]
  min_pool_similarity: high    # prefer same class / size / livery
  maneuver_slices: [straight, maneuver, unknown]
  ais_modes: [withheld, available]
  synthetic_gap_holdout: 0.5   # contiguous fraction removed for synthetic trials
  predictor_horizon_delta: 60  # seconds Arm C predicts past the disappearance point
  seed: 42
```

Trial construction (deterministic, evaluation-only, seed-fixed):

1. Enumerate same-identity tracklet pairs with `camera_id` differing (natural trials) or single tracklets long enough for a contiguous hold-out (synthetic trials).
2. Compute gap duration (timestamp or frame-count); assign to a bin. Reject trials whose gap cannot be binned (record as `unknown` for reporting, exclude from bin analyses but keep in the all-gaps composite).
3. Build distractor pools of sizes 5/10/20 from co-present, visually similar gallery tracklets using the neutral reference embedding (§4.4).
4. For each trial emit query clip (pre-gap), gallery clip (post-gap), pool, maneuver label, and `ais_available_at_test` flag.
5. Require ≥ 500 trials per (gap bin × pool size) cell where data allows; report actual counts and reject analyses on cells with < 50 trials.

### Image processing

- Preserve aspect ratio before crop/pad as specified by model processor.
- Record all resize/crop transforms in experiment metadata.
- Never use augmentation in query/gallery evaluation.
- Ensure no accidental horizontal flip if vessel orientation is considered an identity-relevant cue; make it a configurable train-only ablation.

## 8. Difficulty and Stress Tests

The project is valuable only if it demonstrates where predictive representations help. Every run must produce both overall metrics and condition-specific metrics.

### 8.1 The "too-easy" trap [HARD TEST]

Same-frame occlusion tests (occluder over part of the vessel while it remains visible) are **diagnostics only** in this experiment. Classical trackers with memory already handle them, and the literature reports only incremental gains there. Claims of a predictive-representation advantage must be established on the disappearance-gap protocol (§8.2), not on same-frame occlusion.

### 8.2 Primary hard test: disappearance-gap re-acquisition (DGRA) [HARD TEST]

The DGRA protocol from §4.4/§7.1 is the headline evaluation. Validity gates (pre-registered, checked before interpreting any arm comparison):

1. **Dead-reckoning sanity**: Arm F Top-1 at the long-gap bin ≤ 2× chance for pool size ≥ 10. If F exceeds this, gaps are too short or distractors too easy — fix the trial construction before comparing arms.
2. **Chance floor**: random-rank Top-1 ≈ 1/K within bootstrap CI on every reported cell.
3. **No leakage**: query/gallery identities, distractor membership, and AIS labels are never visible to representation pretraining or probe training.

### 8.3 Natural condition slices

Where annotations are available:

- Occluded vs non-occluded.
- Truncated vs non-truncated.
- Same-camera vs cross-camera.
- Short vs long tracklet.
- Small vs large apparent vessel size.
- Orientation change: low / medium / high.
- Weather/visibility strata where provided.
- Same vessel type vs different vessel type candidate confusions.
- **[HARD TEST]** Gap bin: short / medium / long.
- **[HARD TEST]** Maneuver during gap: straight / maneuver / unknown.
- **[HARD TEST]** Distractor pool size: 5 / 10 / 20.
- **[HARD TEST]** AIS: withheld / available.

### 8.4 Controlled diagnostic suite (secondary evidence only)

Build deterministic evaluation-only transforms with fixed severity levels and saved random seeds:

| Stressor | Levels | Purpose |
|---|---|---|
| Block occlusion | 0%, 15%, 30%, 50% masked area | Diagnostic: partial visibility (do **not** use alone to claim a world-model win) |
| Horizon/sea-glare overlay | low, medium, high | Tests marine visual noise |
| Blur/downsampling | mild, medium, severe | Simulates range/compression |
| Haze/contrast attenuation | mild, medium, severe | Simulates degraded visibility; not true fog penetration |
| Frame dropout | 0%, 25%, 50%, 75% (scattered) | Diagnostic: sparse observation |
| **Full contiguous disappearance** | 100% of a contiguous block, durations per gap bins | **The DGRA case; primary evidence** |
| Time-gap retrieval | short, medium, long quantiles | Tests identity persistence; feeds DGRA bins |
| Crop truncation | 10%, 25%, 40% edge loss | Tests partial vessel views |

Do not describe synthetic haze as "seeing through fog." It tests robustness after degraded visual input only. Scattered frame dropout is not a substitute for full disappearance: a model that only needs *some* observed frame from the middle of the event is not being asked to predict across a gap.

### 8.5 Pre-registered success pattern

Intended evidence for a predictive-video advantage (clean data):

1. Arm B is non-inferior or superior to Arm A on clean data.
2. Arm B materially outperforms Arm A in at least two difficult condition slices.
3. Arm C improves over Arm B in at least one temporal/sparse/occlusion slice.
4. Arm B or C exceeds Arm D on the primary difficult-condition composite metric.

**[HARD TEST] Intended evidence for the disappearance-gap advantage** (the dramatic-gain pattern the protocol is designed to expose):

1. **Widening advantage**: Arm B/C's advantage over Arms A, D/E1/E2, F, G **increases with gap bin** — i.e., `accuracy_B - accuracy_A` at long-gap ≥ `accuracy_B - accuracy_A` at short-gap, with a positive slope difference whose 95% bootstrap CI excludes zero.
2. **Baseline collapse**: Arm F and Arms A/D/E1/E2 fall to ≤ 2× chance at the long-gap bin, while Arm B/C stays ≥ 2× chance (pool ≥ 10), and this holds on the **AIS-withheld** subset (vision-only).
3. **Predictor value**: Arm C beats Arm B at the long-gap bin by ≥ 2 absolute Top-1 points (95% CI excluding zero), i.e., the explicitly predicted future latent is worth more than the observed pre-gap latent alone when the gap is long.
4. **VLA comparison**: Arm B or C beats Arm D by ≥ 3 absolute mAP points on the DGRA composite, with no more than 1 point loss on clean overall mAP.

## 9. Metrics

### Primary metrics

- **mAP**: mean average precision for query-to-gallery retrieval.
- **CMC Rank-1, Rank-5, Rank-10**.
- **mINP** if standard evaluator supports it, to measure hardest-positive retrieval quality.

### Video-track metrics

- Tracklet-level mAP and CMC after temporal aggregation.
- Pairwise verification ROC-AUC and EER.
- Temporal-gap retrieval mAP.
- Re-acquisition rate after frame dropout / occlusion.

### Disappearance-gap metrics [HARD TEST]

For each (gap bin × pool size × AIS mode) cell:

- **Top-1 / Top-5 identity accuracy** against the distractor pool.
- **Chance-normalized accuracy**: `(acc - 1/K) / (1 - 1/K)`; 0 = chance, 1 = perfect.
- **mAP over the distractor pool**.
- **Degradation curve**: accuracy vs gap-duration bin (plotted on a log-gap axis), one curve per arm.
- **Degradation slope**: linear-fit slope of chance-normalized accuracy vs log gap; flatter = more robust.
- **Area under degradation curve** for each arm.
- **Baseline-collapse indicator**: whether an arm's long-gap Top-1 ≤ 2× chance.
- **Advantage-widening statistic**: paired-arm slope difference and per-bin advantage differences, with bootstrap CI (see §8.5).

### Robustness metrics

For each diagnostic stressor and severity level:

```text
absolute_score
score_drop_from_clean
relative_retained_performance = stressed_score / clean_score
area_under_corruption_curve
```

### Statistical requirements

- Run at least 5 independent seeds for probe training.
- Report mean, standard deviation, and 95% bootstrap confidence interval.
- Bootstrap at vessel identity level, not only frame level.
- Use paired bootstrap differences between arms on identical queries and **identical gap trials**.
- Report effect size, not only p-value.

### Target decision thresholds

These are research gates, not claims of prior art:

| Gate | Decision criterion |
|---|---|
| Baseline validity | Arm A reproduces or approaches published benchmark performance if official numbers are available |
| JEPA practical win | Arm B/C gains at least 3 absolute mAP points or 5% relative mAP over Arm A in difficult-condition composite, with 95% CI excluding zero |
| Predictive-state win | Arm C exceeds Arm B by at least 2 absolute mAP points in sparse/occlusion composite |
| VLA comparison | Arm B or C beats Arm D by at least 3 absolute mAP points on difficult-condition composite, with no more than 1 point loss on clean overall mAP |
| **[HARD TEST] DGRA validity** | Arm F long-gap Top-1 ≤ 2× chance (pool ≥ 10); chance floor within CI |
| **[HARD TEST] DGRA dramatic win** | (i) Arm B/C long-gap chance-normalized Top-1 ≥ 2× chance on AIS-withheld subset; (ii) advantage over A/D/E1/E2/F/G widens with gap bin (slope-difference CI excludes zero); (iii) Arm C ≥ +2 Top-1 over B at long-gap bin |
| No-go | If gains are within confidence intervals or only appear on clean imagery, do not claim a world-model advantage |

## 10. Experiment Matrix

Minimum mandatory runs:

| ID | Representation | Temporal source | Trainable component | Test set |
|---|---|---|---|---|
| A1 | CNN Re-ID | observed frames | shared temporal head | clean official split |
| B1 | V-JEPA encoder | observed clip tokens | shared temporal head | clean official split |
| C1 | V-JEPA encoder + predictor | observed + predicted tokens | shared temporal head | clean official split |
| D1 | OpenVLA-derived vision | observed frames/tokens | shared temporal head | clean official split |
| A2–D2 | same as above | same | same | natural occlusion/truncation slices |
| A3–D3 | same as above | same | same | controlled frame-dropout suite |
| A4–D4 | same as above | same | same | controlled block-occlusion suite |
| A5–D5 | same as above | same | same | time-gap and cross-camera subsets |
| **[HARD TEST]** A6–D6, E1, E2 | same as above | same | same | DGRA: all gap bins × pool sizes, AIS withheld |
| **[HARD TEST]** C7 | V-JEPA encoder + predictor (gap-conditioned, §6.C) | observed + gap-predicted tokens | shared temporal head | DGRA long-gap bin, AIS withheld |
| **[HARD TEST]** F1 | Kalman/particle dead-reckoning | positions only | none | DGRA: all gap bins × pool sizes |
| **[HARD TEST]** G1 | ByteTrack/BoT-SORT + appearance embedding | track association | tracker defaults | DGRA: all gap bins × pool sizes |
| **[HARD TEST]** H1 | AIS-fused association | AIS + vision | fusion defaults | DGRA: AIS-available subset only (separate table) |

Optional but recommended:

- Fine-tuned V-JEPA low-rank adapter run, separately labelled `adapted`.
- Self-supervised maritime adaptation with unlabeled training-only tracklets, with no query/gallery leakage.
- Late-fusion feature baseline: V-JEPA + VLA features with the same classifier budget.
- **[HARD TEST]** Probes trained with gap-augmented sampling (query = prefix-only clips, gallery = suffix clips), labelled `gap_augmented` and never merged into the headline comparison.

## 11. Software Architecture

### Repository layout

```text
maritime-reid-worldmodel/
  README.md
  LICENSE
  pyproject.toml
  requirements/
    base.txt
    vjepa.txt
    openvla.txt
  configs/
    data/
    models/
    experiments/
    gap/                     # NEW: DGRA protocol configs
  data/
    README.md
    manifests/
    gap_trials/              # NEW: generated gap-trial manifests
  src/
    data/
      adapters/
      manifest.py
      splits.py
      sampling.py
      augmentations.py
      gap_trials.py          # NEW: disappearance-gap trial builder
      distractor_pool.py     # NEW: similarity-ranked, seeded distractor pools
    models/
      common_head.py
      cnn_reid.py
      vjepa_adapter.py
      vjepa_predictor_adapter.py
      openvla_adapter.py
      temporal_pooling.py
      baselines/
        kalman_deadreckon.py # NEW: Arm F
        tracker_reid.py      # NEW: Arm G (ByteTrack/BoT-SORT wrapper)
        ais_upper_bound.py   # NEW: Arm H
    training/
      losses.py
      trainer.py
      callbacks.py
    evaluation/
      reid_metrics.py
      robustness.py
      bootstrap.py
      degradation.py         # NEW: gap-degradation curves and slopes
      reports.py
    utils/
      reproducibility.py
      logging.py
      registry.py
  scripts/
    prepare_dataset.py
    build_gap_trials.py      # NEW
    extract_features.py
    train_probe.py
    evaluate.py
    run_stress_suite.py
    run_baselines.py         # NEW: F/G/H controls
    aggregate_results.py
  tests/
  outputs/
    .gitkeep
```

### Interfaces

Every representation adapter must implement:

```python
class TrackletEncoder(Protocol):
    name: str
    embedding_dim: int

    def preprocess(self, frames: Tensor) -> Tensor: ...

    @torch.no_grad()
    def encode_observed(self, frames: Tensor, frame_mask: Tensor) -> Tensor:
        """Return [batch, time_or_tokens, dim] features."""

    @torch.no_grad()
    def encode_predicted(self, frames: Tensor, frame_mask: Tensor) -> Tensor | None:
        """Return predictor-derived latent features or None if unsupported."""
```

**[HARD TEST]** The gap-conditioned predictor usage adds one optional method (Arm C only; all other adapters return `None`):

```python
    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> Tensor | None:
        """Return latent tokens predicted for the disappearance interval, or None."""
```

The common Re-ID head must accept any token sequence:

```python
class SharedReIDHead(nn.Module):
    def forward(self, tokens: Tensor, token_mask: Tensor) -> dict[str, Tensor]:
        """Return embedding, logits, attention weights."""
```

Baseline controls implement a distinct protocol (they are not representation adapters):

```python
class GapBaseline(Protocol):
    name: str
    def rank(self, trial: GapTrial, gallery_features: Mapping[str, Tensor]) -> list[str]:
        """Return gallery tracklet ids ranked best-to-worst for the trial."""
```

### Configuration

Use Hydra or equivalent YAML composition. Every executable run must write a fully resolved configuration file to its output directory.

Example experiment configuration (extended):

```yaml
experiment:
  name: vivreid_vjepa_vs_vla_dgra
  seed: 42

data:
  dataset: viv_reid
  manifest_path: data/manifests/viv_reid.jsonl
  frames_per_tracklet: 16
  sample_mode: uniform

gap_protocol:
  enabled: true
  gap_bins_seconds: {short: [10, 30], medium: [60, 120], long: [300, null]}
  pool_sizes: [5, 10, 20]
  synthetic_gap_holdout: 0.5
  ais_modes: [withheld, available]
  seed: 42

model:
  arm: vjepa_encoder
  frozen_backbone: true
  temporal_head: attention

training:
  epochs: 60
  batch_size: 32
  optimizer: adamw
  lr: 0.0003
  losses:
    id_ce_weight: 1.0
    triplet_weight: 1.0

evaluation:
  metrics: [map, rank1, rank5, rank10, minp]
  bootstrap_samples: 2000
  degradation_curve_bins: [short, medium, long]

stress:
  enabled: true
  suites: [block_occlusion, frame_dropout, haze, time_gap, disappearance_gap]
```

## 12. Training Details

### Shared head

Default head:

```text
Input tokens
→ LayerNorm
→ 2-layer temporal Transformer or attention pooling
→ projection MLP (embedding dimension 512)
→ L2 normalization
→ identity classifier (training only)
```

Train only this head for frozen-backbone comparisons. Arms B, C, and D share the head; Arm C additionally invokes only the frozen predictor (§6.C).

### Losses

```text
L_total = L_cross_entropy + lambda_triplet * L_batch_hard_triplet
```

Optional ablation:

- Supervised contrastive loss.
- Circle loss.

Do not change loss functions per model arm in the headline comparison.

**[HARD TEST]** The DGRA protocol is **evaluation-only** for the headline comparison: probes are trained on the standard Re-ID objective, and the disappearance-gap trials are constructed only at test time. Gap-augmented probe training (query = pre-gap prefix, gallery = post-gap suffix) is a separately labelled ablation, never the headline run.

### Identity split hygiene

- Fit classifier only on training identities.
- Query and gallery identities may be shared with each other according to the official Re-ID protocol, but neither may leak into training identities unless the official benchmark explicitly defines a closed-set protocol.
- Validate model/hyperparameters only on a validation split derived from training data or official validation split.
- **[HARD TEST]** Distractor-pool membership and AIS-withheld flags are test-time metadata only; they never influence probe training or feature caching.

## 13. V-JEPA Implementation Notes

- Prefer Hugging Face Transformers for encoder-only feature extraction to reduce custom integration risk; official V-JEPA 2 checkpoint support is documented.
- Pin exact model checkpoint names and package versions.
- Cache backbone features to disk using content-addressed keys that include checkpoint, preprocessing configuration, frame sampling, and dataset manifest hash.
- Preserve temporal token order. Do not collapse to one frame embedding before testing temporal aggregation.
- For predictor experiments, use the official V-JEPA repository/API if needed; write an adapter test using a short synthetic clip before launching full extraction.
- **[HARD TEST]** For gap-conditioned prediction (Arm C), pin the predictor horizon explicitly and record it per trial; verify the predictor accepts a variable-length future horizon or document the closest supported setting.

## 14. VLA Implementation Notes

- Pin a public OpenVLA/Prismatic checkpoint and commit hash.
- Extract vision features before language-model projection whenever available.
- Record the feature source exactly: `openvla_direct_fused`, `prismatic_dinosiglip`, or `fallback_dinov2_plus_siglip`.
- Do not use generated actions as Re-ID features.
- Do not add natural-language labels/prompts to primary runs.
- If OpenVLA installation is too heavy for the primary environment, cache fused vision features in a separate container/job and feed the resulting `.pt` feature store into the common probe pipeline.
- **[HARD TEST]** The static DINOv2-alone and SigLIP-alone controls (E1/E2) use the *same* frozen components used inside the OpenVLA fusion, so the hard test directly answers "does the VLA-derived static appearance stack beat V-JEPA's temporal prediction — and does fusion of the two static encoders change that?"

## 15. Outputs and Reporting

Every experiment output directory must contain:

```text
outputs/<run_id>/
  resolved_config.yaml
  git_state.json
  environment.txt
  dataset_manifest_hash.txt
  gap_trials_hash.txt          # NEW
  metrics.json
  metrics_by_slice.csv
  robustness_curves.csv
  degradation_curves.csv       # NEW: per-arm accuracy vs gap bin
  degradation_slopes.json      # NEW
  per_query_results.parquet
  bootstrap_intervals.json
  confusion_cases.jsonl
  figures/
    overall_metrics.png
    occlusion_retention.png
    frame_dropout_retention.png
    time_gap_curve.png
    vjepa_vs_vla_difficult_subset.png
    gap_curve_rank1.png        # NEW: Top-1 vs gap bin, all arms incl. F/G/H
    advantage_widening.png     # NEW: per-bin advantage differences with CIs
  report.md
```

`report.md` must automatically state:

- Dataset and exact split.
- Model checkpoints and whether frozen/adapted.
- Feature source for VLA arm.
- Overall and subgroup metrics with confidence intervals.
- Paired differences between each arm and Arm A.
- **[HARD TEST]** Gap-trial counts per (bin × pool × AIS) cell; `gap_duration_source` (timestamp vs frame-count); whether dead-reckoning sanity gate passed.
- **[HARD TEST]** Degradation slopes and per-bin advantages with CIs; which arms collapse toward chance at long gaps.
- A short conclusion generated from deterministic rules, not by an LLM.
- Limitations: vision-only, not Hong Kong-specific, no AIS/radar fusion (except the separately-reported arm H upper bound), no proof of fog penetration, gap durations may be frame-count estimates when timestamps are unavailable.

## 16. Unit and Integration Tests

Minimum tests:

- Dataset adapter produces stable IDs and no overlap violations.
- Query/gallery manifest schema validation.
- Identical input clip produces deterministic cached feature key.
- Every adapter returns expected tensor shape and no NaNs.
- Frozen-backbone assertion verifies no gradient on backbone parameters.
- Shared head parameter count is within 5% across B, C, and D arms.
- Re-ID metric implementation matches a small known reference example.
- Stress transforms are deterministic by seed and preserve labels.
- Bootstrap groups samples by vessel identity.
- Predictor arm fails loudly if predictor feature extraction is unavailable.
- **[HARD TEST]** Gap-trial builder is deterministic by seed: same manifest in, identical `gap_trials.jsonl` out.
- **[HARD TEST]** Gap bins partition correctly under both timestamp and frame-count sources; `unknown` gaps excluded from bin analyses.
- **[HARD TEST]** Distractor pools: pool size == configured K; target vessel never in its own pool; chance baseline 1/K reproduced by random ranking on a toy example; pools independent of arm features.
- **[HARD TEST]** Synthetic trials: the removed block is contiguous and disjoint from query/gallery; labels preserved.
- **[HARD TEST]** Degradation slope computed on a toy curve matches hand-computed value; CI machinery runs on the toy example.
- **[HARD TEST]** Kalman dead-reckoning control runs end-to-end on a toy trajectory and ranks candidates by position proximity.
- **[HARD TEST]** Tracker control (G) wrapper instantiates with pinned tracker version and runs on one trial set; AIS-upper-bound (H) refuses to run on `ais_available_at_test: false` trials.

## 17. Compute Strategy

### Tier 1: Feasibility smoke test

- 10–20% of training tracklets.
- 3 seeds.
- Frozen feature extraction.
- One GPU sufficient for probe training after feature caching.
- **[HARD TEST]** Include a reduced DGRA cell set (one pool size, one gap bin per mode) so gap-trial construction, baseline controls F/G, and degradation curves are exercised end-to-end.

Deliverable: pipeline works end-to-end, metrics, slices, and degradation curves render.

### Tier 2: Primary benchmark

- Full ViV-ReID split.
- 5 seeds.
- All mandatory arms and stress suites, plus the full DGRA matrix (A6–D6, C7, E1/E2, F1, G1, H1).
- Feature caching required. Gap trials and distractor pools built once and reused.

### Tier 3: Research extension

- Predictor arm with official V-JEPA code and gap-conditioned future prediction (C7 full horizon sweep).
- Low-rank adaptation or maritime self-supervised adaptation.
- External validation on MVTD + FVessel-style AIS ground truth (§4.5).

## 18. Acceptance Criteria

The coding deliverable is accepted when:

1. A new researcher can reproduce one complete baseline run from README instructions.
2. The primary dataset adapter validates its split and creates a manifest.
3. Arms A, B, and D run end-to-end; Arm C either runs or reports an explicit documented API blocker.
4. All arms share the same evaluation protocol and output format.
5. The results report overall and difficult-condition metrics with confidence intervals.
6. The stress suite runs deterministically and produces robustness curves.
7. **[HARD TEST]** `build_gap_trials.py` produces deterministic gap trials and distractor pools; validity gates (dead-reckoning sanity, chance floor) are computed and reported.
8. **[HARD TEST]** Degradation curves and slopes render for arms A–G on the DGRA cells; arm H reports separately on AIS-available trials.
9. The report makes no unsupported claim that predictive models see through fog or prove vessel identity without observations.
10. The repository contains an explicit comparison conclusion for JEPA vs VLA-derived visual features, including the disappearance-gap regime.

## 19. Phase 2 Boundary: Full Multimodal World Model

Do not implement this in Phase 1, but design feature/output schemas so it can be added later.

Phase 2 would combine:

```text
JEPA vessel video features
+ AIS identity/position/velocity
+ radar tracks
+ satellite observations
+ weather and visibility
+ maritime chart and traffic-lane context
→ uncertainty-aware global vessel belief state
→ local-to-global observation updates
→ global-to-local re-acquisition search cue
```

The Phase 1 Re-ID embedding becomes one likelihood term in a broader association score:

\[
S(i,j) = w_v S_{visual} + w_k S_{kinematic} + w_a S_{AIS} + w_r S_{radar} + w_c S_{context}
\]

**[HARD TEST]** The DGRA protocol is the natural Phase-1 bridge: its `ais_available_at_test` flag, maneuver labels, and gap bins are exactly the conditioning axes Phase 2's fusion score would use. The AIS-available arm H defines the Phase-2 ceiling; the AIS-withheld DGRA results define how much of that ceiling vision-only prediction can close. Phase 2 must retain explicit uncertainty and should use human-authorized tasking for any physical platform.

## 20. README Requirements

The README must include:

- Research question and non-claims.
- Dataset access/licensing steps; do not redistribute restricted data.
- Environment creation commands.
- Checkpoint download/access instructions.
- Commands for feature extraction, training, evaluation, stress suite, gap-trial construction, baseline controls, and report generation.
- Expected directory layout.
- How to add a new encoder adapter and a new gap baseline control.
- Reproducibility statement and known limitations (including frame-count gap estimation when timestamps are absent).

## 21. Suggested Initial Commands

```bash
python scripts/prepare_dataset.py --config-name viv_reid
python scripts/build_gap_trials.py --config-name viv_reid_dgra
python scripts/extract_features.py experiment=vivreid_cnn
python scripts/extract_features.py experiment=vivreid_vjepa_encoder
python scripts/extract_features.py experiment=vivreid_openvla_vision
python scripts/train_probe.py experiment=vivreid_cnn
python scripts/train_probe.py experiment=vivreid_vjepa_encoder
python scripts/train_probe.py experiment=vivreid_openvla_vision
python scripts/run_baselines.py experiment=vivreid_dgra   # Arms F, G, H
python scripts/evaluate.py experiment=vivreid_vjepa_encoder protocol=dgra
python scripts/run_stress_suite.py experiment=vivreid_all_arms
python scripts/aggregate_results.py --runs outputs/vivreid_*
```

## 22. Interpretation Language for Final Results

Use only one of these evidence-grounded conclusions (extended with the disappearance-gap regime):

- **Positive (general):** "On this benchmark, frozen V-JEPA video features improved vessel Re-ID robustness under specified partial-observation conditions relative to the tested CNN and VLA-derived visual baselines."
- **Positive (gap regime):** "Under the disappearance-gap protocol, V-JEPA-based matching stayed above chance at long gaps and its advantage over static-embedding, dead-reckoning, and tracker baselines widened with gap duration, with AIS withheld." (Only if §8.5 hard-test criteria 1–2 hold, including the dead-reckoning sanity gate.)
- **Mixed:** "V-JEPA improved selected difficult-condition slices (including specified gap bins) but did not show a reliable overall advantage; predictive representation benefit is conditional on the observed regime." State exactly which gap bins and pool sizes showed the effect.
- **Negative:** "The tested V-JEPA configuration did not outperform strong conventional or VLA-derived visual baselines, including under the disappearance-gap protocol; no world-model advantage is established by this experiment."

Never conclude that results establish superiority for collision avoidance, maritime surveillance as a whole, ESG compliance, or true multimodal world modelling. Those require the Phase 2 data and evaluation design. Never present the AIS-available arm H comparison as a vision-only win.

---

## Sources for the hard-test design

- Meta AI, *Introducing V-JEPA 2* — benchmark numbers for action anticipation and Something-Something v2 that motivate the "predict across a gap, then verify" structure: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/
- Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv 2506.09985: https://arxiv.org/abs/2506.09985
- VLA-JEPA project page (video-predictive embeddings for VLA models): https://ginwind.github.io/VLA-JEPA/
- Occluded action benchmarks (why same-frame occlusion is an easy regime): https://github.com/rajatmodi62/OccludedActionBenchmark
- OccludeNet dataset: https://github.com/The-Martyr/OccludeNet-Dataset
