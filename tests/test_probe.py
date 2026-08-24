import tempfile
from pathlib import Path

import torch

from data.manifest import TrackletManifest, save_manifests
from training.probe import ProbeArtifacts, build_head, train_probe


def make_manifests(n_per_class=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    manifests = []
    for cls, vid in enumerate(("vA", "vB")):
        for k in range(n_per_class):
            manifests.append(
                TrackletManifest(
                    tracklet_id=f"{vid}_t{k}", vessel_id=vid, camera_id="c0",
                    split="train", frame_paths=[f"f{i}.jpg" for i in range(16)],
                    source_dataset="mvtd",
                )
            )
    return manifests


def test_train_probe_synthetic(tmp_path):
    # separable synthetic features: class A ~ N(+1, 0.1), class B ~ N(-1, 0.1)
    g = torch.Generator().manual_seed(0)
    feats_dir = tmp_path / "features"
    feats_dir.mkdir()
    manifests = make_manifests()
    for m in manifests:
        base = 1.0 if m.vessel_id == "vA" else -1.0
        f = torch.randn(16, 8, generator=g) * 0.1 + base
        torch.save(
            {"tracklet_id": m.tracklet_id, "vessel_id": m.vessel_id, "features": f},
            feats_dir / f"{m.tracklet_id}.pt",
        )

    artifacts = train_probe(
        features_dir=feats_dir,
        manifests=manifests,
        token_dim=8,
        embed_dim=16,
        epochs=20,
        batch_size=8,
        lr=1e-2,
        seed=1,
    )
    assert set(artifacts.class_map) == {"vA", "vB"}
    loss = artifacts.config["loss_history"]
    assert loss[-1] < loss[0]  # training progresses
    assert artifacts.config["train_tracklets"] == 16

    # head roundtrip: save -> load -> build -> forward
    p = tmp_path / "probe.pt"
    artifacts.save(p)
    loaded = ProbeArtifacts.load(p)
    head = build_head(loaded)
    out = head(torch.randn(2, 16, 8), torch.ones(2, 16, dtype=torch.bool))
    assert out["embedding"].shape == (2, 16)
    assert out["logits"].shape == (2, 2)


def test_train_probe_requires_train_features(tmp_path):
    feats_dir = tmp_path / "features"
    feats_dir.mkdir()
    manifests = make_manifests(n_per_class=1)
    manifests[0].split = "query"  # cached tracklet is NOT train -> nothing qualifies
    for m in manifests[:1]:
        torch.save(
            {"tracklet_id": m.tracklet_id, "features": torch.randn(16, 8)},
            feats_dir / f"{m.tracklet_id}.pt",
        )
    import pytest

    with pytest.raises(ValueError):
        train_probe(features_dir=feats_dir, manifests=manifests, token_dim=8, epochs=1)
