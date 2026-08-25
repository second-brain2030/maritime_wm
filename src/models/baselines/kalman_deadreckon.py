"""Arm F: motion-only dead-reckoning control (spec section 6.F).

Ranks gallery tracklets by extrapolated position: the query centroid plus
velocity times the gap. ``predict_position`` fits a constant-velocity Kalman
filter over the observed positions and projects ``gap_steps`` ahead with the
filter's uncertainty. No appearance input.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class GapBaseline(Protocol):
    """Position-only baseline protocol: rank gallery centroids nearest-first."""

    name: str

    def rank(
        self,
        query_centroid: np.ndarray,
        query_velocity: np.ndarray,
        gap_seconds: float,
        gallery_centroids: dict[str, np.ndarray],
    ) -> list[str]: ...


class KalmanDeadReckon:
    """Constant-velocity Kalman dead-reckoning baseline."""

    name = "kalman_deadreckon"

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
    ) -> None:
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        # 4-state constant-velocity KF: state=[x, y, vx, vy], measure=[x, y]
        self.dim_x = 4
        self.dim_z = 2
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def _make_filter(self, initial_xy: np.ndarray):
        try:  # lazy: filterpy is an optional dependency
            from filterpy.kalman import KalmanFilter
        except ImportError as exc:  # pragma: no cover - filterpy absent
            raise ImportError(
                "KalmanDeadReckon.predict_position requires filterpy; "
                "install it (e.g. pip install filterpy) to use the Kalman path"
            ) from exc

        kf = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        kf.x = np.array([initial_xy[0], initial_xy[1], 0.0, 0.0], dtype=float)
        kf.F = self.F
        kf.H = self.H
        kf.P *= 1000.0
        kf.R = np.eye(self.dim_z) * self.measurement_noise
        kf.Q = np.eye(self.dim_x) * self.process_noise
        return kf

    def predict_position(
        self, positions: np.ndarray, gap_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the KF by feeding ``positions`` [N, 2] sequentially, then
        predict ``gap_steps`` ahead. Returns ``(predicted_xy [2],
        uncertainty_std [2])`` where the std is sqrt of the diagonal of P
        for the position components."""
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
            raise ValueError(f"positions must be [N, 2], got {positions.shape}")
        kf = self._make_filter(positions[0])
        for z in positions[1:]:
            kf.predict()
            kf.update(z)
        for _ in range(int(gap_steps)):
            kf.predict()
        return kf.x[:2].copy(), np.sqrt(np.diag(kf.P)[:2])

    def rank(
        self,
        query_centroid: np.ndarray,
        query_velocity: np.ndarray,
        gap_seconds: float,
        gallery_centroids: dict[str, np.ndarray],
    ) -> list[str]:
        """Extrapolate via the Kalman filter and rank gallery centroids
        nearest-first.

        Seeds a 2-point trajectory (t-1, t) from centroid + velocity and
        projects ``gap_seconds`` ahead through ``predict_position`` (the KF
        path). Falls back to linear extrapolation when filterpy is absent.
        """
        centroid = np.asarray(query_centroid, dtype=float)
        velocity = np.asarray(query_velocity, dtype=float)
        gap_steps = max(1, int(round(gap_seconds / self.dt)))
        try:
            positions = np.stack([centroid - velocity * self.dt, centroid])
            predicted, _ = self.predict_position(positions, gap_steps)
        except (ImportError, RuntimeError):
            # filterpy absent: fall back to linear extrapolation
            predicted = centroid + velocity * float(gap_seconds)
        return sorted(
            gallery_centroids.keys(),
            key=lambda tid: float(
                np.linalg.norm(np.asarray(gallery_centroids[tid], dtype=float) - predicted)
            ),
        )
