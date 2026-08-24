"""Arm D kinematic layer (pilot brief P3): constant-velocity Kalman state
prediction across a blackout gap + global-to-local predictive search window
proposal.

Pure numpy: the kinematic state is independent of the vision backbone and is
testable with synthetic trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class KinematicState:
    x: float
    y: float
    vx: float
    vy: float
    position_cov: float = 1.0  # scalar position variance (isotropic)


class ConstantVelocityKalman:
    """Minimal constant-velocity Kalman filter over (x, y, vx, vy).

    State transition over dt: x += vx*dt; v unchanged (process noise adds
    velocity uncertainty so prediction uncertainty grows with the gap).
    """

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state: np.ndarray | None = None  # [x, y, vx, vy]
        self.cov: np.ndarray | None = None    # 4x4

    def init(self, x: float, y: float, vx: float, vy: float) -> None:
        self.state = np.array([x, y, vx, vy], dtype=float)
        # initial prior uncertainty (>= measurement noise so first update trusts
        # the measurement rather than a zero-uncertainty prior)
        self.cov = np.eye(4) * max(self.measurement_noise, 1.0)

    def predict(self, dt: float) -> np.ndarray:
        """Predict state at +dt; returns predicted [x, y] position."""
        if self.state is None:
            raise RuntimeError("Kalman filter not initialized")
        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        Q = np.diag([0.0, 0.0, self.process_noise, self.process_noise]) * dt
        self.cov = F @ self.cov @ F.T + Q
        self.state = F @ self.state
        return self.state[:2].copy()

    def update(self, x: float, y: float) -> None:
        """Measurement update (constant-velocity, position-only)."""
        if self.state is None:
            raise RuntimeError("Kalman filter not initialized")
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * self.measurement_noise
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        z = np.array([x, y], dtype=float)
        self.state = self.state + K @ (z - H @ self.state)
        self.cov = (np.eye(4) - K @ H) @ self.cov

    def uncertainty_radius(self, sigma: float = 2.0) -> float:
        """Predicted position uncertainty radius (meters/pixels) at current state."""
        if self.cov is None:
            raise RuntimeError("Kalman filter not initialized")
        pos_var = float(self.cov[0, 0] + self.cov[1, 1]) / 2.0
        return sigma * np.sqrt(max(pos_var, 1e-9))


def predict_across_gap(observations: Sequence[tuple[float, float, float]], gap_s: float) -> tuple[np.ndarray, float]:
    """Fit constant velocity from (t, x, y) observations, predict at t_end+gap.

    Returns (predicted [x, y], uncertainty radius). Uses least-squares
    velocity from the last two observations when no noise model is needed.
    """
    if len(observations) < 2:
        raise ValueError("need at least 2 observations")
    (t1, x1, y1), (t2, x2, y2) = observations[-2], observations[-1]
    dt = t2 - t1
    if dt <= 0:
        raise ValueError("observation timestamps must be strictly increasing")
    vx, vy = (x2 - x1) / dt, (y2 - y1) / dt
    pred = np.array([x2 + vx * gap_s, y2 + vy * gap_s])
    # naive uncertainty: proportional to gap length and |velocity|
    radius = 0.05 * gap_s * (1.0 + float(np.hypot(vx, vy)))
    return pred, radius


def propose_search_window(
    predicted: np.ndarray,
    uncertainty_radius: float,
    candidates: Sequence[tuple[str, np.ndarray]],
    window_factor: float = 1.0,
) -> list[str]:
    """Global-to-local predictive search window (brief P3 Arm D).

    Gate candidate gallery entries by distance from the predicted state:
    return ids whose position lies within ``window_factor * radius``.
    """
    radius = window_factor * uncertainty_radius
    kept: list[str] = []
    for cid, pos in candidates:
        if float(np.linalg.norm(pos - predicted)) <= radius:
            kept.append(cid)
    return kept
