"""Three-way per-task buffer: replay / calibration / audit."""

from __future__ import annotations

import torch

from cl_bargain.streams.buffer import ReservoirBuffer

SLICES = ("replay", "calibration", "audit")


class ThreeWayBuffer:
    """Per-task buffer split into disjoint replay, calibration and audit reservoirs."""

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        split_seed: int,
        fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    ) -> None:
        if len(fractions) != 3:
            raise ValueError(f"need 3 fractions, got {len(fractions)}")
        if any(f < 0 for f in fractions):
            raise ValueError(f"fractions must be >= 0, got {fractions}")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")

        self.capacity = capacity
        self.device = device
        self.split_seed = split_seed
        self.fractions = tuple(float(f) for f in fractions)

        # floor calibration and audit; replay takes the rounding remainder
        cal_cap = int(capacity * fractions[1])
        aud_cap = int(capacity * fractions[2])
        rep_cap = capacity - cal_cap - aud_cap

        self.capacities = {"replay": rep_cap, "calibration": cal_cap, "audit": aud_cap}
        self._buf = {
            name: ReservoirBuffer(self.capacities[name], device) for name in SLICES
        }
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(split_seed)

        # cumulative routing thresholds
        self._cuts = (fractions[0], fractions[0] + fractions[1])

    def add(self, stream_idx: torch.Tensor, task: int, generator=None) -> None:
        """Route examples to slices; routing depends only on `split_seed`."""
        del generator
        n = int(stream_idx.numel())
        if n == 0:
            return
        u = torch.rand(n, device=stream_idx.device, generator=self._gen)
        assign = torch.bucketize(u, torch.tensor(self._cuts, device=stream_idx.device))
        for k, name in enumerate(SLICES):
            if self.capacities[name] == 0:
                continue
            sel = stream_idx[assign == k]
            if sel.numel():
                self._buf[name].add(sel, task, self._gen)

    def __len__(self) -> int:
        return len(self._buf["replay"])

    @property
    def task_sizes(self) -> dict[int, int]:
        """Replay-slice sizes only; these decide which players are active."""
        return self._buf["replay"].task_sizes

    def held(self, task: int) -> torch.Tensor:
        """Replay slice, under the ReservoirBuffer accessor name."""
        return self._buf["replay"].held(task)

    replay_held = held

    # replay-reservoir storage exposed as `idx[:filled]` for replay baselines
    @property
    def idx(self) -> torch.Tensor:
        return self._buf["replay"].idx

    @property
    def filled(self) -> int:
        return self._buf["replay"].filled

    def calibration_held(self, task: int) -> torch.Tensor:
        return self._buf["calibration"].held(task)

    def audit_held(self, task: int, *, for_certification: bool = False) -> torch.Tensor:
        if not for_certification:
            raise RuntimeError(
                "audit slice requested without for_certification=True. The audit "
                "slice may only be read by the certifier; reading it anywhere on "
                "the training path voids the guarantee."
            )
        return self._buf["audit"].held(task)

    def slice_sizes(self, task: int) -> dict[str, int]:
        return {
            "replay": int(self._buf["replay"].held(task).numel()),
            "calibration": int(self._buf["calibration"].held(task).numel()),
            "audit": int(self._buf["audit"].held(task).numel()),
        }

    def all_slice_sizes(self) -> dict[str, dict[int, int]]:
        return {name: self._buf[name].task_sizes for name in SLICES}

    def assert_disjoint(self) -> None:
        """Raise if any stored stream index appears in more than one slice."""
        seen: dict[str, set[int]] = {}
        for name in SLICES:
            idx, _ = self._buf[name].flat()
            seen[name] = set(idx.tolist())
        for a in range(len(SLICES)):
            for b in range(a + 1, len(SLICES)):
                na, nb = SLICES[a], SLICES[b]
                overlap = seen[na] & seen[nb]
                if overlap:
                    raise AssertionError(
                        f"{na} and {nb} slices share {len(overlap)} stream indices, "
                        f"e.g. {sorted(overlap)[:5]}"
                    )

    def tasks(self) -> list[int]:
        return sorted(self._buf["replay"].task_sizes.keys())


__all__ = ["SLICES", "ThreeWayBuffer"]
