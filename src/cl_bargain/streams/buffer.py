"""Reservoir replay buffers, optionally with a held-out probe slice."""

from __future__ import annotations

import torch


class ReservoirBuffer:
    """Uniform reservoir sample of every example presented so far."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.n_seen = 0
        self.idx = torch.zeros(capacity, dtype=torch.long, device=device)
        self.owner = torch.zeros(capacity, dtype=torch.long, device=device)
        self.filled = 0

    def __len__(self) -> int:
        return self.filled

    def add(self, stream_idx: torch.Tensor, task: int, generator: torch.Generator) -> None:
        n = int(stream_idx.numel())
        seen = self.n_seen + torch.arange(n, device=self.device)
        self.n_seen += n

        n_fill = min(n, self.capacity - self.filled)
        slots = torch.full((n,), -1, dtype=torch.long, device=self.device)
        if n_fill:
            slots[:n_fill] = torch.arange(self.filled, self.filled + n_fill, device=self.device)
            self.filled += n_fill
        if n_fill < n:
            tail = seen[n_fill:]
            rand = torch.rand(n - n_fill, device=self.device, generator=generator)
            draws = (rand * (tail + 1).float()).long()
            slots[n_fill:] = torch.where(draws < self.capacity, draws, -1)

        keep = slots >= 0
        target = slots[keep]
        self.idx[target] = stream_idx[keep]
        self.owner[target] = task

    @property
    def task_sizes(self) -> dict[int, int]:
        owners = self.owner[: self.filled]
        return {int(t): int((owners == t).sum()) for t in torch.unique(owners).tolist()}

    def held(self, task: int) -> torch.Tensor:
        owners = self.owner[: self.filled]
        idx = self.idx[: self.filled]
        return idx[owners == task]

    def flat(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.idx[: self.filled], self.owner[: self.filled]


class ProbeReplayBuffer:
    """Replay slice (trained on) and probe slice (never trained on) as two reservoirs."""

    def __init__(self, capacity: int, probe_fraction: float, device: torch.device) -> None:
        if not 0.0 <= probe_fraction < 1.0:
            raise ValueError(f"probe_fraction must be in [0, 1), got {probe_fraction}")
        self.capacity = capacity
        self.probe_fraction = probe_fraction
        probe_capacity = round(capacity * probe_fraction)
        self.replay = ReservoirBuffer(capacity - probe_capacity, device)
        self.probe = ReservoirBuffer(probe_capacity, device) if probe_capacity > 0 else None

    def add(self, stream_idx: torch.Tensor, task: int, generator: torch.Generator) -> None:
        if self.probe is None:
            self.replay.add(stream_idx, task, generator)
            return
        is_probe = (
            torch.rand(stream_idx.numel(), device=stream_idx.device, generator=generator)
            < self.probe_fraction
        )
        if bool(is_probe.any()):
            self.probe.add(stream_idx[is_probe], task, generator)
        if bool((~is_probe).any()):
            self.replay.add(stream_idx[~is_probe], task, generator)

    def __len__(self) -> int:
        return len(self.replay)

    @property
    def task_sizes(self) -> dict[int, int]:
        """Replay-slice sizes per task."""
        return self.replay.task_sizes

    def held(self, task: int) -> torch.Tensor:
        """Replay slice for `task`."""
        return self.replay.held(task)

    def probe_held(self, task: int) -> torch.Tensor:
        """Probe slice for `task`; never replayed."""
        if self.probe is None:
            return torch.zeros(0, dtype=torch.long, device=self.replay.device)
        return self.probe.held(task)


__all__ = ["ProbeReplayBuffer", "ReservoirBuffer"]
