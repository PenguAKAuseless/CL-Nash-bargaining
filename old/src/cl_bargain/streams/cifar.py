"""Split-CIFAR100 stream, held entirely on the GPU.

Torch-only module -- this and bargain/curvature.py's estimator functions are
the only places in this repo allowed to import torch (see pyproject.toml).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchvision.datasets import CIFAR100

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)

# Classes per task, per stream.
STREAMS: dict[str, tuple[int, ...]] = {
    "uniform": (10,) * 10,
    "imbalanced": (2, 3, 5, 7, 10, 10, 13, 15, 15, 20),
    "split20": (5,) * 20,
}


@dataclass(frozen=True)
class Task:
    index: int
    classes: tuple[int, ...]
    train_idx: torch.Tensor
    test_idx: torch.Tensor

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_examples(self) -> int:
        return int(self.train_idx.numel())


class SplitCIFAR100:
    """A class-incremental CIFAR-100 stream resident on ``device``."""

    def __init__(
        self,
        root: str,
        stream: str,
        *,
        device: torch.device,
        class_order_seed: int,
        download: bool = True,
        task_masked: bool = False,
    ) -> None:
        if stream not in STREAMS:
            raise ValueError(f"unknown stream {stream!r}; known: {list(STREAMS)}")
        self.stream = stream
        self.device = device
        self.splits = STREAMS[stream]

        train = CIFAR100(root=root, train=True, download=download)
        test = CIFAR100(root=root, train=False, download=download)

        order = np.random.default_rng(class_order_seed).permutation(100)
        remap = np.empty(100, dtype=np.int64)
        remap[order] = np.arange(100)

        self.x_train = self._to_device(train.data)
        self.y_train = torch.as_tensor(
            remap[np.asarray(train.targets)], device=device, dtype=torch.long
        )
        self.x_test = self._to_device(test.data)
        self.y_test = torch.as_tensor(
            remap[np.asarray(test.targets)], device=device, dtype=torch.long
        )

        self.tasks: list[Task] = []
        start = 0
        for i, width in enumerate(self.splits):
            classes = tuple(range(start, start + width))
            lo, hi = start, start + width
            train_idx = torch.nonzero((self.y_train >= lo) & (self.y_train < hi), as_tuple=True)[0]
            test_idx = torch.nonzero((self.y_test >= lo) & (self.y_test < hi), as_tuple=True)[0]
            task = Task(index=i, classes=classes, train_idx=train_idx, test_idx=test_idx)
            self.tasks.append(task)
            start = hi

        # Task-incremental training: with task_masked, the loss of an example
        # sees only the logits of its own task (methods.py's `stream_loss`),
        # so class_mask[y] is the set of logits the label y competes with.
        self.class_mask = None
        if task_masked:
            owner = torch.empty(self.n_classes, dtype=torch.long, device=device)
            for task in self.tasks:
                owner[list(task.classes)] = task.index
            self.class_mask = owner[:, None] == owner[None, :]

        self._mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def _to_device(self, data: np.ndarray) -> torch.Tensor:
        chw = torch.from_numpy(data).permute(0, 3, 1, 2)
        return chw.contiguous().to(self.device)

    @property
    def n_classes(self) -> int:
        return sum(self.splits)

    def task_range(self, task: int) -> tuple[int, int]:
        lo = sum(self.splits[:task])
        return lo, lo + self.splits[task]

    def normalise(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return (x_uint8.float().div_(255.0) - self._mean) / self._std

    def augment(self, x_uint8: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        n = x_uint8.shape[0]
        x = torch.nn.functional.pad(x_uint8, (4, 4, 4, 4))
        offsets = torch.randint(0, 9, (n, 2), device=x.device, generator=generator)
        rows = offsets[:, 0, None] + torch.arange(32, device=x.device)
        cols = offsets[:, 1, None] + torch.arange(32, device=x.device)
        x = x.gather(2, rows[:, None, :, None].expand(n, 3, 32, 40))
        x = x.gather(3, cols[:, None, None, :].expand(n, 3, 32, 32))
        flip = torch.rand(n, device=x.device, generator=generator) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return self.normalise(x)


__all__ = ["MEAN", "STD", "STREAMS", "SplitCIFAR100", "Task"]
