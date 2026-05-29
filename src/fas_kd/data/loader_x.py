from __future__ import annotations

import queue
import threading
from typing import Any

import torch
from torch.utils.data import DataLoader


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device=device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(v, device=device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_to_device(v, device=device) for v in batch)
    return batch


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank: int, max_prefetch: int = 6) -> None:
        super().__init__(daemon=True)
        self.generator = generator
        self.local_rank = local_rank
        self.queue: queue.Queue[Any] = queue.Queue(max_prefetch)
        self.start()

    def run(self) -> None:
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item


class DataLoaderX(DataLoader):
    def __init__(self, local_rank: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        self.stream = torch.cuda.Stream(device=local_rank)
        self._iter = None
        self._batch = None

    def __iter__(self):
        self._iter = super().__iter__()
        self._iter = BackgroundGenerator(self._iter, self.local_rank)
        self._preload()
        return self

    def _preload(self) -> None:
        if self._iter is None:
            self._batch = None
            return
        self._batch = next(self._iter, None)
        if self._batch is None:
            return

        with torch.cuda.stream(self.stream):
            self._batch = _move_to_device(self._batch, device=self.device)

    def __next__(self):
        torch.cuda.current_stream(device=self.local_rank).wait_stream(self.stream)
        batch = self._batch
        if batch is None:
            raise StopIteration
        self._preload()
        return batch
