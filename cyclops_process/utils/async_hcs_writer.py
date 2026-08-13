"""
Async HCS Prediction Writer for viscy virtual staining.

This module provides an asynchronous writer that maintains proper FOV structure
while overlapping I/O with GPU compute. Based on viscy's HCSPredictionWriter
but optimized for high throughput with parallel writes.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    torch = None
from iohub.ngff import ImageArray, Position, TransformationMeta, open_ome_zarr
from numpy.typing import DTypeLike, NDArray

_logger = logging.getLogger(__name__)


def _pad_shape(shape: tuple[int, ...], target: int = 5) -> tuple[int, ...]:
    """Pad shape tuple to a target length."""
    pad = target - len(shape)
    return (1,) * pad + shape


def _resize_image(image: ImageArray, t_index: int, z_slice: slice) -> None:
    """Resize image array if incoming stack is not within bounds."""
    if image.shape[0] <= t_index or image.shape[2] < z_slice.stop:
        image.resize(
            max(t_index + 1, image.shape[0]),
            image.channels,
            max(z_slice.stop, image.shape[2]),
            *image.shape[-2:],
        )


class AsyncHCSPredictionWriter:
    """
    Asynchronous HCS Prediction Writer with parallel writes.

    Writes predictions to an HCS OME-Zarr store asynchronously using
    a thread pool for parallel I/O to different positions.

    Args:
        output_store: Path to the output zarr store
        channel_names: List of output channel names (e.g., ['nuclei', 'membrane'])
        z_padding: Z padding for 2.5D predictions (z_window_size // 2)
        num_workers: Number of parallel writer threads
        dataset_scale: Optional scale metadata for the dataset
    """

    def __init__(
        self,
        output_store: str,
        channel_names: list[str],
        z_padding: int = 0,
        num_workers: int = 4,
        dataset_scale: Optional[list] = None,
        input_store: Optional[str] = None,
    ):
        self.output_store = Path(output_store)
        self.channel_names = [ch + "_prediction" for ch in channel_names]
        self.z_padding = z_padding
        self.num_workers = num_workers
        self._dataset_scale = dataset_scale
        self._input_store = Path(input_store) if input_store else None

        self._executor = None
        self._futures = []
        self._plate = None
        self._prediction_index = None

        # Cache for created images (position path -> ImageArray)
        self._image_cache = {}
        self._image_cache_lock = threading.Lock()

        # Position creation lock (only needed when creating new positions)
        self._position_create_lock = threading.Lock()

        # Per-position resize locks to prevent TOCTOU race in _resize_image.
        self._resize_locks = defaultdict(threading.Lock)

    def start(self) -> None:
        """Start the writer and open the output store.

        If input_store was provided, pre-creates all positions with the
        correct (T, C, Z, Y, X) shape from the input, eliminating the
        need for dynamic resize during concurrent writes.
        """
        # Create/open the output store
        self._plate = open_ome_zarr(
            str(self.output_store),
            layout="hcs",
            mode="a",
            channel_names=self.channel_names,
        )
        self._prediction_index = [
            self._plate.get_channel_index(ch) for ch in self.channel_names
        ]
        _logger.info(f"Writing predictions to: '{self.output_store}'")

        # Pre-create all positions with correct shape from input store
        if self._input_store is not None:
            self._precreate_positions()

        # Start thread pool
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self._futures = []

    def _precreate_positions(self) -> None:
        """Pre-create all positions with the correct shape from the input store."""
        with open_ome_zarr(str(self._input_store), mode="r") as inp:
            positions = list(inp.positions())
            if not positions:
                return
            # Get shape from first position
            first_pos_name, _ = positions[0]
            sample_shape = inp[first_pos_name]["0"].shape  # (T, C, Z, Y, X)
            T_input = sample_shape[0]
            Z_input = sample_shape[2] + 2 * self.z_padding
            YX = sample_shape[-2:]

        n_channels = len(self.channel_names)
        full_shape = (T_input, n_channels, Z_input, *YX)
        chunks = _pad_shape(tuple(YX), 5)

        transform = None
        if self._dataset_scale is not None:
            transform = [TransformationMeta(type="scale", scale=self._dataset_scale)]

        n_created = 0
        for pos_name, _ in positions:
            if pos_name in self._plate.zgroup:
                continue
            parts = pos_name.split("/")
            row_name, col_name, pos_id = parts[0], parts[1], parts[2]
            position = self._plate.create_position(row_name, col_name, pos_id)
            image = position.create_zeros(
                "0",
                shape=full_shape,
                dtype=np.float32,
                chunks=chunks,
                transform=transform,
            )
            self._image_cache[f"/{pos_name}/0"] = image
            n_created += 1

        _logger.info(f"Pre-created {n_created} positions with shape {full_shape}")

    def stop(self) -> None:
        """Wait for pending writes and close the store."""
        if self._executor is not None:
            # Wait for all pending writes
            for future in self._futures:
                try:
                    future.result(timeout=60)
                except Exception as e:
                    _logger.error(f"Write error: {e}")
            self._executor.shutdown(wait=True)
            self._executor = None
            self._futures = []

        if self._plate is not None:
            self._plate.close()
            self._plate = None

        self._image_cache.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def _get_or_create_image(
        self, img_name: str, shape: tuple[int], dtype: DTypeLike
    ) -> ImageArray:
        """Get or create an image, with caching."""
        # Fast path: check cache without lock
        if img_name in self._image_cache:
            return self._image_cache[img_name]

        # Slow path: need to create or lookup
        with self._image_cache_lock:
            # Double-check after acquiring lock
            if img_name in self._image_cache:
                return self._image_cache[img_name]

            if img_name in self._plate.zgroup:
                image = self._plate[img_name]
            else:
                # Need to create position - use position lock to avoid races
                with self._position_create_lock:
                    # Check again in case another thread created it
                    if img_name in self._plate.zgroup:
                        image = self._plate[img_name]
                    else:
                        _, row_name, col_name, pos_name, arr_name = img_name.split("/")
                        position = self._plate.create_position(row_name, col_name, pos_name)
                        new_shape = [1] + list(shape)
                        new_shape[1] = len(position.channel_names)

                        transform = None
                        if self._dataset_scale is not None:
                            transform = [TransformationMeta(type="scale", scale=self._dataset_scale)]

                        image = position.create_zeros(
                            arr_name,
                            shape=new_shape,
                            dtype=dtype,
                            chunks=_pad_shape(tuple(new_shape[-2:]), 5),
                            transform=transform,
                        )

            self._image_cache[img_name] = image
            return image

    def _write_sample(
        self,
        img_name: str,
        t_index: int,
        z_index: int,
        prediction: np.ndarray,
    ) -> None:
        """Write a single sample to the store (called from thread pool)."""
        z_index += self.z_padding
        z_slice = slice(z_index, z_index + prediction.shape[-3])

        image = self._get_or_create_image(img_name, prediction.shape, prediction.dtype)

        # Only resize dynamically if positions weren't pre-created
        if self._input_store is None:
            with self._resize_locks[img_name]:
                _resize_image(image, t_index, z_slice)

        # Write directly - zarr handles concurrent writes to different chunks
        image.oindex[t_index, self._prediction_index, z_slice] = prediction

    def write_batch(self, batch: dict, predictions: torch.Tensor) -> None:
        """
        Submit a batch of predictions for parallel async writing.

        Args:
            batch: Batch dict from dataloader with 'index' key
            predictions: Prediction tensor (B, C, Z, Y, X)
        """
        # Transfer whole batch to CPU once (much faster than per-sample)
        predictions_cpu = predictions.cpu().numpy()

        # Submit each sample to the thread pool
        for sample_idx in range(len(batch["index"][0])):
            img_name = batch["index"][0][sample_idx]
            t_index = int(batch["index"][1][sample_idx])
            z_index = int(batch["index"][2][sample_idx])

            # Slice from CPU array (no copy, just view)
            sample_pred = predictions_cpu[sample_idx].copy()  # copy needed for thread safety

            # Submit to thread pool
            future = self._executor.submit(
                self._write_sample, img_name, t_index, z_index, sample_pred
            )
            self._futures.append(future)

        # Periodically clean up completed futures to avoid memory buildup
        if len(self._futures) > 1000:
            self._futures = [f for f in self._futures if not f.done()]

    def wait_pending(self) -> None:
        """Wait for all pending writes to complete."""
        for future in self._futures:
            future.result()
        self._futures = []
