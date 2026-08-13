"""Profile zarr precreation to identify bottlenecks."""
import tempfile
import time
from pathlib import Path
import shutil

from iohub.ngff import open_ome_zarr
from cyclops_utils.io.zarr_utils import ensure_position_array
import numpy as np


def profile_creation_steps(n_positions=100, shape=(1, 2, 1, 2048, 2048)):
    """Profile each step of zarr creation to find bottlenecks."""
    
    chunks = (1, 1, 1, 256, 256)
    dtype = np.float32
    scale = [1.0, 1.0, 0.325, 0.325, 0.325]
    
    print("\n" + "="*70)
    print(f"PROFILING ZARR CREATION BOTTLENECKS ({n_positions} positions)")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        store_path = tmpdir / "profile_test.zarr"
        
        # Step 1: Create HCS store
        t_start = time.time()
        with open_ome_zarr(store_path, layout="hcs", mode="w", channel_names=["test"]) as store:
            pass
        t_hcs = time.time() - t_start
        print(f"1. HCS store creation: {t_hcs:.3f}s")
        
        # Step 2: Create positions with lazy mode
        t_start = time.time()
        with open_ome_zarr(store_path, layout="hcs", mode="r+") as store:
            for i in range(n_positions):
                pos_name = f"A/1/{i:06d}"
                ensure_position_array(store, pos_name, shape, chunks, dtype, scale, lazy=True)
        t_lazy = time.time() - t_start
        print(f"2. Create {n_positions} positions (lazy): {t_lazy:.3f}s ({n_positions/t_lazy:.1f} pos/s)")
        
        # Check disk usage
        lazy_size_mb = sum(f.stat().st_size for f in store_path.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"   Disk usage: {lazy_size_mb:.2f} MB ({lazy_size_mb/n_positions:.3f} MB/position)")
        
        # Step 3: Test writing to one position
        t_start = time.time()
        with open_ome_zarr(store_path, mode="r+") as store:
            test_data = np.random.rand(*shape).astype(dtype)
            store["A/1/000000"]["0"][:] = test_data
        t_write = time.time() - t_start
        print(f"3. Write data to one position: {t_write:.3f}s")
        
        # Check if chunks were created
        chunk_count = len(list((store_path / "A" / "1" / "000000" / "0").rglob("*")))
        print(f"   Chunks created: {chunk_count}")
        
        # Compare with eager mode on fewer positions (it's slower)
        eager_store = tmpdir / "eager_test.zarr"
        n_eager = min(10, n_positions)  # Test fewer positions
        
        t_start = time.time()
        with open_ome_zarr(eager_store, layout="hcs", mode="w", channel_names=["test"]) as store:
            for i in range(n_eager):
                pos_name = f"A/1/{i:06d}"
                ensure_position_array(store, pos_name, shape, chunks, dtype, scale, lazy=False)
        t_eager = time.time() - t_start
        
        eager_size_mb = sum(f.stat().st_size for f in eager_store.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"\n4. Create {n_eager} positions (eager): {t_eager:.3f}s ({n_eager/t_eager:.1f} pos/s)")
        print(f"   Disk usage: {eager_size_mb:.2f} MB ({eager_size_mb/n_eager:.3f} MB/position)")
        
        # Extrapolate to full dataset
        print(f"\n" + "-"*70)
        print(f"EXTRAPOLATION TO FULL DATASET:")
        print(f"-"*70)
        
        # Typical OPS dataset: ~8 stores × ~1500 positions each = 12,000 total positions
        full_positions = 12000
        
        lazy_time_est = (t_lazy / n_positions) * full_positions
        eager_time_est = (t_eager / n_eager) * full_positions
        
        print(f"For {full_positions} positions (8 stores × 1500 pos):")
        print(f"  Lazy mode:  {lazy_time_est/60:.1f} minutes")
        print(f"  Eager mode: {eager_time_est/60:.1f} minutes")
        print(f"  Speedup:    {eager_time_est/lazy_time_est:.1f}x faster")
        
        lazy_disk_est = (lazy_size_mb / n_positions) * full_positions / 1024
        eager_disk_est = (eager_size_mb / n_eager) * full_positions / 1024
        
        print(f"\nDisk usage estimate:")
        print(f"  Lazy mode:  {lazy_disk_est:.1f} GB")
        print(f"  Eager mode: {eager_disk_est:.1f} GB")
        print(f"  Savings:    {(eager_disk_est - lazy_disk_est):.1f} GB ({eager_disk_est/lazy_disk_est:.1f}x smaller)")
        
        print("="*70)
        
        # Cleanup
        shutil.rmtree(store_path)
        shutil.rmtree(eager_store)


if __name__ == "__main__":
    # Test with different scales
    print("Testing with 100 positions (quick test)...")
    profile_creation_steps(n_positions=100)
    
    print("\n\nTesting with 1000 positions (realistic scale)...")
    profile_creation_steps(n_positions=1000)

