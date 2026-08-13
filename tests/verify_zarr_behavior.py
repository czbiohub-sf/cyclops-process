"""Verify that zarr.create_dataset doesn't write chunks immediately."""
import tempfile
import zarr
from pathlib import Path
import shutil

def count_chunk_files(zarr_path):
    """Count actual chunk files (not metadata files)."""
    chunk_files = [f for f in Path(zarr_path).rglob("*") 
                   if f.is_file() and not f.name.startswith('.')]
    return len(chunk_files)

def test_zarr_lazy_behavior():
    print("\n" + "="*70)
    print("TESTING: Does zarr.create_dataset() pre-write chunks?")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a zarr array
        store_path = tmpdir / "test.zarr"
        shape = (10, 10, 100, 100)  # Small array for testing
        chunks = (1, 1, 50, 50)
        
        print(f"\nCreating array with shape {shape}, chunks {chunks}")
        print(f"Total possible chunks: {10*10*2*2} = 400 chunks")
        
        # Method 1: zarr.create_dataset
        print("\n1. Using zarr.create_dataset()...")
        arr = zarr.open_array(
            str(store_path / "lazy"),
            mode='w',
            shape=shape,
            chunks=chunks,
            dtype='float32',
            fill_value=0,
        )
        
        lazy_chunks = count_chunk_files(store_path / "lazy")
        lazy_size_kb = sum(f.stat().st_size for f in (store_path / "lazy").rglob("*")) / 1024
        print(f"   Chunk files created: {lazy_chunks}")
        print(f"   Disk usage: {lazy_size_kb:.1f} KB")
        
        # Method 2: zarr.zeros (creates and fills)
        print("\n2. Using zarr.zeros() (explicit fill)...")
        arr2 = zarr.zeros(
            shape=shape,
            chunks=chunks,
            dtype='float32',
            store=str(store_path / "eager"),
        )
        
        eager_chunks = count_chunk_files(store_path / "eager")
        eager_size_kb = sum(f.stat().st_size for f in (store_path / "eager").rglob("*")) / 1024
        print(f"   Chunk files created: {eager_chunks}")
        print(f"   Disk usage: {eager_size_kb:.1f} KB")
        
        # Method 3: Write to lazy array
        print("\n3. After writing to ONE chunk of lazy array...")
        arr[0, 0, :, :] = 1.0  # Write to first chunk
        
        lazy_chunks_after = count_chunk_files(store_path / "lazy")
        lazy_size_kb_after = sum(f.stat().st_size for f in (store_path / "lazy").rglob("*")) / 1024
        print(f"   Chunk files created: {lazy_chunks_after}")
        print(f"   Disk usage: {lazy_size_kb_after:.1f} KB")
        
        print("\n" + "="*70)
        print("CONCLUSION:")
        print("="*70)
        if lazy_chunks == 0:
            print("✓ zarr.create_dataset() does NOT pre-write chunks (lazy!)")
            print(f"✓ Only metadata is written ({lazy_size_kb:.1f} KB)")
        else:
            print("✗ zarr.create_dataset() DOES pre-write chunks (not lazy!)")
            
        if eager_chunks > 0:
            print(f"✓ zarr.zeros() DOES pre-write all chunks ({eager_chunks} chunks, {eager_size_kb:.1f} KB)")
        
        if lazy_chunks_after > lazy_chunks:
            print(f"✓ Chunks created on-demand when data is written")
            
        print("="*70)
        
        # Cleanup
        shutil.rmtree(store_path)

if __name__ == "__main__":
    test_zarr_behavior()

