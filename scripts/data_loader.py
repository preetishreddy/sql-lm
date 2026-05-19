import numpy as np
import json
from pathlib import Path
from scripts.config import BATCH_SIZE, CONTEXT_LENGTH

class CorpusLoader:
    def __init__(self, data_dir: str, split: str = 'train', seed: int = 42):
        """
        Data loader for the pre-tokenized SQL-LM corpus.
        
        Args:
            data_dir: Path to the directory containing .npy files and manifest.json.
            split: 'train' or 'val'.
            seed: RNG seed for reproducible sampling.
        """
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / 'manifest.json'
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
            
        self.split = split
        self.rng = np.random.default_rng(seed)

        # mmap_mode='r': reads from disk on demand, avoids massive RAM spikes
        self.arrays = {}
        self.sizes = {}
        
        for name, info in self.manifest['sources'].items():
            path = self.data_dir / f'{name}_{split}.npy'
            if path.exists():
                # We use mmap to keep memory usage low even with 5GB+ of data
                self.arrays[name] = np.load(str(path), mmap_mode='r')
                self.sizes[name] = info[split]['sequences']
            else:
                print(f"Warning: {path} not found, skipping source {name}")

        if not self.arrays:
            raise RuntimeError(f"No {split} data files found in {data_dir}")

        # Sampling probabilities from manifest proportions
        # We use the target_proportion to ensure the model sees the intended mix
        self.names = list(self.arrays.keys())
        props = np.array([self.manifest['sources'][n]['target_proportion'] for n in self.names])
        self.probs = props / props.sum()

        total_seqs = sum(self.sizes.values())
        print(f"CorpusLoader: {len(self.arrays)} sources, {total_seqs:,} sequences, split={split}")

    def next_batch(self, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Sample a random batch of sequences from the corpus.
        
        Returns:
            np.ndarray of shape [batch_size, CONTEXT_LENGTH], dtype int32.
        """
        # 1. Select a source based on corpus proportions
        source = self.rng.choice(self.names, p=self.probs)
        arr = self.arrays[source]
        
        # 2. Sample random indices within that source
        indices = self.rng.integers(0, len(arr), size=batch_size)
        
        # 3. Load and cast to int32 (JAX embedding lookup requires int32/int64)
        return arr[indices].astype(np.int32)

    def val_batch_iter(self, batch_size: int = BATCH_SIZE, n_batches: int = 50):
        """
        Representative validation iterator.
        
        Samples sources by their corpus proportions to provide a balanced val loss.
        """
        for _ in range(n_batches):
            source = self.rng.choice(self.names, p=self.probs)
            arr = self.arrays[source]
            # Ensure we have enough data for a batch
            if len(arr) <= batch_size:
                yield arr[:].astype(np.int32)
                continue
            i = int(self.rng.integers(0, len(arr) - batch_size))
            yield arr[i:i+batch_size].astype(np.int32)

    def epoch_iterator(self, batch_size: int = BATCH_SIZE):
        """
        Sequential scan over every sequence in the split.
        Useful for final evaluation or if exact val loss is needed.
        """
        for name in self.names:
            arr = self.arrays[name]
            for i in range(0, len(arr) - batch_size + 1, batch_size):
                yield arr[i:i+batch_size].astype(np.int32)
