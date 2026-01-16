from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property
import h5py
def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6, 
        max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


from typing import Union, Dict, Optional, List
import os
import numbers
import numpy as np
from functools import cached_property
import h5py
import random
class HDF5Buffer:
    """
    HDF5-based temporal datastructure for reading LIBERO and similar datasets.
    Assumes first dimension to be time. Organized by episodes.
    """
    def __init__(self, 
                 file_path: str,
                #  demo_key: str,
                 mode: str = 'r'):
        """
        Initialize HDF5 buffer from file path.
        
        Args:
            file_path: Path to the HDF5 file
            mode: File mode ('r' for read, 'r+' for read/write)
        """
        self.file_path = os.path.expanduser(file_path)
        
        self.mode = mode
        self.file = h5py.File(self.file_path, mode)
        demo_keys=self.file['data'].keys()
        self.demo_keys=list(demo_keys)
        demo_key=random.choice(list(demo_keys))
        self.demo_key=demo_key
        # Check if the file has the expected structure
        if 'data' not in self.file:
            raise ValueError("HDF5 file must contain 'data' group")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, 'file') and self.file:
            self.file.close()
    
    # ============= properties =================
    @cached_property
    def data(self):
        """Return the data group from HDF5 file."""
        return self.file['data'][self.demo_key]
    
    @cached_property
    def meta(self):
        """Return the meta group if exists, otherwise None."""
        return self.file.get('meta', None)
    
    @property
    def episode_ends(self):
        """Get episode ends from meta data."""
        if self.meta and 'episode_ends' in self.meta:
            return self.meta['episode_ends'][:]
        return None
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        return f"HDF5Buffer(file_path='{self.file_path}', mode='{self.mode}')"
    
    def keys(self):
        """Return keys of the data group."""
        keys=list(self.data.keys())+list([ f'obs/{key}'for key in self.data['obs'].keys()])
        temp=['obs']
        key_list=[key for key in keys if key not in temp ]
        return key_list
    def values(self):
        """Return datasets in the data group."""
        return [self.data[key] for key in self.data.keys()]
    
    def items(self):
        """Return (key, dataset) pairs from data group."""
        return [(key, self.data[key]) for key in self.data.keys()]
    
    def __getitem__(self, key):
        """Get dataset by key."""
        return self.data[key]
    
    def __contains__(self, key):
        """Check if key exists in data group."""
        return key in self.data
    
    # =========== dataset properties ==============
    @property
    def n_episodes(self) -> int:
        """Get number of episodes."""
        if self.episode_ends is not None:
            return len(self.episode_ends)
        # If no episode_ends metadata, count demo groups
        return len(self.demo_keys)
    
    @property
    def episode_keys(self) -> List[str]:
        """Get list of episode keys."""
        return len(self.demo_keys)
    
    def get_episode_length(self, episode_key: str) -> int:
        """Get length of a specific episode."""
        if episode_key in self.data:
            episode_group = self.data[episode_key]
            # Get length from the first dataset in the episode
            for dataset_name in episode_group:
                if hasattr(episode_group[dataset_name], 'shape'):
                    return episode_group[dataset_name].shape[0]
        return 0
    
    # =========== data access methods ==============
    def get_episode(self, episode_idx: Union[int, str], copy: bool = False) -> Dict[str, np.ndarray]:
        """
        Get all data for a specific episode.
        
        Args:
            episode_idx: Episode index or key
            copy: Whether to copy data to numpy arrays
            
        Returns:
            Dictionary of episode data
        """
        if isinstance(episode_idx, int):
            episode_keys = sorted(self.episode_keys)
            if episode_idx < 0 or episode_idx >= len(episode_keys):
                raise IndexError(f"Episode index {episode_idx} out of range")
            episode_key = episode_keys[episode_idx]
        else:
            episode_key = episode_idx
        
        if episode_key not in self.data:
            raise KeyError(f"Episode {episode_key} not found")
        
        episode_group = self.data[episode_key]
        result = {}
        
        for dataset_name in episode_group:
            dataset = episode_group[dataset_name]
            if copy:
                result[dataset_name] = dataset[()]
            else:
                result[dataset_name] = dataset
        
        return result
    
    def get_dataset(self, dataset_name: str, episode_idx: Optional[int] = None, 
                   copy: bool = False) -> Union[np.ndarray, h5py.Dataset]:
        """
        Get specific dataset from HDF5 file.
        
        Args:
            dataset_name: Name of the dataset to retrieve
            episode_idx: Optional episode index (if None, looks in root data)
            copy: Whether to copy data to numpy array
            
        Returns:
            Requested dataset or numpy array
        """
        if episode_idx is not None:
            episode_key = f"demo_{episode_idx}"
            if episode_key in self.data and dataset_name in self.data[episode_key]:
                dataset = self.data[episode_key][dataset_name]
                return dataset[()] if copy else dataset
        
        # Look in root data group
        if dataset_name in self.data:
            dataset = self.data[dataset_name]
            return dataset[()] if copy else dataset
        
        raise KeyError(f"Dataset {dataset_name} not found")
    
    def get_obs(self, episode_idx: int, timestep: int, 
                obs_type: str = 'rgb_obs', copy: bool = False) -> np.ndarray:
        """
        Get specific observation from an episode.
        
        Args:
            episode_idx: Episode index
            timestep: Timestep within episode
            obs_type: Type of observation ('rgb_obs', 'depth_obs', etc.)
            copy: Whether to copy data to numpy array
            
        Returns:
            Requested observation
        """
        episode_data = self.get_episode(episode_idx, copy=False)
        
        if obs_type not in episode_data:
            raise KeyError(f"Observation type {obs_type} not found in episode")
        
        obs_data = episode_data[obs_type]
        if timestep < 0 or timestep >= obs_data.shape[0]:
            raise IndexError(f"Timestep {timestep} out of range")
        
        return obs_data[timestep] if not copy else obs_data[timestep].copy()
    
    def get_actions(self, episode_idx: int, copy: bool = False) -> np.ndarray:
        """Get actions for an episode."""
        return self.get_dataset('actions', episode_idx, copy)
    
    def get_rewards(self, episode_idx: int, copy: bool = False) -> np.ndarray:
        """Get rewards for an episode."""
        return self.get_dataset('rewards', episode_idx, copy)
    
    def get_states(self, episode_idx: int, copy: bool = False) -> np.ndarray:
        """Get states for an episode."""
        return self.get_dataset('states', episode_idx, copy)
    
    # =========== metadata methods ==============
    def get_language_instruction(self, episode_idx: int) -> str:
        """Get language instruction for an episode if available."""
        try:
            # Check if language instruction is stored in attributes
            episode_key = f"demo_{episode_idx}"
            if episode_key in self.data and 'language_instruction' in self.data[episode_key].attrs:
                return self.data[episode_key].attrs['language_instruction']
            
            # Check in meta data
            if self.meta and 'language_instructions' in self.meta:
                instructions = self.meta['language_instructions']
                if episode_idx < len(instructions):
                    return instructions[episode_idx].decode('utf-8') if isinstance(instructions[episode_idx], bytes) else instructions[episode_idx]
        except:
            pass
        
        return "No language instruction available"
    
    def get_episode_info(self, episode_idx: int) -> Dict:
        """Get metadata information for an episode."""
        episode_key = f"demo_{episode_idx}"
        info = {
            'episode_key': episode_key,
            'length': self.get_episode_length(episode_key),
            'language_instruction': self.get_language_instruction(episode_idx)
        }
        
        # Add any other attributes
        if episode_key in self.data:
            for attr_name, attr_value in self.data[episode_key].attrs.items():
                info[attr_name] = attr_value
        
        return info
    
    # =========== utility methods ==============
    def list_episodes(self) -> List[Dict]:
        """List all episodes with basic information."""
        episodes = []
        for i in range(self.n_episodes):
            episodes.append(self.get_episode_info(i))
        return episodes
    
    def list_datasets(self) -> Dict[str, List[str]]:
        """List all available datasets organized by type."""
        datasets = {'episode_datasets': set(), 'global_datasets': set()}
        
        # Check episode datasets (from first episode)
        if self.n_episodes > 0:
            first_episode = self.get_episode(0, copy=False)
            datasets['episode_datasets'] = set(first_episode.keys())
        
        # Check global datasets in data group
        datasets['global_datasets'] = set(self.data.keys()) - set(self.episode_keys)
        
        return datasets
    
    def get_stats(self) -> Dict:
        """Get statistics about the dataset."""
        stats = {
            'file_path': self.file_path,
            'n_episodes': self.n_episodes,
            'episode_keys': self.episode_keys,
            'datasets': self.list_datasets(),
            'total_size': os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0
        }
        
        # Add episode length statistics
        if self.n_episodes > 0:
            lengths = [self.get_episode_length(key) for key in self.episode_keys]
            stats['episode_lengths'] = {
                'min': min(lengths),
                'max': max(lengths),
                'mean': sum(lengths) / len(lengths),
                'total': sum(lengths)
            }
        
        return stats

# ============= convenience functions ===============
def open_hdf5(file_path: str, mode: str = 'r') -> HDF5Buffer:
    """Convenience function to open HDF5 file."""
    return HDF5Buffer(file_path, mode)

def print_hdf5_info(file_path: str):
    """Print information about HDF5 file."""
    with HDF5Buffer(file_path) as buffer:
        stats = buffer.get_stats()
        print(f"HDF5 File: {stats['file_path']}")
        print(f"Number of episodes: {stats['n_episodes']}")
        print(f"Total size: {stats['total_size'] / (1024**2):.2f} MB")
        
        if 'episode_lengths' in stats:
            lengths = stats['episode_lengths']
            print(f"Episode lengths: min={lengths['min']}, max={lengths['max']}, mean={lengths['mean']:.1f}")
        
        print("\nAvailable datasets:")
        for category, datasets in stats['datasets'].items():
            if datasets:
                print(f"  {category}: {sorted(datasets)}")
        
        # Print first few episodes info
        print(f"\nFirst 3 episodes:")
        for i in range(min(3, stats['n_episodes'])):
            info = buffer.get_episode_info(i)
            print(f"  Episode {i}: {info}")

# Example usage
if __name__ == "__main__":
    # Example usage
    hdf5_file = "path/to/your/liberO_demo.hdf5"
    
    # Print file info
    print_hdf5_info(hdf5_file)
    
    # Open and use the buffer
    with HDF5Buffer(hdf5_file) as buffer:
        # Get stats
        stats = buffer.get_stats()
        print(f"Dataset contains {stats['n_episodes']} episodes")
        
        # Access specific episode
        if stats['n_episodes'] > 0:
            episode_data = buffer.get_episode(0, copy=True)
            print(f"Episode 0 has {len(episode_data)} datasets")
            
            # Get specific observation
            rgb_obs = buffer.get_obs(0, 0, 'rgb_obs', copy=True)
            print(f"First RGB observation shape: {rgb_obs.shape}")