import os
import re
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Union, Iterator, Any


class LAMMPSTrajectory:
    """
    A class for handling LAMMPS trajectory dump files.
    
    This class provides methods for reading and parsing LAMMPS dump files,
    accessing trajectory data as NumPy arrays, and performing on-the-fly analysis.
    
    Attributes:
        filename (str): Path to the LAMMPS dump file
        n_frames (int): Total number of frames in the trajectory
        n_atoms (int): Number of atoms in each frame
        timesteps (np.ndarray): Array of timesteps for each frame
        box_dims (np.ndarray): Array of box dimensions for each frame (shape: [n_frames, 3, 2])
        atom_columns (List[str]): List of column names in the dump file
        _cache (Dict): Cache for frequently accessed frames
    """

    def __init__(self, filename: str, cache_size: int = 5):
        """
        Initialize the LAMMPSTrajectory object.
        
        Args:
            filename (str): Path to the LAMMPS dump file
            cache_size (int, optional): Number of frames to cache. Defaults to 5.
        
        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file is not a valid LAMMPS dump file
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        self.filename = filename
        self._cache_size = cache_size
        self._cache = {}
        
        # Scan the file to determine basic properties
        self._scan_file()

    def _scan_file(self) -> None:
        """
        Scan the dump file to determine number of frames, atoms, and structure.
        
        This method reads through the file once to determine:
        - Total number of frames
        - Number of atoms per frame
        - Column names
        - Timesteps for each frame
        - Box dimensions for each frame        
        """

        frame_starts = []
        timesteps = []
        box_dims = []

        with open(self.filename, 'r') as f:
            line = f.readline()
            while line:
                # Check for ITEM: TIMESTEP
                if line.startswith("ITEM: TIMESTEP"):
                    frame_starts.append(f.tell() - len(line))
                    
                    # Read timestep
                    timestep = int(f.readline().strip())
                    timesteps.append(timestep)
                    
                    # Read number of atoms
                    f.readline()  # ITEM: NUMBER OF ATOMS
                    n_atoms_line = f.readline().strip()
                    if len(frame_starts) == 1:
                        self.n_atoms = int(n_atoms_line)
                    
                    # Read box bounds
                    box_header = f.readline() # ITEM: BOX BOUNDS [optional tilt factors] [boundary types]

                    x_bounds_line = f.readline().strip().split()
                    y_bounds_line = f.readline().strip().split()
                    z_bounds_line = f.readline().strip().split()

                    x_bounds = [float(x_bounds_line[0]), float(x_bounds_line[1])]
                    y_bounds = [float(y_bounds_line[0]), float(y_bounds_line[1])]
                    z_bounds = [float(z_bounds_line[0]), float(z_bounds_line[1])]
                    
                    box_dims.append([x_bounds, y_bounds, z_bounds])
                    
                    # Get column names from the first frame
                    atoms_header = f.readline().strip()  # ITEM: ATOMS ...
                    if len(frame_starts) == 1:
                        # Extract column names from header
                        match = re.search(r"ITEM: ATOMS\s+(.*)", atoms_header)
                        if match:
                            self.atom_columns = match.group(1).split()
                        else:
                            raise ValueError("Invalid LAMMPS dump file: missing ATOMS header")
                
                line = f.readline()
        
        self.n_frames = len(frame_starts)
        self._frame_starts = np.array(frame_starts)
        self.timesteps = np.array(timesteps)
        self.box_dims = np.array(box_dims)
        
        if self.n_frames == 0:
            raise ValueError("No valid frames found in the LAMMPS dump file")
        
    def get_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """
        Get a specific frame from the trajectory.
        
        Args:
            frame_idx (int): Index of the frame to retrieve
        
        Returns:
            Dict[str, np.ndarray]: Dictionary with atom data arrays keyed by column names
        
        Raises:
            IndexError: If frame_idx is out of range
        """

        if frame_idx < 0 or frame_idx >= self.n_frames:
            raise IndexError(f"Frame index {frame_idx} out of range (0-{self.n_frames-1})")
        
        # Check if frame is in cache, if frame is already in cache - then returns frame from here rather than reading from the file
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        
        # Read the frame from file
        with open(self.filename, 'r') as f:
            f.seek(self._frame_starts[frame_idx])
            
            # Skip header lines
            for _ in range(9):  # Timestep to ATOMS header
                f.readline()
            
            # Initialize arrays for each column
            frame_data = {col: np.zeros(self.n_atoms) for col in self.atom_columns}
            
            # Read atom data
            for i in range(self.n_atoms):
                line = f.readline().strip().split()
                for j, col in enumerate(self.atom_columns):
                    frame_data[col][i] = float(line[j])
        
        # Cache the frame
        self._update_cache(frame_idx, frame_data)
        
        return frame_data

    def _update_cache(self, frame_idx: int, frame_data: Dict[str, np.ndarray]) -> None:
        """
        Cache system is designed to store recently accessed frames in memory - this will help will data analysis that requires multiple frames
        Update the frame cache with a new frame.
        
        Args:
            frame_idx (int): Index of the frame
            frame_data (Dict[str, np.ndarray]): Frame data
        """

        # Add to cache
        self._cache[frame_idx] = frame_data
        
        # Remove oldest frame if cache is full
        if len(self._cache) > self._cache_size:
            # Get the oldest key that's not the current frame_idx
            keys_to_consider = [k for k in self._cache.keys() if k != frame_idx]
            if keys_to_consider:
                oldest_key = min(keys_to_consider)
                del self._cache[oldest_key]
    
    def iterate_frames(self, start: int = 0, end: Optional[int] = None, step: int = 1) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
        """
        Iterate through frames in the trajectory, loading only one frame at a time into memory.
        
        This method uses a generator pattern with the 'yield' keyword to provide memory-efficient
        iteration. Each iteration step loads and returns just a single frame,
        rather than loading all frames at once.
        
        Args:
            start (int, optional): Starting frame index. Defaults to 0.
            end (int, optional): Ending frame index (exclusive). Defaults to n_frames.
            step (int, optional): Step size for skipping frames. Defaults to 1.
        
        Yields:
            Tuple[int, Dict[str, np.ndarray]]: Tuple of (frame_index, frame_data)
                The frame_data is a dictionary mapping column names to numpy arrays of values.
        
        Examples:
            >>> # Process every 10th frame
            >>> for idx, frame in trajectory.iterate_frames(step=10):
            >>>     analyze_frame(frame)
            >>>
            >>> # Process a specific range of frames
            >>> for idx, frame in trajectory.iterate_frames(start=100, end=200):
            >>>     print(f"Processing frame {idx}")
        
        Notes:
            - The generator approach allows for efficient memory usage since only one frame 
            is loaded at a time, making it possible to process trajectories larger than 
            available RAM.
            - Each yielded frame is loaded fresh from disk using the indexed frame positions.
        """

        if end is None:
            end = self.n_frames
        for i in range(start, end, step):
            yield i, self.get_frame(i)

    def analyze_trajectory(self, analysis_fn: Callable, start: int = 0, end: Optional[int] = None, step: int = 1, **kwargs) -> Any:
        """
        Apply an analysis function to each frame in the trajectory.
        
        Args:
            analysis_fn (Callable): Function that takes frame data and additional kwargs
            start (int, optional): Starting frame index. Defaults to 0.
            end (int, optional): Ending frame index (exclusive). Defaults to n_frames.
            step (int, optional): Step size. Defaults to 1.
            **kwargs: Additional keyword arguments to pass to the analysis function
        
        Returns:
            Any: Result of the analysis function
        """

        results = []
        
        for idx, frame in self.iterate_frames(start, end, step):
            result = analysis_fn(frame, box_dims=self.box_dims[idx], 
                                timestep=self.timesteps[idx], **kwargs)
            results.append(result)
        
        return results
    
    def select_atoms(self, frame_idx: int, selection_criteria: Dict[str, Any]) -> np.ndarray:
        """
        Select atoms from a frame based on criteria.
        
        Args:
            frame_idx (int): Index of the frame
            selection_criteria (Dict[str, Any]): Dictionary of {column: value/condition}
                For example: {'type': 1} or {'x': lambda x: x > 10.0}
        
        Returns:
            np.ndarray: Boolean mask of selected atoms
        """

        frame = self.get_frame(frame_idx)
        mask = np.ones(self.n_atoms, dtype=bool)
        
        for col, criterion in selection_criteria.items():
            if col not in frame:
                raise ValueError(f"Column '{col}' not found in frame data")
            
            if callable(criterion):
                mask &= criterion(frame[col])
            else:
                mask &= (frame[col] == criterion)
        
        return mask
    
    def get_atom_ids(self, frame_idx: int) -> np.ndarray:
        """
        Get atom IDs for a specific frame.
        
        Args:
            frame_idx (int): Index of the frame
        
        Returns:
            np.ndarray: Array of atom IDs
        
        Raises:
            ValueError: If 'id' column is not present in the dump file
        """

        if 'id' not in self.atom_columns:
            raise ValueError("Atom ID column not found in dump file")
        
        frame = self.get_frame(frame_idx)
        return frame['id'].astype(int)
    
    def get_atom_types(self, frame_idx: int) -> np.ndarray:
        """
        Get atom types for a specific frame.
        
        Args:
            frame_idx (int): Index of the frame
        
        Returns:
            np.ndarray: Array of atom types
        
        Raises:
            ValueError: If 'type' column is not present in the dump file
        """

        if 'type' not in self.atom_columns:
            raise ValueError("Atom type column not found in dump file")
        
        frame = self.get_frame(frame_idx)
        return frame['type'].astype(int)
    
    def get_atom_property(self, frame_idx: int, property_name: str) -> np.ndarray:
        """
        Get a specific atom property for a frame.
        
        Args:
            frame_idx (int): Index of the frame
            property_name (str): Name of the property (column)
        
        Returns:
            np.ndarray: Array of property values
        
        Raises:
            ValueError: If property is not present in the dump file
        """

        if property_name not in self.atom_columns:
            raise ValueError(f"Property '{property_name}' not found in dump file")
        
        frame = self.get_frame(frame_idx)
        return frame[property_name]
    
    def __len__(self) -> int:
        """Return the number of frames in the trajectory."""
        return self.n_frames
    
    def __getitem__(self, idx: Union[int, slice]) -> Dict[str, np.ndarray]:
        """
        Get a frame or slice of frames from the trajectory.
        
        Args:
            idx (Union[int, slice]): Frame index or slice
        
        Returns:
            Dict[str, np.ndarray] or List[Dict[str, np.ndarray]]: Frame data
        """
        if isinstance(idx, int):
            return self.get_frame(idx)
        elif isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or self.n_frames
            step = idx.step or 1
            
            return [self.get_frame(i) for i in range(start, stop, step)]
        else:
            raise TypeError("Index must be int or slice")