import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from mdtoolkit.core.utils import minimum_image_distance, extract_box_lengths

def calculate_rdf_bin_properties(max_distance: float, n_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate properties for RDF bins.
    
    Args:
        max_distance (float): Maximum distance for RDF calculation
        n_bins (int): Number of distance bins
        
    Returns:
        Tuple containing:
            bin_edges (np.ndarray): Edges of bins, shape (n_bins+1,)
            bin_widths (np.ndarray): Width of each bin, shape (n_bins,)
            bin_centers (np.ndarray): Centers of bins, shape (n_bins,)
            bin_volumes (np.ndarray): Volumes of spherical shells, shape (n_bins,)
    """

    bin_edges = np.linspace(0, max_distance, n_bins + 1)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_volumes = 4 * np.pi * bin_centers**2 * bin_widths
    
    return bin_edges, bin_widths, bin_centers, bin_volumes

def create_rdf_analyzer(max_distance: float = 10.0, n_bins: int = 100, type_a: int = 1, type_b: Optional[int] = None, normalize: bool = True) -> Callable:
    """
    Create an RDF analyzer function for use with LAMMPSTrajectory.analyze_trajectory.
    
    This function creates an analysis function that calculates the radial
    distribution function (RDF) between atoms of type_a and type_b. The function
    maintains internal state to accumulate histogram data across all frames.
    
    Args:
        max_distance (float): Maximum distance to consider for RDF (in same units as trajectory)
        n_bins (int): Number of distance bins for the histogram
        type_a (int): First atom type for RDF calculation
        type_b (int, optional): Second atom type for RDF calculation. If None, use type_a.
        normalize (bool): Whether to normalize the final RDF
    
    Returns:
        Callable: Analysis function compatible with LAMMPSTrajectory.analyze_trajectory
    
    Example:
        >>> from mdtoolkit.core.trajectory import LAMMPSTrajectory
        >>> from mdtoolkit.analysis.rdf import create_rdf_analyzer, process_rdf_results
        >>> 
        >>> # Load trajectory
        >>> traj = LAMMPSTrajectory("water.dump")
        >>> 
        >>> # Create RDF analyzer for O-O pairs (assuming type 1 is oxygen)
        >>> rdf_analyzer = create_rdf_analyzer(max_distance=10.0, n_bins=100, type_a=1, type_b=1)
        >>> 
        >>> # Analyze trajectory
        >>> results = traj.analyze_trajectory(rdf_analyzer)
        >>> 
        >>> # Process and get final RDF
        >>> distances, rdf = process_rdf_results(results)
        >>> 
        >>> # Plot RDF
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(distances, rdf)
        >>> plt.xlabel('r (Å)')
        >>> plt.ylabel('g(r)')
        >>> plt.title('O-O Radial Distribution Function')
        >>> plt.show()
    """

    # Initialize the second type if it is not provided, this will in turn be a self-self RDF
    if type_b is None:
        type_b = type_a

    bin_edges, bin_widths, bin_centers, bin_volumes = calculate_rdf_bin_properties(max_distance, n_bins)

    # State variables - these will be maintained throughout all frames
    histogram = np.zeros(n_bins)
    n_frames = 0

    total_num_type_a = 0
    total_num_type_b = 0
    total_volume = 0.0

    def analyze_frame(frame: Dict[str, np.ndarray], box_dims: np.ndarray, timestep: float, **kwargs) -> Dict[str, Any]:
        """
        Analyze a single frame to update the RDF calculation.
        
        Args:
            frame: Dictionary with atom data
            box_dims: Box dimensions for this frame
            timestep: Current timestep
            **kwargs: Additional arguments
            
        Returns:
            Dict with frame-specific results
        """

        nonlocal histogram, n_frames, total_num_type_a, total_num_type_b, total_volume

        mask_type_a = frame['type'] == type_a
        mask_type_b = frame['type'] == type_b

        num_type_a = np.sum(mask_type_a)
        num_type_b = np.sum(mask_type_b)

        if num_type_a == 0:
            raise ValueError(f"Frame at timestep {timestep} contains no atoms of type {type_a}")
        if num_type_b == 0:
            raise ValueError(f"Frame at timestep {timestep} contains no atoms of type {type_b}")
        
        pos_a = np.column_stack([frame['x'][mask_type_a], frame['y'][mask_type_a], frame['z'][mask_type_a]])
        pos_b = np.column_stack([frame['x'][mask_type_b], frame['y'][mask_type_b], frame['z'][mask_type_b]])

        box_lengths = extract_box_lengths(box_dims)
        box_volume = np.prod(box_lengths)

        frame_hist = np.zeros(n_bins)

        self_rdf = type_a == type_b

        for i, pos_i in enumerate(pos_a):
            # Avoid double counting if two types are the same
            if self_rdf:
                j_start = i + 1
            else:
                j_start = 0

            for j in range(j_start, len(pos_b)):
                if self_rdf and i == j:
                    continue

                distance = minimum_image_distance(pos_i, pos_b[j], box_lengths)

                if distance < max_distance:
                    bin_idx = int(distance / max_distance * n_bins)
                    if bin_idx < n_bins:
                        frame_hist[bin_idx] += 1

        # Update globals outside of routine
        histogram += frame_hist

        n_frames += 1
        total_num_type_a += num_type_a
        total_num_type_b += num_type_b
        total_volume += box_volume

        # Return frame-level results
        return {
            'timestep': timestep,
            'frame_histogram': frame_hist,
            'num_type_a': num_type_a,
            'num_type_b': num_type_b,
            'box_volume': box_volume,  # Fixed key name: was 'box volume' with a space
            'bin_centers': bin_centers,
        }
    
    # Method to get the final RDF from the internal state
    def get_rdf():
        if n_frames == 0:
            return bin_centers, np.zeros_like(bin_centers)
        
        avg_n_a = total_num_type_a / n_frames
        avg_n_b = total_num_type_b / n_frames
        avg_volume = total_volume / n_frames
        
        # Calculate number density for later use in coordination number calculations
        # Use number density of type_b particles
        number_density = avg_n_b / avg_volume
        
        hist_copy = histogram.copy()
        
        if normalize:
            if type_a == type_b:
                normalization = avg_n_a * (avg_n_a - 1) / 2  
                density = avg_n_a / avg_volume
            else:
                normalization = avg_n_a * avg_n_b  
                density = avg_n_b / avg_volume  
            
            if normalization > 0:
                # g(r) = hist(r) * V / (N_A * N_B * dV(r) * n_frames)
                density = normalization / avg_volume
                for i in range(n_bins):
                        if bin_volumes[i] > 0:
                            hist_copy[i] = hist_copy[i] * avg_volume / (n_frames * normalization * bin_volumes[i])

        # Store calculation parameters in the function for metadata access
        get_rdf.metadata = {
            'type_a': type_a,
            'type_b': type_b,
            'max_distance': max_distance,
            'n_bins': n_bins,
            'normalize': normalize,
            'n_frames': n_frames,
            'avg_n_a': avg_n_a,
            'avg_n_b': avg_n_b,
            'avg_volume': avg_volume,
            'number_density': number_density  # Number density of type_b particles
        }
        
        return bin_centers, hist_copy
    
    # Add method to create RDFResult directly - MOVED HERE FROM THE BOTTOM
    def get_rdf_result(metadata=None):
        """Create an RDFResult object from this analyzer's data"""
        from mdtoolkit.output.output_rdf import RDFResult
        
        distances, rdf_values = get_rdf()
        
        # Combine metadata from the analyzer with any provided metadata
        combined_metadata = get_rdf.metadata.copy() if hasattr(get_rdf, 'metadata') else {}
        if metadata:
            combined_metadata.update(metadata)
            
        return RDFResult(distances, rdf_values, combined_metadata)
    
    # Attach both methods to the analyzer function
    analyze_frame.get_rdf = get_rdf
    analyze_frame.get_rdf_result = get_rdf_result
    
    return analyze_frame

def process_rdf_results(results: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process RDF analysis results from LAMMPSTrajectory.analyze_trajectory.
    
    This function extracts and processes the raw histogram data from the
    trajectory analysis to produce the final RDF.
    
    Args:
        results: List of frame results from analyze_trajectory
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (distances, rdf_values)
    """

    if not results:
        raise ValueError("No results to process")
    
    bin_centers = results[0].get('bin_centers')
    if bin_centers is None:
        raise ValueError("Results don't contain bin_centers")
    
    if hasattr(results, 'get_rdf'):
        return results.get_rdf()
    
    # Otherwise, do manual processing
    n_bins = len(bin_centers)
    total_hist = np.zeros(n_bins)
    n_frames = len(results)
    
    for result in results:
        if not result.get('skipped', False):
            total_hist += result['frame_histogram']
    
    total_n_a = sum(result.get('num_type_a', 0) for result in results)
    total_n_b = sum(result.get('num_type_b', 0) for result in results)
    total_volume = sum(result.get('box_volume', 0) for result in results)  # Fixed key name
    
    avg_n_a = total_n_a / n_frames if n_frames > 0 else 0
    avg_n_b = total_n_b / n_frames if n_frames > 0 else 0
    avg_volume = total_volume / n_frames if n_frames > 0 else 0
    
    same_type = results[0].get('num_type_a', 0) == results[0].get('num_type_b', 0)
    
    if same_type:
        normalization = avg_n_a * (avg_n_a - 1) / 2
    else:
        normalization = avg_n_a * avg_n_b
    
    max_distance = bin_centers[-1] * 2  
    bin_properties = calculate_rdf_bin_properties(max_distance, n_bins)
    bin_volumes = bin_properties[3]  

    rdf = np.zeros_like(total_hist)
    if normalization > 0:
        density = normalization / avg_volume
        for i in range(n_bins):
            if bin_volumes[i] > 0:
                rdf[i] = total_hist[i] / (n_frames * density * bin_volumes[i])
    
    return bin_centers, rdf