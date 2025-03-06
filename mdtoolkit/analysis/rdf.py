import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

def create_rdf_analyzer(max_distance: float = 10.0, n_bins: int = 100, type_a: int = 1, type_b: Optional[int] = None, normalize: bool = True) -> Callable:
    """
    Create an RDF analyzer function for use with LAMMPSTrajectory.analyze_trajectory.
    
    This factory function creates an analysis function that calculates the radial
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

    bin_edges = np.linspace(0, max_distance, n_bins + 1)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_volumes = 4 * np.pi * bin_centers**2 * bin_widths

    # State variable - these will be maintained throughout all frames
    histogram = np.zeros(n_bins)
    n_frames = 0

    total_num_type_a = 0
    total_num_type_b = 0
    total_volume = 0.0

    # Fucntion which will be called each frame
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

        # Skip if no atoms of the required types
        ## Unsure why this is here - I think it is a check to see if the types exist or not, but that is more of an error check.  
        ### If they don't exist, the analysis should exit and an error message should be printed
        if num_type_a == 0 or num_type_b == 0:
            return {
                'timestep': timestep,
                'frame_histogram': np.zeros(n_bins),
                'num_type_a': 0,
                'num_type_b': 0,
                'skipped': True
            }
        
        pos_a = np.column_stack([frame['x'][mask_type_a], frame['y'][mask_type_a], frame['z'][mask_type_a]])
        pos_b = np.column_stack([frame['x'][mask_type_b], frame['y'][mask_type_b], frame['z'][mask_type_b]])

        box_size = np.array([
            box_dims[0][1] - box_dims[0][0],
            box_dims[1][1] - box_dims[1][0],
            box_dims[2][1] - box_dims[2][0]
        ])
        
        box_volume = np.prod(box_size)
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

                diff = pos_i - pos_b[j]

                # Minimimum Image convention
                ## Not a very good implementation of this, cannot handle cases for which particles are more than one periodic image away
                ### Want to standardize this in some core functionality functions - no need to duplicate periodic distances every analysis code
                for dim in range(3):
                    if diff[dim] > 0.5 * box_size[dim]:
                        diff[dim] -= box_size[dim]
                    elif diff[dim] < -0.5 * box_size[dim]:
                        diff[dim] += box_size[dim]

                distance = np.sqrt(np.sum(diff**2))

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
            'box volume': box_volume,
            'bin_centers': bin_centers,
        }
    
    # Method to get the final RDF from the internal state
    def get_rdf():
        if n_frames == 0:
            return bin_centers, np.zeros_like(bin_centers)
        
        avg_n_a = total_num_type_a / n_frames
        avg_n_b = total_num_type_b / n_frames
        avg_volume = total_volume / n_frames
        
        # Copy the histogram to avoid modifying the internal state
        hist_copy = histogram.copy()
        
        if normalize:
            # Calculate proper normalization factors
            if type_a == type_b:
                normalization = avg_n_a * (avg_n_a - 1) / 2  # This is correct
                density = avg_n_a / avg_volume
            else:
                normalization = avg_n_a * avg_n_b  # This is correct
                density = avg_n_b / avg_volume  # Density of type_b particles
            
            # Avoid division by zero
            if normalization > 0:
                # Classical RDF normalization formula:
                # g(r) = hist(r) * V / (N_A * N_B * dV(r) * n_frames)
                density = normalization / avg_volume
                for i in range(n_bins):
                        if bin_volumes[i] > 0:
                            hist_copy[i] = hist_copy[i] * avg_volume / (n_frames * normalization * bin_volumes[i])
        
        return bin_centers, hist_copy
    
    # Attach the get_rdf method to the analyzer function
    analyze_frame.get_rdf = get_rdf
    
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
    
    # Sum histograms
    for result in results:
        if not result.get('skipped', False):
            total_hist += result['frame_histogram']
    
    # Collect statistics needed for normalization
    total_n_a = sum(result.get('num_type_a', 0) for result in results)
    total_n_b = sum(result.get('num_type_b', 0) for result in results)
    total_volume = sum(result.get('bpx_volume', 0) for result in results)
    
    # Calculate average statistics
    avg_n_a = total_n_a / n_frames if n_frames > 0 else 0
    avg_n_b = total_n_b / n_frames if n_frames > 0 else 0
    avg_volume = total_volume / n_frames if n_frames > 0 else 0
    
    # Determine if same type
    same_type = results[0].get('num_type_a', 0) == results[0].get('num_type_b', 0)
    
    # Normalization factor
    if same_type:
        normalization = avg_n_a * (avg_n_a - 1) / 2
    else:
        normalization = avg_n_a * avg_n_b
    
    # Bin volumes for normalization
    max_distance = results[0].get('max_distance', bin_centers[-1] * 2)
    bin_width = max_distance / n_bins
    bin_edges = np.linspace(0, max_distance, n_bins + 1)
    bin_volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    
    # Normalize to get RDF
    rdf = np.zeros_like(total_hist)
    if normalization > 0:
        density = normalization / avg_volume
        for i in range(n_bins):
            if bin_volumes[i] > 0:
                rdf[i] = total_hist[i] / (n_frames * density * bin_volumes[i])
    
    return bin_centers, rdf