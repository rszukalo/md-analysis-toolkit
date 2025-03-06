import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

from mdtoolkit.output.output_base import AnalysisResult
from mdtoolkit.analysis.rdf import process_rdf_results


class RDFResult(AnalysisResult):
    """
    Container for Radial Distribution Function (RDF) analysis results.
    
    This class provides specific methods for working with RDF data,
    including property accessors for distances and RDF values.
    
    Attributes:
        data (Dict[str, np.ndarray]): Dictionary containing 'distances' and 'rdf_values'
        metadata (Dict[str, Any]): Dictionary of metadata about the analysis
    """
    
    def __init__(self, distances: np.ndarray, rdf_values: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an RDFResult object.
        
        Args:
            distances (np.ndarray): Array of distance values (r)
            rdf_values (np.ndarray): Array of g(r) values
            metadata (Dict[str, Any], optional): Dictionary of metadata about the analysis
        """
        # Store the specific data for RDF analysis
        data = {
            'distances': np.asarray(distances),
            'rdf_values': np.asarray(rdf_values)
        }
        
        # Initialize with default metadata
        metadata = metadata or {}
        if 'analysis_type' not in metadata:
            metadata['analysis_type'] = 'RDF'
            
        super().__init__(data, metadata)
    
    @property
    def distances(self) -> np.ndarray:
        """
        Get the distance values (r).
        
        Returns:
            np.ndarray: Array of distance values
        """
        # Ensure we return a proper numpy array
        return np.asarray(self.data['distances'])
    
    @property
    def rdf_values(self) -> np.ndarray:
        """
        Get the g(r) values.
        
        Returns:
            np.ndarray: Array of g(r) values
        """
        # Ensure we return a proper numpy array
        return np.asarray(self.data['rdf_values'])
    
    @classmethod
    def from_analyzer_output(cls, analyzer_results: Union[List[Dict[str, Any]], Callable], 
                           metadata: Optional[Dict[str, Any]] = None) -> 'RDFResult':
        """
        Create RDFResult from the output of an RDF analyzer.
        
        This method handles both:
        1. The function object returned by create_rdf_analyzer (with get_rdf method)
        2. The list of results from trajectory.analyze_trajectory
        
        Args:
            analyzer_results: Either the analyzer function or list of results
            metadata (Dict[str, Any], optional): Additional metadata to include
            
        Returns:
            RDFResult: A properly initialized result object
            
        Raises:
            ValueError: If the analyzer results cannot be processed
        """
        # Try to extract results using different methods
        try:
            if hasattr(analyzer_results, 'get_rdf'):
                # If it's the analyzer function with attached get_rdf method
                distances, rdf_values = analyzer_results.get_rdf()
            else:
                # Assume it's the list of results from analyze_trajectory
                distances, rdf_values = process_rdf_results(analyzer_results)
                
            # Create result object
            return cls(distances, rdf_values, metadata)
            
        except Exception as e:
            raise ValueError(f"Could not process analyzer results: {e}")
    
    def get_first_peak(self) -> Tuple[float, float]:
        """
        Get the position and height of the first peak in the RDF.
        
        Returns:
            Tuple[float, float]: (position, height) of the first peak
        """
        # Ensure we have data to work with
        distances = self.distances
        rdf_values = self.rdf_values
        
        if len(distances) == 0:
            raise ValueError("No distance data available")
            
        # Skip the first few bins as they often have artifacts (r -> 0)
        start_idx = max(3, len(distances) // 20)  # Skip first 5% or at least 3 bins
        
        # Find the maximum in the remaining data
        peak_idx = start_idx + np.argmax(rdf_values[start_idx:])
        peak_position = distances[peak_idx]
        peak_height = rdf_values[peak_idx]
        
        return peak_position, peak_height
    
    def get_coordination_number(self, r_min: float = 0.0, r_max: Optional[float] = None) -> float:
        """
        Calculate the coordination number by integrating the RDF.
        
        The coordination number is calculated as:
        CN = 4πρ∫(r_min to r_max) g(r) r² dr
        
        Args:
            r_min (float): Minimum integration distance
            r_max (float, optional): Maximum integration distance. If None, use the first minimum after the first peak.
            
        Returns:
            float: Coordination number
        """
        if 'number_density' not in self.metadata:
            raise ValueError("Number density not found in metadata. Cannot calculate coordination number.")
        
        # Ensure we have data to work with
        distances = self.distances
        rdf_values = self.rdf_values
        
        if len(distances) == 0:
            raise ValueError("No distance data available")
            
        density = self.metadata['number_density']
        
        # Find suitable r_max if not provided
        if r_max is None:
            peak_pos, _ = self.get_first_peak()
            # Find the first minimum after the peak
            peak_idx = np.argmin(np.abs(distances - peak_pos))
            after_peak = rdf_values[peak_idx:]
            min_idx = peak_idx + np.argmin(after_peak)
            r_max = distances[min_idx]
        
        # Find indices corresponding to r_min and r_max
        idx_min = np.argmin(np.abs(distances - r_min))
        idx_max = np.argmin(np.abs(distances - r_max))
        
        # Ensure indices are within valid range
        idx_min = max(0, idx_min)
        idx_max = min(len(distances) - 1, idx_max)
        
        # Get the relevant slice of data
        r = distances[idx_min:idx_max+1]
        g_r = rdf_values[idx_min:idx_max+1]
        
        # Calculate coordination number using trapezoidal integration
        integrand = 4 * np.pi * density * g_r * r**2
        dr = r[1] - r[0]  # Assuming uniform spacing
        
        # Trapezoidal rule integration
        coord_num = np.trapz(integrand, dx=dr)
        
        return coord_num
    
    @classmethod
    def _load_npy(cls, filename: str) -> 'RDFResult':
        """
        Load from NumPy binary file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            RDFResult: Loaded result object
        """
        loaded = np.load(filename, allow_pickle=True).item()
        
        # Create a new instance with the loaded data
        data = loaded['data']
        metadata = loaded['metadata']
        
        # Extract the key arrays
        distances = np.asarray(data['distances'])
        rdf_values = np.asarray(data['rdf_values'])
        
        return cls(distances, rdf_values, metadata)
    
    @classmethod
    def _load_json(cls, filename: str) -> 'RDFResult':
        """
        Load from JSON file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            RDFResult: Loaded result object
        """
        with open(filename, 'r') as f:
            loaded = json.load(f)
        
        # Convert lists back to numpy arrays
        data = loaded['data']
        metadata = loaded['metadata']
        
        # Extract the key arrays
        distances = np.asarray(data['distances'])
        rdf_values = np.asarray(data['rdf_values'])
        
        return cls(distances, rdf_values, metadata)