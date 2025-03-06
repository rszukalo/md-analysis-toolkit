import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union, Type, TypeVar

T = TypeVar('T', bound='AnalysisResult')

class AnalysisResult:
    """
    Base class for analysis results with standardized save/load methods.
    
    This class provides a common interface for saving and loading analysis results
    in different formats (npy and json).
    
    Attributes:
        data (Dict[str, np.ndarray]): Dictionary of data arrays
        metadata (Dict[str, Any]): Dictionary of metadata about the analysis
    """
    
    def __init__(self, data: Dict[str, np.ndarray], metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an AnalysisResult object.
        
        Args:
            data (Dict[str, np.ndarray]): Dictionary of data arrays
            metadata (Dict[str, Any], optional): Dictionary of metadata about the analysis
        """

        self.data = data
        self.metadata = metadata or {}
        
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()
        
        if 'analysis_type' not in self.metadata:
            self.metadata['analysis_type'] = self.__class__.__name__
            
    def save(self, filename: str) -> None:
        """
        Save result to file, format determined by extension (.npy or .json).
        
        Args:
            filename (str): Output filename with extension
            
        Raises:
            ValueError: If file extension is not supported
        """

        ext = os.path.splitext(filename)[1].lower()
        if ext == '.npy':
            self.save_npy(filename)
        elif ext == '.json':
            self.save_json(filename)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .npy or .json")
            
    def save_npy(self, filename: str) -> None:
        """
        Save as NumPy binary file (efficient storage).
        
        Args:
            filename (str): Output filename
        """

        # Create a dictionary with both data and metadata
        save_dict = {
            'data': self.data,
            'metadata': self.metadata
        }
        np.save(filename, save_dict, allow_pickle=True)
        
    def save_json(self, filename: str) -> None:
        """
        Save as JSON (human readable).
        
        Args:
            filename (str): Output filename
        """

        # Create a JSON-serializable dictionary
        save_dict = {
            'data': {k: v.tolist() for k, v in self.data.items()},
            'metadata': self.metadata
        }
        
        with open(filename, 'w') as f:
            json.dump(save_dict, f, indent=2)
    
    @classmethod
    def load(cls: Type[T], filename: str) -> T:
        """
        Load result from file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            AnalysisResult: Loaded result object
            
        Raises:
            ValueError: If file extension is not supported
        """

        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.npy':
            return cls._load_npy(filename)
        elif ext == '.json':
            return cls._load_json(filename)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .npy or .json")
    
    @classmethod
    def _load_npy(cls: Type[T], filename: str) -> T:
        """
        Load from NumPy binary file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            AnalysisResult: Loaded result object
        """

        loaded = np.load(filename, allow_pickle=True).item()
        return cls(loaded['data'], loaded['metadata'])
    
    @classmethod
    def _load_json(cls: Type[T], filename: str) -> T:
        """
        Load from JSON file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            AnalysisResult: Loaded result object
        """
        
        with open(filename, 'r') as f:
            loaded = json.load(f)
        
        data = {k: np.array(v) for k, v in loaded['data'].items()}
        
        return cls(data, loaded['metadata'])