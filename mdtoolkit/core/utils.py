import numpy as np
from typing import Tuple, Union, List, Optional, Any, Callable

def minimum_image_distance(pos_a: np.ndarray, pos_b: np.ndarray, box_lengths: np.ndarray, orthorhombic: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate the minimum image distance between two points under periodic boundary conditions.
    
    This implementation correctly handles particles that are more than one periodic image apart
    by using the modulo operation for wrapping coordinates.
    
    Args:
        pos_a (np.ndarray): Position of first particle, shape (3,) or (n, 3)
        pos_b (np.ndarray): Position of second particle, shape (3,) or (n, 3)
        box_lengths (np.ndarray): Box dimensions in each direction, shape (3,)
        orthorhombic (bool): If True, assumes an orthorhombic box. 
                            If False, requires full box information (not yet implemented)
                            
    Returns:
        Union[float, np.ndarray]: Minimum image distance(s)
    
    Raises:
        NotImplementedError: If orthorhombic=False, as non-orthorhombic boxes are not yet supported
        
    Notes:
        - For non-orthorhombic boxes, additional information about the box tilt factors
          would be required. This functionality will be added in the future.
        - When passing arrays of positions, pos_a and pos_b must have the same shape.
    """

    if not orthorhombic:
        raise NotImplementedError("Non-orthorhombic boxes are not yet supported")
    
    single_point = (len(pos_a.shape) == 1)
    
    if single_point:
        pos_a = pos_a.reshape(1, 3)
        pos_b = pos_b.reshape(1, 3)
    
    diff = pos_a - pos_b
    
    # Apply minimum image convention using modulo arithmetic
    diff = np.remainder(diff + box_lengths/2, box_lengths) - box_lengths/2
    
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    if single_point:
        return distances[0]
    else:
        return distances


def minimum_image_vector(pos_a: np.ndarray, pos_b: np.ndarray, box_lengths: np.ndarray, orthorhombic: bool = True) -> np.ndarray:
    """
    Calculate the minimum image vector from pos_b to pos_a under periodic boundary conditions.
    
    Args:
        pos_a (np.ndarray): Position of first particle, shape (3,) or (n, 3)
        pos_b (np.ndarray): Position of second particle, shape (3,) or (n, 3)
        box_lengths (np.ndarray): Box dimensions in each direction, shape (3,)
        orthorhombic (bool): If True, assumes an orthorhombic box. 
                            If False, requires full box information (not yet implemented)
                            
    Returns:
        np.ndarray: Minimum image vector(s) from pos_b to pos_a, shape (3,) or (n, 3)
    
    Raises:
        NotImplementedError: If orthorhombic=False, as non-orthorhombic boxes are not yet supported
    """

    if not orthorhombic:
        raise NotImplementedError("Non-orthorhombic boxes are not yet supported")
    
    single_point = (len(pos_a.shape) == 1)
    
    if single_point:
        pos_a = pos_a.reshape(1, 3)
        pos_b = pos_b.reshape(1, 3)
    
    diff = pos_a - pos_b
    
    # Apply minimum image convention using modulo arithmetic
    diff = np.remainder(diff + box_lengths/2, box_lengths) - box_lengths/2
    
    if single_point:
        return diff[0]
    else:
        return diff


def extract_box_lengths(box_dims: np.ndarray) -> np.ndarray:
    """
    Extract box lengths from box dimensions array.
    
    Args:
        box_dims (np.ndarray): Box dimensions as stored in LAMMPSTrajectory
                              shape [3, 2] where [:,0] is min and [:,1] is max
        
    Returns:
        np.ndarray: Box lengths in each dimension, shape (3,)
    """
    
    return np.array([
        box_dims[0][1] - box_dims[0][0],  # x length
        box_dims[1][1] - box_dims[1][0],  # y length
        box_dims[2][1] - box_dims[2][0]   # z length
    ])