import time
import functools
import contextlib
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
import numpy as np
from collections import defaultdict


class TimingProfiler:
    """
    A class for profiling and recording execution times of operations in the MD toolkit.
    
    This class provides methods to time function calls, track statistics for different
    operations, and generate summaries of timing data. It's designed to be used either
    directly or through the provided decorators and context managers.
    
    Attributes:
        operations (Dict[str, List[float]]): Dictionary mapping operation names to lists of timing measurements
        current_operation (Optional[str]): Name of the operation currently being timed
        start_time (Optional[float]): Start time of the current operation
    """
    
    def __init__(self):
        """Initialize a new TimingProfiler instance."""
        self.operations = defaultdict(list)
        self.current_operation = None
        self.start_time = None
    
    def start(self, operation_name: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation_name (str): Name of the operation to time
        """
        if self.current_operation is not None:
            raise RuntimeError(f"Already timing operation '{self.current_operation}'")
        
        self.current_operation = operation_name
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """
        Stop timing the current operation and record the elapsed time.
        
        Returns:
            float: Elapsed time in seconds
            
        Raises:
            RuntimeError: If no operation is currently being timed
        """
        if self.current_operation is None:
            raise RuntimeError("No operation is currently being timed")
        
        elapsed_time = time.perf_counter() - self.start_time
        self.operations[self.current_operation].append(elapsed_time)
        
        self.current_operation = None
        self.start_time = None
        
        return elapsed_time
    
    def time_operation(self, operation_name: str) -> contextlib.contextmanager:
        """
        Create a context manager for timing an operation.
        
        Args:
            operation_name (str): Name of the operation to time
            
        Returns:
            contextlib.ContextManager: Context manager that times the operation
            
        Example:
            >>> profiler = TimingProfiler()
            >>> with profiler.time_operation("read_frame"):
            >>>     frame = trajectory.get_frame(0)
        """
        return _TimingContextManager(self, operation_name)
    
    def get_times(self, operation_name: str) -> List[float]:
        """
        Get all recorded times for a specific operation.
        
        Args:
            operation_name (str): Name of the operation
            
        Returns:
            List[float]: List of elapsed times in seconds
        """
        return self.operations.get(operation_name, [])
    
    def get_statistics(self, operation_name: str) -> Dict[str, float]:
        """
        Get timing statistics for a specific operation.
        
        Args:
            operation_name (str): Name of the operation
            
        Returns:
            Dict[str, float]: Dictionary containing the following statistics:
                - count: Number of measurements
                - total: Total elapsed time
                - min: Minimum elapsed time
                - max: Maximum elapsed time
                - mean: Mean elapsed time
                - median: Median elapsed time
                - std: Standard deviation of elapsed times
        """
        times = self.get_times(operation_name)
        
        if not times:
            return {
                'count': 0,
                'total': 0.0,
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0
            }
        
        np_times = np.array(times)
        
        return {
            'count': len(times),
            'total': np.sum(np_times),
            'min': np.min(np_times),
            'max': np.max(np_times),
            'mean': np.mean(np_times),
            'median': np.median(np_times),
            'std': np.std(np_times)
        }
    
    def reset(self, operation_name: Optional[str] = None) -> None:
        """
        Reset timing data.
        
        Args:
            operation_name (str, optional): Name of operation to reset. If None, reset all operations.
        """
        if operation_name is None:
            self.operations.clear()
        else:
            if operation_name in self.operations:
                self.operations[operation_name] = []
    
    def get_all_operations(self) -> List[str]:
        """
        Get a list of all operations that have been timed.
        
        Returns:
            List[str]: List of operation names
        """
        return list(self.operations.keys())
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get a summary of timing statistics for all operations.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping operation names to their statistics
        """
        return {op: self.get_statistics(op) for op in self.get_all_operations()}
    
    def print_summary(self, sort_by: str = 'total') -> None:
        """
        Print a summary of timing statistics for all operations.
        
        Args:
            sort_by (str): Statistic to sort by ('total', 'mean', 'count', etc.)
        """
        summary = self.get_summary()
        
        # Sort operations by the specified statistic
        sorted_ops = sorted(
            summary.items(),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=True
        )
        
        # Print header
        print(f"{'Operation':<30} {'Count':<8} {'Total (s)':<12} {'Mean (ms)':<12} {'Median (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Std (ms)':<12}")
        print("-" * 110)
        
        # Print each operation
        for op_name, stats in sorted_ops:
            print(
                f"{op_name:<30} "
                f"{stats['count']:<8} "
                f"{stats['total']:<12.4f} "
                f"{stats['mean']*1000:<12.4f} "
                f"{stats['median']*1000:<12.4f} "
                f"{stats['min']*1000:<12.4f} "
                f"{stats['max']*1000:<12.4f} "
                f"{stats['std']*1000:<12.4f}"
            )


class _TimingContextManager:
    """Context manager for timing operations with TimingProfiler."""
    
    def __init__(self, profiler: TimingProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
    
    def __enter__(self):
        self.profiler.start(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop()
        return False  # Don't suppress exceptions


def time_function(profiler: TimingProfiler, operation_name: Optional[str] = None):
    """
    Decorator for timing functions.
    
    Args:
        profiler (TimingProfiler): The profiler to use for timing
        operation_name (str, optional): Name of the operation. If None, use the function name.
        
    Returns:
        Callable: Decorated function
        
    Example:
        >>> profiler = TimingProfiler()
        >>> @time_function(profiler)
        >>> def process_data(data):
        >>>     # Process data...
        >>>     return result
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            profiler.start(op_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.stop()
        return wrapper
    return decorator


def create_analysis_timer(profiler: TimingProfiler, analysis_name: str):
    """
    Create a decorator specifically for timing analysis functions.
    
    This decorator is designed to be used with the analysis functions passed to
    LAMMPSTrajectory.analyze_trajectory.
    
    Args:
        profiler (TimingProfiler): The profiler to use for timing
        analysis_name (str): Name of the analysis operation
        
    Returns:
        Callable: Decorator for timing analysis functions
        
    Example:
        >>> profiler = TimingProfiler()
        >>> rdf_analyzer = create_rdf_analyzer(max_distance=10.0, n_bins=100, type_a=1)
        >>> timed_rdf_analyzer = create_analysis_timer(profiler, "rdf_calculation")(rdf_analyzer)
        >>> trajectory.analyze_trajectory(timed_rdf_analyzer)
    """
    def decorator(analysis_fn):
        @functools.wraps(analysis_fn)
        def wrapper(frame, *args, **kwargs):
            # Record the timestep if available in kwargs
            operation_suffix = f"_{kwargs.get('timestep', '')}" if 'timestep' in kwargs else ""
            operation_name = f"{analysis_name}{operation_suffix}"
            
            with profiler.time_operation(operation_name):
                return analysis_fn(frame, *args, **kwargs)
        
        # Preserve any attributes or methods of the original function
        for attr_name in dir(analysis_fn):
            if not attr_name.startswith('__'):
                attr = getattr(analysis_fn, attr_name)
                if callable(attr):
                    # Also time any attached methods
                    timed_attr = time_function(profiler, f"{analysis_name}.{attr_name}")(attr)
                    setattr(wrapper, attr_name, timed_attr)
                else:
                    setattr(wrapper, attr_name, attr)
        
        return wrapper
    return decorator