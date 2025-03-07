import time
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union, Tuple, Set
import matplotlib.pyplot as plt
from functools import wraps

from mdtoolkit.core.trajectory import LAMMPSTrajectory
from mdtoolkit.benchmarking.timing import TimingProfiler, time_function, create_analysis_timer


class TrajectoryBenchmark:
    """
    Class for benchmarking trajectory operations with varying numbers of frames.
    
    This class helps analyze how performance scales with the number of frames processed,
    which is useful for extrapolating performance for large trajectories.
    
    Attributes:
        trajectory (LAMMPSTrajectory): The trajectory to benchmark
        profiler (TimingProfiler): The profiler used for timing
        frame_counts (List[int]): List of frame counts to benchmark
        results (Dict): Dictionary storing benchmark results
    """
    
    def __init__(self, trajectory: LAMMPSTrajectory, frame_counts: Optional[List[int]] = None):
        """
        Initialize a TrajectoryBenchmark instance.
        
        Args:
            trajectory (LAMMPSTrajectory): The trajectory to benchmark
            frame_counts (List[int], optional): List of frame counts to benchmark.
                If None, defaults to [1, 2, 5, 10, 20, 50, 100] (or max frames available).
        """
        self.trajectory = trajectory
        self.profiler = TimingProfiler()
        
        # Set default frame counts if not provided
        if frame_counts is None:
            max_frames = min(100, len(trajectory))
            if max_frames <= 10:
                # For very small trajectories
                self.frame_counts = list(range(1, max_frames + 1))
            else:
                # Create a reasonable progression
                counts = [1, 2, 5]
                current = 10
                while current <= max_frames:
                    counts.append(current)
                    current *= 2
                self.frame_counts = counts
        else:
            # Ensure all requested frame counts are available
            max_available = len(trajectory)
            self.frame_counts = [fc for fc in frame_counts if fc <= max_available]
        
        self.results = {}
        
    def benchmark_frame_reading(self, repetitions: int = 3, clear_cache: bool = True) -> Dict:
        """
        Benchmark the performance of reading frames.
        
        Args:
            repetitions (int): Number of times to repeat each benchmark for statistical significance
            clear_cache (bool): Whether to clear the trajectory's cache between benchmarks
            
        Returns:
            Dict: Dictionary containing benchmark results
        """
        results = {'frame_count': [], 'total_time': [], 'time_per_frame': []}
        
        # Create a timed version of get_frame
        timed_get_frame = time_function(self.profiler, "get_frame")(self.trajectory.get_frame)
        
        # Replace the original get_frame method temporarily
        original_get_frame = self.trajectory.get_frame
        self.trajectory.get_frame = timed_get_frame
        
        try:
            for frame_count in self.frame_counts:
                # Reset timing data for this frame count
                self.profiler.reset()
                
                total_time = 0
                for rep in range(repetitions):
                    # Clear cache if requested
                    if clear_cache:
                        self.trajectory._cache.clear()
                    
                    # Time this repetition manually (bypass potential context manager issues)
                    start_time = time.perf_counter()
                    
                    # Read the frames
                    for i in range(frame_count):
                        frame_idx = i % len(self.trajectory)
                        self.trajectory.get_frame(frame_idx)
                    
                    # Record time for this repetition
                    elapsed = time.perf_counter() - start_time
                    total_time += elapsed
                
                # Calculate average time per frame
                avg_total_time = total_time / repetitions
                avg_time_per_frame = avg_total_time / frame_count if frame_count > 0 else 0
                
                # Save results
                results['frame_count'].append(frame_count)
                results['total_time'].append(avg_total_time)
                results['time_per_frame'].append(avg_time_per_frame)
        
        finally:
            # Restore the original get_frame method
            self.trajectory.get_frame = original_get_frame
        
        # Store results
        self.results['frame_reading'] = results
        
        return results
    
    def benchmark_analysis(self, analysis_fn: Callable, analysis_name: str, 
                          repetitions: int = 3, clear_cache: bool = True) -> Dict:
        """
        Benchmark the performance of an analysis function.
        
        Args:
            analysis_fn (Callable): The analysis function to benchmark
            analysis_name (str): Name for the analysis operation
            repetitions (int): Number of times to repeat each benchmark for statistical significance
            clear_cache (bool): Whether to clear the trajectory's cache between benchmarks
            
        Returns:
            Dict: Dictionary containing benchmark results
        """
        results = {
            'frame_count': [], 
            'total_time': [], 
            'analysis_time': [], 
            'read_time': [], 
            'analysis_time_per_frame': []
        }
        
        # Make a copy of the original analysis function to avoid modifying it
        original_analysis_fn = analysis_fn
        
        try:
            for frame_count in self.frame_counts:
                # Initialize timing metrics for this frame count
                total_times = []
                read_times = []
                analysis_times = []
                
                for rep in range(repetitions):
                    # Clear cache if requested
                    if clear_cache:
                        self.trajectory._cache.clear()
                    
                    # Time total execution, reading, and analysis separately
                    start_total = time.perf_counter()
                    
                    # Track frame reading time
                    read_time_start = time.perf_counter()
                    frames = [self.trajectory.get_frame(i % len(self.trajectory)) for i in range(frame_count)]
                    read_time = time.perf_counter() - read_time_start
                    
                    # Track analysis time
                    analysis_time_start = time.perf_counter()
                    for i, frame in enumerate(frames):
                        original_analysis_fn(frame, box_dims=self.trajectory.box_dims[i % len(self.trajectory)], 
                                          timestep=self.trajectory.timesteps[i % len(self.trajectory)])
                    analysis_time = time.perf_counter() - analysis_time_start
                    
                    # Record total time
                    total_time = time.perf_counter() - start_total
                    
                    total_times.append(total_time)
                    read_times.append(read_time)
                    analysis_times.append(analysis_time)
                
                # Calculate averages
                avg_total_time = sum(total_times) / repetitions
                avg_read_time = sum(read_times) / repetitions
                avg_analysis_time = sum(analysis_times) / repetitions
                avg_analysis_per_frame = avg_analysis_time / frame_count if frame_count > 0 else 0
                
                # Save results
                results['frame_count'].append(frame_count)
                results['total_time'].append(avg_total_time)
                results['read_time'].append(avg_read_time)
                results['analysis_time'].append(avg_analysis_time)
                results['analysis_time_per_frame'].append(avg_analysis_per_frame)
        
        except Exception as e:
            print(f"Exception during benchmarking: {e}")
            import traceback
            traceback.print_exc()
        
        # Store results
        self.results[f'analysis_{analysis_name}'] = results
        
        return results
    
    def extrapolate_time(self, result_key: str, target_frames: List[int]) -> Dict:
        """
        Extrapolate timing for larger frame counts based on benchmark results.
        
        Args:
            result_key (str): Key of the benchmark result to use for extrapolation
            target_frames (List[int]): List of frame counts to extrapolate to
            
        Returns:
            Dict: Dictionary with extrapolated timing information
            
        Raises:
            KeyError: If the specified result_key is not found in benchmark results
        """
        if result_key not in self.results:
            raise KeyError(f"No benchmark results found for '{result_key}'")
        
        results = self.results[result_key]
        
        # Get data for fitting
        frame_counts = np.array(results['frame_count'])
        total_times = np.array(results['total_time'])
        
        # Need at least two data points for fitting
        if len(frame_counts) < 2:
            # Use a simple linear model with slope based on the single data point
            if len(frame_counts) == 1 and frame_counts[0] > 0:
                slope = total_times[0] / frame_counts[0]
                intercept = 0
            else:
                # Default to a very basic model
                slope = 0.001  # 1ms per frame
                intercept = 0
                
            estimated_times = [slope * frames + intercept for frames in target_frames]
        else:
            # Fit a linear model: time = a * frames + b
            try:
                coeffs = np.polyfit(frame_counts, total_times, 1)
                a, b = coeffs
                
                # Extrapolate to target frames
                estimated_times = [a * frames + b for frames in target_frames]
            except Exception as e:
                print(f"Warning: Error fitting model for '{result_key}': {e}")
                # Default to a simple average
                avg_time_per_frame = sum(total_times) / sum(frame_counts) if sum(frame_counts) > 0 else 0.001
                estimated_times = [avg_time_per_frame * frames for frames in target_frames]
                a, b = avg_time_per_frame, 0
        
        extrapolated = {
            'frame_count': target_frames,
            'estimated_time': estimated_times,
            'model_coefficients': {'slope': a, 'intercept': b}
        }
        
        return extrapolated
    
    def plot_benchmark(self, result_key: str, ax=None, show_extrapolation: bool = True,
                     target_frames: Optional[List[int]] = None) -> plt.Figure:
        """
        Plot benchmark results.
        
        Args:
            result_key (str): Key of the benchmark result to plot
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, create new figure.
            show_extrapolation (bool): Whether to show extrapolated values
            target_frames (List[int], optional): Frame counts to extrapolate to
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            KeyError: If the specified result_key is not found in benchmark results
        """
        if result_key not in self.results:
            raise KeyError(f"No benchmark results found for '{result_key}'")
        
        results = self.results[result_key]
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Plot actual benchmark data
        ax.plot(results['frame_count'], results['total_time'], 'o-', label='Benchmark Data')
        
        if 'read_time' in results and 'analysis_time' in results:
            # For analysis benchmarks, also plot read and analysis times separately
            ax.plot(results['frame_count'], results['read_time'], 's--', label='Read Time')
            ax.plot(results['frame_count'], results['analysis_time'], '^--', label='Analysis Time')
        
        # Show extrapolation if requested
        if show_extrapolation:
            if target_frames is None:
                # Default extrapolation to twice the max benchmarked frame count
                max_frames = max(results['frame_count'])
                target_frames = list(range(max_frames + 1, max_frames * 2 + 1, max_frames // 5))
            
            extrapolated = self.extrapolate_time(result_key, target_frames)
            ax.plot(extrapolated['frame_count'], extrapolated['estimated_time'], 'r--', label='Extrapolated')
        
        # Add labels and legend
        ax.set_xlabel('Number of Frames')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Benchmark Results for {result_key}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
        
    def plot_scaling_analysis(self, ax=None) -> plt.Figure:
        """
        Plot how different operations scale with number of frames.
        
        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, create new figure.
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Plot time per frame for each benchmark result
        for key, results in self.results.items():
            if 'time_per_frame' in results:
                ax.plot(results['frame_count'], results['time_per_frame'], 'o-', label=f'{key} (per frame)')
            elif 'analysis_time_per_frame' in results:
                ax.plot(results['frame_count'], results['analysis_time_per_frame'], 'o-', label=f'{key} (per frame)')
        
        # Add labels and legend
        ax.set_xlabel('Number of Frames')
        ax.set_ylabel('Time per Frame (seconds)')
        ax.set_title('Scaling Analysis: Time per Frame vs. Number of Frames')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def to_report(self, include_plots: bool = True) -> Dict:
        """
        Generate a comprehensive report of all benchmark results.
        
        Args:
            include_plots (bool): Whether to include plot data in the report
            
        Returns:
            Dict: Dictionary containing benchmark results and analysis
        """
        report = {
            'trajectory_info': {
                'path': self.trajectory.filename,
                'total_frames': len(self.trajectory),
                'atoms_per_frame': self.trajectory.n_atoms
            },
            'benchmark_params': {
                'frame_counts': self.frame_counts
            },
            'results': self.results.copy(),
            'extrapolations': {}
        }
        
        # Add extrapolations for large frame counts
        for key in self.results:
            try:
                report['extrapolations'][key] = self.extrapolate_time(
                    key, 
                    [len(self.trajectory), len(self.trajectory) * 10, len(self.trajectory) * 100]
                )
            except Exception as e:
                print(f"Warning: Could not extrapolate time for '{key}': {e}")
                report['extrapolations'][key] = {
                    'frame_count': [len(self.trajectory), len(self.trajectory) * 10, len(self.trajectory) * 100],
                    'estimated_time': [0, 0, 0],
                    'model_coefficients': {'slope': 0, 'intercept': 0}
                }
        
        # Add plots if requested
        if include_plots:
            report['plots'] = {}
            # Note: In a real implementation, you would convert plots to image data or save them to files
            
        return report
