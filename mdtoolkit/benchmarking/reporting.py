import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime


class BenchmarkReport:
    """
    Class for generating and visualizing benchmark reports.
    
    This class provides methods for saving, loading, and visualizing benchmark results.
    It works with the data produced by TrajectoryBenchmark.
    
    Attributes:
        data (Dict): The benchmark report data
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize a BenchmarkReport instance.
        
        Args:
            data (Dict[str, Any]): Benchmark report data
        """
        self.data = data if data is not None else {}
        
        # Add timestamp if not present
        if 'timestamp' not in self.data:
            self.data['timestamp'] = datetime.now().isoformat()
    
    @classmethod
    def from_benchmark(cls, benchmark):
        """
        Create a BenchmarkReport from a TrajectoryBenchmark instance.
        
        Args:
            benchmark: A TrajectoryBenchmark instance
            
        Returns:
            BenchmarkReport: A new report instance
        """
        try:
            report_data = benchmark.to_report()
            return cls(report_data)
        except Exception as e:
            print(f"Warning: Error generating report from benchmark: {e}")
            # Create with minimal data to avoid errors
            return cls({
                'trajectory_info': {
                    'path': benchmark.trajectory.filename,
                    'total_frames': len(benchmark.trajectory),
                    'atoms_per_frame': benchmark.trajectory.n_atoms
                },
                'benchmark_params': {
                    'frame_counts': benchmark.frame_counts
                },
                'results': benchmark.results.copy(),
                'extrapolations': {}
            })
    
    def save(self, filename: str) -> None:
        """
        Save the benchmark report to a file.
        
        Args:
            filename (str): Output filename (JSON)
        """
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    @classmethod
    def load(cls, filename: str) -> 'BenchmarkReport':
        """
        Load a benchmark report from a file.
        
        Args:
            filename (str): Input filename (JSON)
            
        Returns:
            BenchmarkReport: Loaded report instance
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls(data)
    
    def plot_all_benchmarks(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot all benchmark results in a single figure.
        
        Args:
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Check if we have any results
        if 'results' not in self.data or not self.data.get('results'):
            ax.set_xlabel('Number of Frames')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Benchmark Results - No Data Available')
            return fig
        
        # Plot each benchmark result
        for key, results in self.data['results'].items():
            if 'frame_count' in results and 'total_time' in results:
                ax.plot(results['frame_count'], results['total_time'], 'o-', label=key)
        
        # Add labels and legend
        ax.set_xlabel('Number of Frames')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Benchmark Results')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_comparison(self, keys: List[str], figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot a comparison of selected benchmark results.
        
        Args:
            keys (List[str]): List of benchmark keys to compare
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            KeyError: If any of the specified keys is not found in benchmark results
        """
        for key in keys:
            if key not in self.data['results']:
                raise KeyError(f"No benchmark results found for '{key}'")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot selected benchmark results
        for key in keys:
            results = self.data['results'][key]
            ax.plot(results['frame_count'], results['total_time'], 'o-', label=key)
        
        # Add labels and legend
        ax.set_xlabel('Number of Frames')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Benchmark Comparison')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_extrapolation(self, key: str, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot benchmark results with extrapolation.
        
        Args:
            key (str): Benchmark key to plot
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            KeyError: If the specified key is not found in benchmark results or extrapolations
        """
        # Check if the data exists
        if 'results' not in self.data or key not in self.data.get('results', {}):
            print(f"Warning: No benchmark results found for '{key}'")
            # Create a basic empty plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel('Number of Frames')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'Benchmark Extrapolation for {key} - No Data Available')
            return fig
            
        if 'extrapolations' not in self.data or key not in self.data.get('extrapolations', {}):
            print(f"Warning: No extrapolation found for '{key}'")
            # Create a plot with just the benchmark data
            fig, ax = plt.subplots(figsize=figsize)
            results = self.data['results'][key]
            ax.plot(results.get('frame_count', []), results.get('total_time', []), 'o-', label='Benchmark Data')
            ax.set_xlabel('Number of Frames')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'Benchmark Data for {key} - No Extrapolation Available')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            return fig
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual benchmark data
        results = self.data['results'][key]
        ax.plot(results.get('frame_count', []), results.get('total_time', []), 'o-', label='Benchmark Data')
        
        # Plot extrapolation
        extrapolation = self.data['extrapolations'][key]
        ax.plot(extrapolation.get('frame_count', []), extrapolation.get('estimated_time', []), 'r--', label='Extrapolated')
        
        # Add labels and legend
        ax.set_xlabel('Number of Frames')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Benchmark Extrapolation for {key}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def print_summary(self) -> None:
        """Print a summary of benchmark results to the console."""
        print(f"=== Benchmark Summary ===")
        print(f"Trajectory: {self.data['trajectory_info']['path']}")
        print(f"Total frames: {self.data['trajectory_info']['total_frames']}")
        print(f"Atoms per frame: {self.data['trajectory_info']['atoms_per_frame']}")
        print(f"Benchmark frame counts: {self.data['benchmark_params']['frame_counts']}")
        print()
        
        # Print results for each benchmark
        for key, results in self.data['results'].items():
            print(f"--- {key} ---")
            max_frame_idx = len(results['frame_count']) - 1
            print(f"Time for {results['frame_count'][max_frame_idx]} frames: {results['total_time'][max_frame_idx]:.4f} seconds")
            
            if 'time_per_frame' in results:
                print(f"Average time per frame: {results['time_per_frame'][max_frame_idx]:.4f} seconds")
            elif 'analysis_time_per_frame' in results:
                print(f"Average analysis time per frame: {results['analysis_time_per_frame'][max_frame_idx]:.4f} seconds")
            
            # Print extrapolations if available
            if key in self.data['extrapolations']:
                print(f"Extrapolations:")
                extrap = self.data['extrapolations'][key]
                for i, frames in enumerate(extrap['frame_count']):
                    print(f"  {frames} frames: {extrap['estimated_time'][i]:.4f} seconds")
            
            print()
    
    def plot_scaling(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot how time per frame changes with the number of frames.
        
        Args:
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for key, results in self.data['results'].items():
            if 'time_per_frame' in results:
                ax.plot(results['frame_count'], results['time_per_frame'], 'o-', label=f"{key} (per frame)")
            elif 'analysis_time_per_frame' in results:
                ax.plot(results['frame_count'], results['analysis_time_per_frame'], 'o-', label=f"{key} (per frame)")
        
        ax.set_xlabel('Number of Frames')
        ax.set_ylabel('Time per Frame (seconds)')
        ax.set_title('Scaling Analysis: Time per Frame vs. Number of Frames')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def estimate_time_for_frames(self, key: str, frame_count: int) -> float:
        """
        Estimate time for processing a given number of frames.
        
        Args:
            key (str): Benchmark key to use for estimation
            frame_count (int): Number of frames to estimate
            
        Returns:
            float: Estimated time in seconds
            
        Raises:
            KeyError: If the specified key is not found in extrapolations
        """
        if 'extrapolations' not in self.data or key not in self.data.get('extrapolations', {}):
            print(f"Warning: No extrapolation found for '{key}'. Using default estimate.")
            # Use a default estimate (1ms per frame)
            return 0.001 * frame_count
        
        extrap = self.data['extrapolations'][key]
        
        # Use the model coefficients for estimation
        a = extrap.get('model_coefficients', {}).get('slope', 0.001)
        b = extrap.get('model_coefficients', {}).get('intercept', 0)
        
        estimated_time = a * frame_count + b
        
        return estimated_time