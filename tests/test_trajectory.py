import os
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, mock_open

from mdtoolkit.core.trajectory import LAMMPSTrajectory

class TestLAMMPSTrajectory(unittest.TestCase):
    """Test cases for LAMMPSTrajectory class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""

        self.temp_file = self._create_mock_dump_file()
        
    def tearDown(self):
        """Clean up after each test method."""

        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def _create_mock_dump_file(self):
        """Create a mock LAMMPS dump file for testing."""

        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            # Frame 1
            f.write("ITEM: TIMESTEP\n")
            f.write("1000\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write("4\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("ITEM: ATOMS id type x y z vx vy vz\n")
            f.write("1 1 1.0 2.0 3.0 0.1 0.2 0.3\n")
            f.write("2 1 4.0 5.0 6.0 0.4 0.5 0.6\n")
            f.write("3 2 7.0 8.0 9.0 0.7 0.8 0.9\n")
            f.write("4 2 10.0 11.0 12.0 1.0 1.1 1.2\n")
            
            # Frame 2
            f.write("ITEM: TIMESTEP\n")
            f.write("2000\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write("4\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("ITEM: ATOMS id type x y z vx vy vz\n")
            f.write("1 1 1.1 2.1 3.1 0.11 0.21 0.31\n")
            f.write("2 1 4.1 5.1 6.1 0.41 0.51 0.61\n")
            f.write("3 2 7.1 8.1 9.1 0.71 0.81 0.91\n")
            f.write("4 2 10.1 11.1 12.1 1.01 1.11 1.21\n")
            
            return f.name
    
    def test_init(self):
        """Test initialization of LAMMPSTrajectory."""
        traj = LAMMPSTrajectory(self.temp_file)
        
        # Verify basic properties
        self.assertEqual(traj.filename, self.temp_file)
        self.assertEqual(traj.n_frames, 2)
        self.assertEqual(traj.n_atoms, 4)
        self.assertEqual(traj.atom_columns, ['id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        self.assertEqual(len(traj.timesteps), 2)
        self.assertEqual(traj.timesteps[0], 1000)
        self.assertEqual(traj.timesteps[1], 2000)
        
        # Check box dimensions - note: we're testing just the shape of the first two dimensions
        # and the content of the bounds
        self.assertEqual(traj.box_dims.shape[0], 2)  # 2 frames
        self.assertEqual(traj.box_dims.shape[1], 3)  # 3 dimensions (x, y, z)
        
        # Check the actual bounds values
        np.testing.assert_array_equal(traj.box_dims[0][0][:2], [0.0, 10.0])
        np.testing.assert_array_equal(traj.box_dims[0][1][:2], [0.0, 10.0])
        np.testing.assert_array_equal(traj.box_dims[0][2][:2], [0.0, 10.0])
    
    def test_triclinic_box(self):
        """Test that the trajectory reader can handle triclinic box formats."""

        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("1000\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write("4\n")
            f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            f.write("0.0 10.0 1.0\n")  # xlo xhi xy
            f.write("0.0 10.0 1.0\n")  # ylo yhi xz
            f.write("0.0 10.0 1.0\n")  # zlo zhi yz
            f.write("ITEM: ATOMS id type x y z\n")
            f.write("1 1 1.0 2.0 3.0\n")
            f.write("2 1 4.0 5.0 6.0\n")
            f.write("3 2 7.0 8.0 9.0\n")
            f.write("4 2 10.0 11.0 12.0\n")
            
            triclinic_file = f.name
        
        try:
            # Test with triclinic box format
            traj = LAMMPSTrajectory(triclinic_file)
            
            # Check that we can still read the basic bounds correctly
            self.assertEqual(traj.box_dims.shape[0], 1)  # 1 frame
            self.assertEqual(traj.box_dims.shape[1], 3)  # 3 dimensions
            
            # Check the bounds values (should have the first two values from each line)
            np.testing.assert_array_equal(traj.box_dims[0][0][:2], [0.0, 10.0])
            np.testing.assert_array_equal(traj.box_dims[0][1][:2], [0.0, 10.0])
            np.testing.assert_array_equal(traj.box_dims[0][2][:2], [0.0, 10.0])
            
        finally:
            # Clean up
            if os.path.exists(triclinic_file):
                os.remove(triclinic_file)

    def test_get_frame(self):
        """Test get_frame method."""
        traj = LAMMPSTrajectory(self.temp_file)
        
        # Get first frame
        frame0 = traj.get_frame(0)
        
        for col in ['id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz']:
            self.assertIn(col, frame0)
        
        np.testing.assert_array_equal(frame0['id'], [1, 2, 3, 4])
        np.testing.assert_array_equal(frame0['type'], [1, 1, 2, 2])
        np.testing.assert_array_equal(frame0['x'], [1.0, 4.0, 7.0, 10.0])
        
        # Get second frame
        frame1 = traj.get_frame(1)
        
        np.testing.assert_array_equal(frame1['x'], [1.1, 4.1, 7.1, 10.1])
    
    def test_frame_cache(self):
        """Test the frame caching mechanism."""
        traj = LAMMPSTrajectory(self.temp_file, cache_size=1)
        
        # Get first frame - should be cached
        frame0 = traj.get_frame(0)
        self.assertIn(0, traj._cache)
        
        # Get second frame - should replace first frame in cache
        frame1 = traj.get_frame(1)
        self.assertIn(1, traj._cache)
        self.assertNotIn(0, traj._cache)  # First frame should be evicted
        
        # Get first frame again - should reload from file
        frame0_again = traj.get_frame(0)
        self.assertIn(0, traj._cache)
        self.assertNotIn(1, traj._cache)  # Second frame should be evicted
        
        # Verify data is the same
        np.testing.assert_array_equal(frame0['x'], frame0_again['x'])

    def test_iterate_frames(self):
        """Test iterate_frames method."""
        traj = LAMMPSTrajectory(self.temp_file)
        
        # Collect frames from iterator
        frames = []
        for idx, frame in traj.iterate_frames():
            frames.append((idx, frame))
        
        # Check that we got all frames
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][0], 0)  # First index
        self.assertEqual(frames[1][0], 1)  # Second index
        
        # Check actual data
        np.testing.assert_array_equal(frames[0][1]['x'], [1.0, 4.0, 7.0, 10.0])
        np.testing.assert_array_equal(frames[1][1]['x'], [1.1, 4.1, 7.1, 10.1])
        
        # Test with start, end, step
        frames_subset = []
        for idx, frame in traj.iterate_frames(start=1, end=2, step=1):
            frames_subset.append((idx, frame))
        
        # Should only get second frame
        self.assertEqual(len(frames_subset), 1)
        self.assertEqual(frames_subset[0][0], 1)
    
    def test_analyze_trajectory(self):
        """Test analyze_trajectory method."""
        traj = LAMMPSTrajectory(self.temp_file)
        
        # Define a simple analysis function
        def analyze_mean_position(frame, box_dims, timestep, **kwargs):
            return np.mean(frame['x']), np.mean(frame['y']), np.mean(frame['z'])
        
        # Analyze the trajectory
        results = traj.analyze_trajectory(analyze_mean_position)
        
        # Check results
        self.assertEqual(len(results), 2)
        # First frame mean positions
        self.assertAlmostEqual(results[0][0], 5.5)  # Mean x
        self.assertAlmostEqual(results[0][1], 6.5)  # Mean y
        self.assertAlmostEqual(results[0][2], 7.5)  # Mean z
        
        # Second frame mean positions (should be slightly higher)
        self.assertAlmostEqual(results[1][0], 5.6)  # Mean x
        self.assertAlmostEqual(results[1][1], 6.6)  # Mean y
        self.assertAlmostEqual(results[1][2], 7.6)  # Mean z
    
    def test_select_atoms(self):
        """Test select_atoms method."""
        traj = LAMMPSTrajectory(self.temp_file)
        
        # Select atoms of type 1
        type1_mask = traj.select_atoms(0, {'type': 1})
        self.assertEqual(sum(type1_mask), 2)  # Should have 2 atoms of type 1
        
        # Select atoms where x > 5
        x_mask = traj.select_atoms(0, {'x': lambda x: x > 5})
        self.assertEqual(sum(x_mask), 2)  # Should have 2 atoms with x > 5
        
        # Combined selection
        combined_mask = traj.select_atoms(0, {
            'type': 2,
            'z': lambda z: z > 10
        })
        self.assertEqual(sum(combined_mask), 1)  # Only one atom meets both criteria
    
    def test_get_atom_properties(self):
        """Test methods for getting atom properties."""
        traj = LAMMPSTrajectory(self.temp_file)
        
        # Test get_atom_ids
        ids = traj.get_atom_ids(0)
        np.testing.assert_array_equal(ids, [1, 2, 3, 4])
        
        # Test get_atom_types
        types = traj.get_atom_types(0)
        np.testing.assert_array_equal(types, [1, 1, 2, 2])
        
        # Test get_atom_property
        x_coords = traj.get_atom_property(0, 'x')
        np.testing.assert_array_equal(x_coords, [1.0, 4.0, 7.0, 10.0])
    
    def test_file_not_found(self):
        """Test behavior when file is not found."""
        with self.assertRaises(FileNotFoundError):
            LAMMPSTrajectory("non_existent_file.dump")
    
    def test_invalid_frame_index(self):
        """Test behavior with invalid frame index."""
        traj = LAMMPSTrajectory(self.temp_file)
        
        with self.assertRaises(IndexError):
            traj.get_frame(-1)  # Negative index
        
        with self.assertRaises(IndexError):
            traj.get_frame(2)  # Index beyond n_frames
    
    def test_len_and_getitem(self):
        """Test __len__ and __getitem__ methods."""
        traj = LAMMPSTrajectory(self.temp_file)
        
        # Test __len__
        self.assertEqual(len(traj), 2)
        
        # Test __getitem__ with integer
        frame0 = traj[0]
        np.testing.assert_array_equal(frame0['x'], [1.0, 4.0, 7.0, 10.0])
        
        # Test __getitem__ with slice
        frames = traj[0:2]
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(frames[0]['x'], [1.0, 4.0, 7.0, 10.0])
        np.testing.assert_array_equal(frames[1]['x'], [1.1, 4.1, 7.1, 10.1])
class TestLAMMPSTrajectoryEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for LAMMPSTrajectory."""
    
    def test_empty_file(self):
        """Test behavior with an empty file."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            empty_file = f.name
        
        try:
            with self.assertRaises(ValueError):
                LAMMPSTrajectory(empty_file)
        finally:
            os.remove(empty_file)
    
    def test_invalid_format(self):
        """Test behavior with file in invalid format."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("This is not a LAMMPS dump file\n")
            invalid_file = f.name
        
        try:
            with self.assertRaises(ValueError):
                LAMMPSTrajectory(invalid_file)
        finally:
            os.remove(invalid_file)
    
    def test_missing_property(self):
        """Test requesting a property that doesn't exist."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            # Create minimal valid dump file
            f.write("ITEM: TIMESTEP\n")
            f.write("1000\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write("1\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("ITEM: ATOMS id x y z\n")  # No 'type' column
            f.write("1 1.0 2.0 3.0\n")
            valid_file = f.name
        
        try:
            traj = LAMMPSTrajectory(valid_file)
            
            # Should raise error when requesting non-existent property
            with self.assertRaises(ValueError):
                traj.get_atom_types(0)
                
            with self.assertRaises(ValueError):
                traj.get_atom_property(0, 'vx')  # vx doesn't exist
                
        finally:
            os.remove(valid_file)

if __name__ == '__main__':
    unittest.main()