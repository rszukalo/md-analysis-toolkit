import os
import unittest
import numpy as np
from pathlib import Path

from mdtoolkit.core.trajectory import LAMMPSTrajectory

class TestWaterTrajectory(unittest.TestCase):
    """Test LAMMPSTrajectory with real water simulation data."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test fixture once for all tests."""
        # Get the path to the water.dump file
        project_root = Path(__file__).resolve().parent.parent
        cls.dump_file = os.path.join(project_root, "test_data", "water.dump")
        
        # Ensure the file exists
        if not os.path.exists(cls.dump_file):
            raise FileNotFoundError(f"Test file not found: {cls.dump_file}")
        
        # Load the trajectory
        cls.traj = LAMMPSTrajectory(cls.dump_file)
    
    def test_file_structure(self):
        """Test that the water file structure is correctly parsed."""
        # Check basic structure
        self.assertGreater(self.traj.n_frames, 0, "Should have at least one frame")
        self.assertEqual(self.traj.n_atoms, 1500, "Should have 1500 atoms (500 water molecules)")
        
        # Check that required columns exist
        required_columns = ['id', 'type', 'x', 'y', 'z']
        for col in required_columns:
            self.assertIn(col, self.traj.atom_columns, f"Missing required column: {col}")
    
    def test_atom_types(self):
        """Test that atom types are correctly identified."""
        # Get atom types from the first frame
        types = self.traj.get_atom_types(0)
        
        # Count oxygen (type 1) and hydrogen (type 2) atoms
        n_oxygen = np.sum(types == 1)
        n_hydrogen = np.sum(types == 2)
        
        # Verify expected counts
        self.assertEqual(n_oxygen, 500, "Should have 500 oxygen atoms")
        self.assertEqual(n_hydrogen, 1000, "Should have 1000 hydrogen atoms")
    
    def test_frame_reading(self):
        """Test that frames can be read correctly."""
        # Read the first frame
        frame0 = self.traj.get_frame(0)
        
        # Check that the expected data is present
        self.assertEqual(len(frame0['id']), 1500, "First frame should have 1500 atoms")
        
        # Verify data types
        self.assertTrue(isinstance(frame0['x'], np.ndarray), "Coordinates should be numpy arrays")
        
        # Check positions are within the actual box bounds
        box_min_x = self.traj.box_dims[0][0][0]
        box_max_x = self.traj.box_dims[0][0][1]
        box_min_y = self.traj.box_dims[0][1][0]
        box_max_y = self.traj.box_dims[0][1][1]
        box_min_z = self.traj.box_dims[0][2][0]
        box_max_z = self.traj.box_dims[0][2][1]
        
        # Check that all coordinates are within box boundaries
        self.assertTrue(np.all(frame0['x'] >= box_min_x), "All x positions should be >= box_min_x")
        self.assertTrue(np.all(frame0['x'] <= box_max_x), "All x positions should be <= box_max_x")
        self.assertTrue(np.all(frame0['y'] >= box_min_y), "All y positions should be >= box_min_y")
        self.assertTrue(np.all(frame0['y'] <= box_max_y), "All y positions should be <= box_max_y")
        self.assertTrue(np.all(frame0['z'] >= box_min_z), "All z positions should be >= box_min_z")
        self.assertTrue(np.all(frame0['z'] <= box_max_z), "All z positions should be <= box_max_z")
    
    def test_multiple_frames(self):
        """Test that multiple frames can be read correctly."""
        # Only run this test if there are at least 2 frames
        if self.traj.n_frames < 2:
            self.skipTest("Trajectory has fewer than 2 frames")
        
        # Read two consecutive frames
        frame0 = self.traj.get_frame(0)
        frame1 = self.traj.get_frame(1)
        
        # Check that frames have the same structure
        self.assertEqual(len(frame0['id']), len(frame1['id']), "Frames should have same number of atoms")
        
        # Verify that positions are different (simulation has progressed)
        # We use almost equal with high tolerance since water molecules can move very little
        # between consecutive frames in some cases
        positions_identical = np.allclose(frame0['x'], frame1['x']) and \
                              np.allclose(frame0['y'], frame1['y']) and \
                              np.allclose(frame0['z'], frame1['z'])
        
        self.assertFalse(positions_identical, "Positions should change between frames")
    
    def test_atom_selection(self):
        """Test selecting atoms based on criteria."""
        # Select oxygen atoms
        oxygen_mask = self.traj.select_atoms(0, {'type': 1})
        self.assertEqual(sum(oxygen_mask), 500, "Should select 500 oxygen atoms")
        
        # Select hydrogen atoms
        hydrogen_mask = self.traj.select_atoms(0, {'type': 2})
        self.assertEqual(sum(hydrogen_mask), 1000, "Should select 1000 hydrogen atoms")
    
    def test_iterate_frames(self):
        """Test iterating through frames."""
        # Count frames using the iterator
        n_frames = 0
        for idx, frame in self.traj.iterate_frames():
            n_frames += 1
            
            # Verify each frame has the correct number of atoms
            self.assertEqual(len(frame['id']), 1500, f"Frame {idx} should have 1500 atoms")
        
        # Check that we got the expected number of frames
        self.assertEqual(n_frames, self.traj.n_frames, "Iterator should return all frames")
        
        # Test iteration with step=2 (if enough frames)
        if self.traj.n_frames >= 3:
            frames_step2 = []
            for idx, frame in self.traj.iterate_frames(step=2):
                frames_step2.append(idx)
            
            # Should only include frames with indices 0, 2, 4, etc.
            for idx in frames_step2:
                self.assertEqual(idx % 2, 0, f"With step=2, should only get even indices, got {idx}")
    
    def test_box_dimensions(self):
        """Test that box dimensions are read correctly."""
        # Check that box dimensions are present
        self.assertEqual(self.traj.box_dims.shape[0], self.traj.n_frames, "Box dimensions should exist for each frame")
        
        # Check that each dimension has min and max values
        self.assertGreaterEqual(self.traj.box_dims.shape[2], 2, 
                            "Each dimension should have at least min/max values")
        
    def test_get_atom_properties(self):
        """Test methods for getting atom properties."""
        # Test get_atom_ids
        ids = self.traj.get_atom_ids(0)
        self.assertEqual(len(ids), 1500, "Should get 1500 atom IDs")
        self.assertEqual(min(ids), 1, "First atom ID should be 1")
        self.assertEqual(max(ids), 1500, "Last atom ID should be 1500")
        
        # Test get_atom_property for positions
        x_coords = self.traj.get_atom_property(0, 'x')
        self.assertEqual(len(x_coords), 1500, "Should get 1500 x coordinates")
        
        # Test error case - request non-existent property
        non_existent_prop = 'non_existent_property'
        if non_existent_prop not in self.traj.atom_columns:
            with self.assertRaises(ValueError):
                self.traj.get_atom_property(0, non_existent_prop)


if __name__ == "__main__":
    unittest.main()