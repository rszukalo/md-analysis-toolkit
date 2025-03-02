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
        # Create a temporary mock LAMMPS dump file
        self.temp_file = self._create_mock_dump_file()
        
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary file
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def _create_mock_dump_file(self):
        """Create a mock LAMMPS dump file for testing."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            # Frame 1
            f.write("ITEM: TIMESTEP\n")
            f.write("1000\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write("4\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("0.0 10.0 0.0\n")
            f.write("0.0 10.0 0.0\n")
            f.write("0.0 10.0 0.0 \n")
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
            f.write("0.0 10.0 0.0\n")
            f.write("0.0 10.0 0.0\n")
            f.write("0.0 10.0 0.0 \n")
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
        
        # Check box dimensions
        self.assertEqual(traj.box_dims.shape, (2, 3, 2))  # (n_frames, 3 dimensions, min/max)
        np.testing.assert_array_equal(traj.box_dims[0][0], [0.0, 10.0])
    