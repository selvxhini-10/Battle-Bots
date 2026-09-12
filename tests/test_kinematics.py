from pathlib import Path
import unittest

import numpy as np

from solemates.kinematics import KinematicModel


class KinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parents[1]
        urdf = root / "chopped_urdf_v2/chopped_urdf_v2/urdf/chopped_urdf_v2.urdf"
        cls.model = KinematicModel(urdf)

    def test_arm_chains_have_seven_independent_joints(self):
        self.assertEqual(len(self.model.movable("arm_base", "left_eef")), 7)
        self.assertEqual(len(self.model.movable("arm_base", "right_eef")), 7)

    def test_solver_can_return_to_a_known_reachable_point(self):
        known = {"lj0": -0.75, "lj1": 0.2, "lj2": -0.3, "lj3": 0.4}
        goal = self.model.forward("arm_base", "left_eef", known)[:3, 3]
        _solution, error, solved = self.model.solve_position("arm_base", "left_eef", goal)
        self.assertTrue(solved, f"IK error was {error}")
        self.assertLess(error, 0.01)


if __name__ == "__main__":
    unittest.main()
