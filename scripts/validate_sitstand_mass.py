"""CPU checks of the measured mass model and its registered task wiring.

Run: .venv/bin/python scripts/validate_sitstand_mass.py -v
"""

import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from mjlab_microduck.robot.xgoduck_mass_calibration import (
    SITSTAND_MASS_CALIBRATION,
    calibrate_sitstand_mass,
)


XML = SITSTAND_MASS_CALIBRATION.parent / "robot_allcollisions_backlash.xml"


class SitstandMassCalibrationTest(unittest.TestCase):
    def test_compiled_mass_inertia_and_geometry(self):
        baseline = mujoco.MjSpec.from_file(str(XML)).compile()
        calibrated = calibrate_sitstand_mass(mujoco.MjSpec.from_file(str(XML))).compile()
        config = json.loads(SITSTAND_MASS_CALIBRATION.read_text())
        head_ids = [calibrated.body(name).id for name in config["head_neck_bodies"]]
        self.assertAlmostEqual(float(calibrated.body_mass.sum()), .870)
        self.assertAlmostEqual(float(calibrated.body_mass[head_ids].sum()), .275)
        self.assertAlmostEqual(float(calibrated.body("trunk_base").mass[0]), .307)
        # Density scaling must preserve inertial axes, CoM, and all kinematics.
        for field in ("body_ipos", "body_iquat", "body_pos", "body_quat",
                      "geom_pos", "geom_quat", "geom_size", "geom_contype",
                      "geom_conaffinity", "jnt_pos", "jnt_axis", "jnt_range",
                      "site_pos", "site_quat", "qpos0", "actuator_trnid"):
            np.testing.assert_array_equal(getattr(baseline, field), getattr(calibrated, field))
        for body_id in range(1, calibrated.nbody):
            ratio = calibrated.body_mass[body_id] / baseline.body_mass[body_id]
            np.testing.assert_allclose(
                calibrated.body_inertia[body_id], baseline.body_inertia[body_id] * ratio,
                rtol=1e-10, atol=1e-15,
            )
            inertia = calibrated.body_inertia[body_id]
            self.assertTrue(np.all(inertia > 0))
            self.assertLessEqual(float(inertia.max()), float(inertia.sum() - inertia.max()) + 1e-12)
            if body_id not in head_ids and body_id != calibrated.body("trunk_base").id:
                self.assertEqual(calibrated.body_mass[body_id], baseline.body_mass[body_id])

    def test_repeated_application_and_diagonal_inertia(self):
        first = calibrate_sitstand_mass(mujoco.MjSpec.from_file(str(XML)))
        expected = first.compile()
        repeated = calibrate_sitstand_mass(first).compile()
        np.testing.assert_allclose(expected.body_mass, repeated.body_mass, rtol=1e-12)
        np.testing.assert_allclose(expected.body_inertia, repeated.body_inertia, rtol=1e-12)
        # Exercise diagonal tensors directly to avoid XML serialization rounding.
        diagonal = mujoco.MjSpec.from_file(str(XML))
        baseline = diagonal.compile()
        for body in diagonal.bodies:
            if body.mass > 0:
                body_id = baseline.body(body.name).id
                body.fullinertia = [float("nan")] * 6
                body.inertia = baseline.body_inertia[body_id]
                body.iquat = baseline.body_iquat[body_id]
        result = calibrate_sitstand_mass(diagonal).compile()
        np.testing.assert_allclose(expected.body_mass, result.body_mass, rtol=1e-12)
        np.testing.assert_allclose(expected.body_inertia, result.body_inertia, rtol=1e-10, atol=1e-15)

    def test_invalid_budget_fails_before_mutation(self):
        config = json.loads(SITSTAND_MASS_CALIBRATION.read_text())
        config["total_mass_kg"] = .300
        spec = mujoco.MjSpec.from_file(str(XML))
        before = spec.compile().body_mass.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "mass budget"):
                calibrate_sitstand_mass(spec, path)
        np.testing.assert_array_equal(before, spec.compile().body_mass)

    def test_registered_tasks_and_legacy_isolation(self):
        # Import mjlab first so its plugin loader initializes the task registry.
        from mjlab.tasks.registry import load_env_cfg

        for terrain in ("Flat", "Rough"):
            for play in (False, True):
                cfg = load_env_cfg(f"Mjlab-SitStand-{terrain}-XgoDuck", play=play)
                model = cfg.scene.entities["robot"].spec_fn().compile()
                self.assertAlmostEqual(float(model.body_mass.sum()), .870)
                self.assertAlmostEqual(sum(float(model.body(name).mass[0]) for name in
                                           ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")), .275)
        for task in ("Mjlab-Velocity-Flat-XgoDuck", "Mjlab-StandUp-Flat-XgoDuck",
                     "Mjlab-Roulade-Flat-XgoDuck"):
            model = load_env_cfg(task).scene.entities["robot"].spec_fn().compile()
            self.assertAlmostEqual(float(model.body_mass.sum()), .7909)


if __name__ == "__main__":
    unittest.main()
