"""Measured SitStand masses applied after loading the generated robot XML.

The CAD converter's 0.8 kg normalization remains the legacy baseline for other
tasks. This overlay sets absolute measured group targets each time SitStand
loads, including after XML regeneration; never bake it into the shared XML.
Battery mass is informational and already included in the measured total.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco


SITSTAND_MASS_CALIBRATION = (
    Path(__file__).parent / "xgoduck" / "sitstand_mass_calibration.json"
)


def _scale_body(body, mass: float) -> None:
    """Preserve CoM/inertial axes and scale the existing tensor with mass."""
    if not math.isfinite(body.mass) or body.mass <= 0:
        raise ValueError(f"Expected positive explicit mass for {body.name}")
    factor = mass / body.mass
    # MjSpec uses a NaN sentinel for unspecified fullinertia. Diagonal and
    # full tensors are alternative XML representations; preserve whichever
    # was provided rather than setting both.
    if all(math.isfinite(float(v)) for v in body.fullinertia):
        body.fullinertia = body.fullinertia * factor
    else:
        body.inertia = body.inertia * factor
    body.mass = mass


def calibrate_sitstand_mass(
    spec: mujoco.MjSpec,
    calibration_path: Path = SITSTAND_MASS_CALIBRATION,
) -> mujoco.MjSpec:
    """Apply measured head/neck and total mass targets in place.

    Other links retain their masses. The remaining mass is assigned to the
    trunk, with its existing CoM and density-scaled inertia. This provisional
    lumped model avoids inventing an additional battery or mounting position.
    Absolute targets make repeat application idempotent.
    """
    calibration = json.loads(calibration_path.read_text())
    total = float(calibration["total_mass_kg"])
    head_total = float(calibration["head_neck_mass_kg"])
    if not (math.isfinite(total) and math.isfinite(head_total) and 0 < head_total < total):
        raise ValueError("Expected 0 < head/neck mass < total mass")
    names = calibration["head_neck_bodies"]
    residual_name = calibration["residual_body"]
    if not names or len(set(names)) != len(names) or residual_name in names:
        raise ValueError("Head/neck and residual body selections must be distinct")
    bodies = {body.name: body for body in spec.bodies}
    for name in [*names, residual_name]:
        if name not in bodies:
            raise ValueError(f"Calibration body missing: {name}")
    # Source robot bodies supply explicit inertials. Reject unresolved mass
    # rather than silently budgeting geometry-derived mass as zero.
    for body in spec.bodies:
        if (not math.isfinite(body.mass) or body.mass < 0
                or (body.name != "world" and body.mass == 0 and len(body.geoms))):
            raise ValueError(f"Expected explicit nonnegative mass for {body.name}")
    head = [bodies[name] for name in names]
    if any(body.mass <= 0 for body in head) or bodies[residual_name].mass <= 0:
        raise ValueError("Calibrated bodies must have positive explicit masses")
    current_head = sum(body.mass for body in head)
    fixed_mass = sum(
        body.mass for body in spec.bodies
        if body.name not in names and body.name != residual_name
    )
    residual_mass = total - head_total - fixed_mass
    if residual_mass <= 0:
        raise ValueError("Measured total leaves no positive mass budget for the trunk")
    for body in head:
        _scale_body(body, body.mass * head_total / current_head)
    _scale_body(bodies[residual_name], residual_mass)
    return spec
