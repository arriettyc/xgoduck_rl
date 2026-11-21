"""Evaluate a SitStand ONNX on paired legacy/calibrated training physics.

Example:
  .venv/bin/python scripts/evaluate_sitstand.py --onnx /path/to/xgoduck_sitstand.onnx

Uses the registered task's BAM/backlash physics and actor observations. This
is a simulation baseline and an action-filter ablation, not a full replay of
the MCU, IMU estimator, runtime safety controller, or policy handoff.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.robot.xgoduck_constants import get_standup_spec
from mjlab_microduck.tasks import mdp


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("logs/evaluation/sitstand_1120"))
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segment-s", type=float, default=6.0)
    parser.add_argument("--walking-onnx", type=Path,
                        help="Optionally switch stand commands to this walking policy after 1 s")
    parser.add_argument("--stand-handoff-s", type=float, default=1.)
    parser.add_argument("--masses", nargs="+", choices=("legacy", "calibrated"),
                        default=["legacy", "calibrated"])
    parser.add_argument("--filters", nargs="+", choices=("none", "runtime"),
                        default=["none", "runtime"])
    return parser.parse_args()


class OnnxPolicy:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        if self.input.shape != [1, 61] or self.output.shape != [1, 14]:
            raise ValueError("Expected fixed-batch SitStand ONNX: [1,61] → [1,14]")
        self.metadata = self.session.get_modelmeta().custom_metadata_map

    def __call__(self, observations):
        # Preserve the original ONNX graph and execute its declared batch size.
        observations = np.asarray(observations, dtype=np.float32)
        return np.concatenate([
            self.session.run([self.output.name], {self.input.name: row[None]})[0]
            for row in observations
        ])


def configure(args, mass):
    cfg = load_env_cfg("Mjlab-SitStand-Flat-XgoDuck", play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 2 + 2 * args.segment_s + 10
    cfg.auto_reset = False
    cfg.curriculum = {}
    cfg.terminations = {}  # Track nonfinite states explicitly; never auto-reset falls.
    keep = {"reset_scene_to_default", "reset_base", "reset_robot_joints",
            "reset_passive_joints", "reset_backlash_joints", "set_ground_state",
            "reset_action_history", "expand_bam_friction_fields", "foot_friction"}
    cfg.events = {name: term for name, term in cfg.events.items() if name in keep}
    cfg.events["foot_friction"].params["ranges"] = (.5, .5)
    cfg.events["reset_robot_joints"].params.update(
        position_range=(-.01, .01), velocity_range=(0., 0.))
    cfg.events["set_ground_state"].params.update(
        sitting_joint_noise_std=.01, sitting_tilt_max=math.radians(2),
        sitting_z_min=.085868, sitting_z_max=.085868,
        standing_z_min=.139158, standing_z_max=.139158)
    robot = cfg.scene.entities["robot"]
    if mass == "legacy":
        robot.spec_fn = get_standup_spec  # Same full-contact XML, no mass overlay.
    robot.articulation.actuators = tuple(replace(
        actuator, vin_range=(7.7, 7.7), vin_drop_gain_range=(.1, .1),
        delay_min_lag=3, delay_max_lag=3) for actuator in robot.articulation.actuators)
    cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
    cfg.commands["head_pose"].resampling_time_range = (1e9, 1e9)
    cfg.commands["head_pose"].ranges = ((0., 0.),) * 4
    cfg.commands["head_pose"].zero_command_prob = 1.
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            if term and "max_angle_deg" in term.params:
                term.params["max_angle_deg"] = 0.
    return cfg


def summarize(trace, target, start, stop, dt, eligible):
    z = trace["height_m"][start:stop]
    tilt = trace["tilt_deg"][start:stop]
    pitch = trace["pitch_deg"][start:stop]
    speed = trace["speed_m_s"][start:stop]
    finite = trace["finite"][start:stop].astype(bool)
    goal = ((np.abs(z - target) <= .012) & (tilt <= 15.) & (speed <= .05) & finite)
    window = max(1, round(.5 / dt))
    arrived = np.zeros(goal.shape[1], dtype=bool)
    arrival = np.full(goal.shape[1], np.nan)
    for end in range(window, len(goal) + 1):
        good = goal[end-window:end].all(axis=0)
        newly = good & ~arrived
        arrival[newly] = end * dt
        arrived |= good
    tail = goal[-min(len(goal), round(1 / dt)):].mean(axis=0)
    backward_fall = (pitch < -45.).any(axis=0)
    any_fall = (tilt > 60.).any(axis=0)
    success = arrived & (tail >= .8) & ~any_fall & ~backward_fall & finite.all(axis=0) & eligible
    vertical_speed = np.diff(z, axis=0) / dt
    pitch_rate = np.diff(np.unwrap(np.deg2rad(pitch), axis=0), axis=0) / dt
    return {
        "n": int(goal.shape[1]), "eligible_n": int(eligible.sum()),
        "success_n": int(success.sum()), "success_rate": float(success.mean()),
        "success_rate_given_eligible": float(success[eligible].mean()) if eligible.any() else None,
        "arrived_n": int(arrived.sum()),
        "median_arrival_s": float(np.nanmedian(arrival)) if arrived.any() else None,
        "arrived_by_1s_n": int(np.sum(np.isfinite(arrival) & (arrival <= 1. + 1e-6))),
        "goal_at_1s_n": int(goal[min(len(goal)-1, max(0, round(1/dt)-1))].sum()),
        "backward_fall_n": int(backward_fall.sum()), "any_fall_n": int(any_fall.sum()),
        "peak_backward_pitch_deg": float(np.maximum(-pitch, 0).max()),
        "final_median_height_m": float(np.median(z[-1])),
        "final_median_pitch_deg": float(np.median(pitch[-1])),
        "peak_abs_torque_nm": float(trace["torque_nm"][start:stop].max()),
        "peak_descent_speed_m_s": float(np.maximum(-vertical_speed, 0).max()),
        "peak_rise_speed_m_s": float(np.maximum(vertical_speed, 0).max()),
        "peak_abs_pitch_rate_rad_s": float(np.abs(pitch_rate).max()),
        "mean_joint_tracking_rmse_rad": float(trace["tracking_rmse_rad"][start:stop].mean()),
        "support_contacts": {name: float(trace[name][start:stop].mean()) for name in
                             ("feet_contact_n", "head_contact_n", "trunk_contact_n")},
        "nonfinite_n": int((~finite).any(axis=0).sum()),
    }, goal[-window:].all(axis=0)


def run_cycle(env, policy, initial, filter_mode, duration, seed, output,
              walking_policy=None, handoff_s=1.):
    n, dt = env.num_envs, env.step_dt
    params = env.event_manager.get_term_cfg("set_ground_state").params
    params.update(sitting_prob=float(initial == "sit"), standing_prob=float(initial == "stand"))
    observations, _ = env.reset(seed=seed)
    command = env.command_manager.get_term("twist")
    robot = env.scene["robot"]
    servo_ids = mdp._servo_joint_ids(env, robot)
    servo_cfg = SceneEntityCfg("robot", joint_ids=servo_ids)
    sim_home = robot.data.default_joint_pos[:, servo_ids].cpu().numpy()
    # Runtime v0.3.6 uses HOME_DEG rather than rounded ONNX metadata.
    runtime_home = np.deg2rad(np.array(
        [0, -5, -22, 0, 22, 20, 20, 0, 0, 0, 5, 22, 0, -22], dtype=np.float32))
    home = runtime_home if filter_mode == "runtime" else sim_home
    alphas = np.zeros((1, 14), dtype=np.float32)
    if filter_mode == "runtime":
        alphas[:] = .35
        alphas[:, 5:9] = .45
    previous = np.zeros((n, 14), dtype=np.float32)
    filtered = previous.copy()
    lengths = [round(2 / dt), round(duration / dt), round(duration / dt)]
    flags = [int(initial == "sit"), int(initial != "sit"), int(initial == "sit")]
    telemetry = {key: [] for key in ("height_m", "pitch_deg", "tilt_deg", "speed_m_s",
                  "torque_nm", "tracking_rmse_rad", "finite", "joint_pos_rad",
                  "joint_target_rad", "raw_action", "feet_contact_n", "head_contact_n",
                  "trunk_contact_n")}
    contact_ids = {}
    model = env.sim.mj_model
    terrain_ids = [i for i in range(model.ngeom) if model.geom(i).name.endswith("terrain")]
    if not terrain_ids:
        raise ValueError("No terrain geometry found for support-contact telemetry")
    for name, pattern in (("feet_contact_n", ("left_foot_collision", "right_foot_collision")),
                          ("head_contact_n", ("head_collision",)),
                          ("trunk_contact_n", ("trunk_collision",))):
        contact_ids[name] = [i for i in range(model.ngeom)
                             if any(model.geom(i).name.endswith(p) for p in pattern)]
    for segment, (steps, flag) in enumerate(zip(lengths, flags)):
        # Runtime resets the selected policy's action history on a switch.
        if walking_policy is not None and segment:
            previous.fill(0.)
            filtered.fill(0.)
        command.vel_command_b[:] = 0.
        command.vel_command_b[:, 0] = flag
        for step in range(steps):
            handoff = (walking_policy is not None and segment > 0 and flag == 0
                       and step >= round(handoff_s / dt))
            if handoff and step == round(handoff_s / dt):
                previous.fill(0.)
                filtered.fill(0.)
            obs = env.observation_manager.compute()["actor"].cpu().numpy().copy()
            if obs.shape != (n, 61):
                raise ValueError(f"Unexpected actor observation shape {obs.shape}")
            obs[:, 6:20] += sim_home - home
            obs[:, 34:48] = previous  # Runtime remembers raw, unfiltered output.
            obs[:, 48:51] = [flag, 0., 0.]  # Command flips precede cached observation refresh.
            walking_idle = walking_policy is not None and segment == 0 and initial == "stand"
            raw = (walking_policy if handoff or walking_idle else policy)(obs)
            if not np.isfinite(raw).all():
                raise RuntimeError("Nonfinite policy output")
            previous = raw.copy()
            filtered = alphas * filtered + (1 - alphas) * raw
            applied = filtered + home - sim_home
            with torch.no_grad():
                observations, _, _, _, _ = env.step(torch.as_tensor(applied, device=env.device))
            gravity = robot.data.projected_gravity_b.cpu().numpy()
            pitch = np.rad2deg(np.arctan2(gravity[:, 0], -gravity[:, 2]))
            tilt = np.rad2deg(np.arccos(np.clip(-gravity[:, 2], -1, 1)))
            encoder = mdp.joint_pos_rel_backlash(env, asset_cfg=servo_cfg).cpu().numpy() + sim_home
            targets = robot.data.joint_pos_target[:, servo_ids].cpu().numpy()
            values = {
                "height_m": (robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]).cpu().numpy(),
                "pitch_deg": pitch, "tilt_deg": tilt,
                "speed_m_s": robot.data.root_link_lin_vel_w.norm(dim=1).cpu().numpy(),
                "torque_nm": robot.data.actuator_force.abs().amax(dim=1).cpu().numpy(),
                "tracking_rmse_rad": np.sqrt(np.mean((targets-encoder)**2, axis=1)),
                "finite": np.isfinite(encoder).all(axis=1) & np.isfinite(pitch),
                "joint_pos_rad": encoder, "joint_target_rad": targets, "raw_action": raw,
            }
            # Count active floor contact points, not the number of touching feet.
            contacts = env.sim.data.contact
            contact_count = int(env.sim.data.nacon[0].item())
            geom = contacts.geom[:contact_count].cpu().numpy()
            distances = contacts.dist[:contact_count].cpu().numpy()
            margins = contacts.includemargin[:contact_count].cpu().numpy()
            worlds = contacts.worldid[:contact_count].cpu().numpy()
            for key, ids in contact_ids.items():
                active = ((np.isin(geom[:, 0], ids) | np.isin(geom[:, 1], ids))
                          & (distances <= margins) & (worlds >= 0) & (worlds < n)
                          & (geom >= 0).all(axis=1))
                # Require the other geom to belong to the world (terrain).
                terrain = np.isin(geom[:, 0], terrain_ids) | np.isin(geom[:, 1], terrain_ids)
                values[key] = np.bincount(worlds[active & terrain].astype(int), minlength=n)[:n]
            for key, value in values.items():
                telemetry[key].append(value.copy())
            if not values["finite"].all():
                raise RuntimeError("Nonfinite simulated state; aborting without reset")
        print(f"  {initial} cycle segment {segment}: flag={flag}, {steps} steps", flush=True)
    trace = {key: np.stack(value) for key, value in telemetry.items()}
    np.savez_compressed(output, **trace)
    segments = {}
    start = 0
    eligible = np.ones(n, dtype=bool)
    for length, flag, label in zip(lengths, flags, ("initial_hold", "transition_1", "transition_2")):
        target = .085868 if flag else .139158
        skip = round(.5/dt) if label == "initial_hold" else 0
        segments[label], eligible = summarize(trace, target, start+skip, start+length, dt, eligible)
        segments[label]["target"] = "sit" if flag else "stand"
        start += length
    return segments


def main():
    args = arguments()
    if args.num_envs < 1 or args.segment_s < 1:
        raise ValueError("Need positive environment count and at least 1 s per segment")
    args.out.mkdir(parents=True, exist_ok=True)
    policy = OnnxPolicy(args.onnx)
    walking_policy = OnnxPolicy(args.walking_onnx) if args.walking_onnx else None
    if walking_policy and args.filters != ["runtime"]:
        raise ValueError("Handoff evaluation requires --filters runtime")
    report = {
        "policy_path": str(args.onnx.resolve()),
        "policy_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "policy_metadata": policy.metadata,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "settings": {"num_envs": args.num_envs, "seed": args.seed, "segment_s": args.segment_s,
                     "foot_friction": .5, "voltage_v": 7.7, "sag_gain": .1,
                     "actuator_delay_ms": 15, "reset_joint_noise_rad": .01,
                     "reset_tilt_deg": 2, "head_command": "HOME", "pushes": False,
                     "observation_noise": False, "mass_com_randomization": False},
        "limitations": ["Ideal projected gravity; training observation delays retained.",
                        "Runtime mode applies action EMA and HOME only; no full IMU estimator, MCU safety, or handoff.",
                        "One-second arrival metrics do not simulate the runtime handoff to walking.",
                        "Nominal seeded reset variation is not a robustness certification."],
        "experiments": {},
    }
    if walking_policy:
        report["walking_policy"] = {
            "path": str(args.walking_onnx.resolve()),
            "sha256": hashlib.sha256(args.walking_onnx.read_bytes()).hexdigest(),
            "metadata": walking_policy.metadata, "stand_handoff_s": args.stand_handoff_s,
        }
        report["limitations"][1:3] = [
            "Runtime HOME, action EMA, history reset, and stand-to-walk timeout reproduced; no MCU safety, IMU estimator, or 0.2 s action-preparation hold."
        ]
    for mass in args.masses:
        for filter_mode in args.filters:
            key = f"{mass}_{filter_mode}"
            print(f"Starting {key}", flush=True)
            env = ManagerBasedRlEnv(cfg=configure(args, mass), device=args.device)
            try:
                model = env.sim.mj_model
                report["experiments"][key] = {"nominal_mass_kg": float(model.body_mass.sum())}
                for initial in ("stand", "sit"):
                    report["experiments"][key][initial + "_cycle"] = run_cycle(
                        env, policy, initial, filter_mode, args.segment_s, args.seed,
                        args.out / f"{key}_{initial}.npz", walking_policy, args.stand_handoff_s)
                    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            finally:
                env.close()
    print(json.dumps(report["experiments"], indent=2))
    print(f"Saved report: {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
