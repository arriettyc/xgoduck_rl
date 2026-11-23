"""Check a local Microduck SitStand bundle against the registered XgoDuck task.

No training or physics steps are performed. The environment is constructed and
reset only to resolve real observation dimensions and joint/action ordering.
Only safe weights-only checkpoint loading and safe YAML parsing are used.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from rsl_rl.models import MLPModel


def model_kwargs(cfg):
    return {key: deepcopy(getattr(cfg, key)) for key in
            ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path,
                        default=Path("reports/sitstand_checkpoint_compatibility.json"))
    args = parser.parse_args()
    bundle = json.loads((args.bundle_dir / "bundle.json").read_text())
    checkpoint = args.bundle_dir / bundle["checkpoint_file"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != bundle["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from bundle provenance")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source = yaml.safe_load((args.bundle_dir / "wandb_config.yaml").read_text())
    source_env = source["env_cfg"]["value"]
    source_agent = source["train_cfg"]["value"]
    source_onnx = args.bundle_dir / "alpha_sitstand.onnx"
    if hashlib.sha256(source_onnx.read_bytes()).hexdigest() != bundle["onnx"][0]["sha256"]:
        raise ValueError("Reference ONNX hash differs from bundle provenance")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(source_onnx), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map

    task = "Mjlab-SitStand-Flat-XgoDuck"
    target_env = load_env_cfg(task)
    target_agent = load_rl_cfg(task)
    # Resolve the actual interface without stepping physics or restoring a run.
    cfg = deepcopy(target_env)
    cfg.scene.num_envs = 2
    cfg.seed = 42
    cfg.curriculum = {}
    cfg.events = {name: term for name, term in cfg.events.items()
                  if name.startswith("reset_") or name in
                  ("set_ground_state", "expand_bam_friction_fields")}
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        obs, _ = env.reset()
        cpu_obs = TensorDict({key: value.cpu() for key, value in obs.items()}, batch_size=[2])
        actor = MLPModel(cpu_obs, target_agent.obs_groups, "actor", env.action_manager.total_action_dim,
                         **model_kwargs(target_agent.actor))
        critic = MLPModel(cpu_obs, target_agent.obs_groups, "critic", 1,
                          **model_kwargs(target_agent.critic))
        actor.load_state_dict(saved["actor_state_dict"], strict=True)
        critic.load_state_dict(saved["critic_state_dict"], strict=True)
        actor.eval()
        critic.eval()
        with torch.no_grad():
            actor_output = actor(cpu_obs)
            critic_output = critic(cpu_obs)
        if not (torch.isfinite(actor_output).all() and torch.isfinite(critic_output).all()):
            raise ValueError("Loaded networks produce nonfinite outputs")

        # Verify loaded actor + normalization reproduce its own exported policy.
        generator = np.random.default_rng(42)
        observations = generator.normal(0, .3, size=(256, 61)).astype(np.float32)
        observations[:, 3:6] = [0., 0., -1.]
        observations[:, 48:51] = 0.
        observations[:, 48] = generator.integers(0, 2, size=256)
        observations[:, 55:61] = 0.
        observations = np.concatenate([observations, cpu_obs["actor"].numpy()])
        with torch.no_grad():
            pytorch_actions = actor(TensorDict({"actor": torch.from_numpy(observations)},
                                               batch_size=[len(observations)])).numpy()
        reference_actions = np.concatenate([
            session.run(None, {session.get_inputs()[0].name: row[None]})[0]
            for row in observations])
        np.testing.assert_allclose(pytorch_actions, reference_actions, rtol=1e-4, atol=1e-5)

        # W&B YAML sorts mapping keys; never interpret that alphabetical order
        # as the original observation order. The pinned source uses mjlab 1.3's
        # base order, removes unused terms, and appends head/body command slots.
        if version("mjlab") != "1.3.0":
            raise ValueError("Reconstructing source critic order requires mjlab==1.3.0")
        base = make_velocity_env_cfg()
        layouts = {}
        for group in ("actor", "critic"):
            source_terms = source_env["observations"][group]["terms"]
            source_order = [name for name in base.observations[group].terms if name in source_terms]
            source_order.extend(name for name in ("head_command", "body_command")
                                if name not in source_order)
            if set(source_order) != set(source_terms):
                raise ValueError("Unexpected source observation terms; inspect pinned source")
            target_order = env.observation_manager.active_terms[group]
            if group == "actor":
                if source_order != metadata["observation_names"].split(","):
                    raise ValueError("Source reconstruction disagrees with exported actor order")
            layouts[group] = {
                "source_order": source_order, "target_order": target_order,
                "order_matches": source_order == target_order,
                "target_term_dimensions": env.observation_manager.group_obs_term_dim[group],
                "source_input_dimension": int(saved[group+"_state_dict"]["mlp.0.weight"].shape[1]),
                "target_input_dimension": int(cpu_obs[group].shape[1]),
                "terms": {name: {
                    "source_function": source_terms[name]["func"],
                    "target_function": (lambda f: f.__module__+"."+f.__name__)(
                        env.observation_manager.get_term_cfg(group, name).func),
                    "source_scale": source_terms[name].get("scale"),
                    "target_scale": (lambda s: s.cpu().tolist() if isinstance(s, torch.Tensor) else s)(
                        env.observation_manager.get_term_cfg(group, name).scale),
                } for name in source_order},
            }

        robot = env.scene["robot"]
        ids, _ = robot.find_joints(r"^(?!passive_).*")
        names = [robot.joint_names[index] for index in ids]
        source_names = metadata["joint_names"].split(",")
        action_ids = env.action_manager.get_term("joint_pos")._target_ids
        action_names = [robot.joint_names[index] for index in action_ids]
        observation_ids = env.observation_manager.get_term_cfg(
            "actor", "joint_pos").params["asset_cfg"].joint_ids
        observation_names = [robot.joint_names[index] for index in observation_ids]
        if not source_names == names == action_names == observation_names:
            raise ValueError("Source output, target action, and target joint-observation orders differ")
        source_home_patterns = source_env["scene"]["entities"]["robot"]["init_state"]["joint_pos"]
        source_home = []
        for name in source_names:
            matches = [value for pattern, value in source_home_patterns.items() if re.fullmatch(pattern, name)]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous source HOME for {name}")
            source_home.append(matches[0])
        target_home = robot.data.default_joint_pos[0, ids].cpu().numpy()
        report = {
            "source_bundle": bundle,
            "source_hub_revision": (args.bundle_dir / "hub_revision.txt").read_text().strip(),
            "checkpoint_sha256_verified": digest,
            "packages": {key: version(key) for key in ("mjlab", "rsl-rl-lib", "torch")},
            "strict_actor_load": True, "strict_critic_load": True,
            "actor_onnx_equivalence": {"samples": len(observations),
                "max_abs_error": float(np.max(np.abs(pytorch_actions-reference_actions))),
                "rtol": 1e-4, "atol": 1e-5},
            "observation_layouts": layouts,
            "critic_order_basis": "Pinned SitStand source + identical mjlab 1.3.0 base factory; YAML mappings are sorted.",
            "joints": {"source_order": source_names, "target_order": names,
                       "target_action_order": action_names,
                       "target_observation_order": observation_names,
                       "order_matches": source_names == names,
                       "home": [{"name": name, "source_rad": float(s), "target_rad": float(t),
                                 "target_minus_source_deg": float(np.rad2deg(t-s))}
                                for name, s, t in zip(names, source_home, target_home)]},
            "source_action": source_env["actions"]["joint_pos"],
            "target_action": {key: getattr(target_env.actions["joint_pos"], key) for key in
                              ("scale", "offset", "use_default_offset", "preserve_order", "clip")},
            "source_posture_command": source_env["commands"]["twist"],
            "target_posture_command": {key: getattr(target_env.commands["twist"], key) for key in
                                       ("sit_prob", "ramp_s", "sit_z", "stand_z")},
            "source_actuator": source_env["scene"]["entities"]["robot"]["articulation"]["actuators"][0],
            "target_actuator": asdict(target_env.scene.entities["robot"].articulation.actuators[0]),
            "target_actuator_identification": json.loads(Path(
                target_env.scene.entities["robot"].articulation.actuators[0].json_path).read_text()),
            "normalizer_counts": {name: int(saved[name+"_state_dict"]["obs_normalizer.count"])
                                  for name in ("actor", "critic")},
            "source_exploration_std": saved["actor_state_dict"]["distribution.std_param"].tolist(),
            "source_network_config": {key: source_agent[key] for key in ("actor", "critic", "algorithm")},
            "target_network_config": {key: asdict(getattr(target_agent, key)) for key in
                                      ("actor", "critic", "algorithm")},
            "control_hz": {"source": 1/(source_env["decimation"]*source_env["sim"]["mujoco"]["timestep"]),
                           "target": 1/env.step_dt},
            "source_saved_iteration": saved["iter"],
            "source_saved_curriculum_counter": saved.get("infos", {}).get("env_state"),
            "target_nominal_mass_kg": float(env.sim.mj_model.body_mass.sum()),
            "optimizer_state_present": "optimizer_state_dict" in saved,
            "limitations": ["Only construction/reset and network inference; no rollout, optimization, or hardware test.",
                            "Strict loading establishes tensor compatibility, not critic accuracy or physical transfer.",
                            "Source and target HOME and physics differ; optimizer/curriculum must be handled deliberately."],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str)+"\n")
        print(f"Strict actor/critic loading passed; ONNX max error {report['actor_onnx_equivalence']['max_abs_error']}")
        print(f"Report: {args.out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
