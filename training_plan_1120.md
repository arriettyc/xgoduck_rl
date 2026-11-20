# XgoDuck SitStand training plan — 2026-11-20

## Objective

Train reliable stand → sit → stand transitions on the measured XgoDuck hardware, without falling backward or requiring sustained actuator overload. The default standing position currently holds on hardware; descent and ascent fail with the heavier head/rear assembly.

## Progress and next action

- [x] Review local SitStand configuration, robot physics, inference conventions, and upstream Microduck reference.
- [x] Record hardware measurements: **870 g total**, **275 g complete head + neck**, **100 g battery**.
- [x] Confirm that head geometry and battery mounting position are intended to match this repository's setup.
- [ ] **1. Update and validate the mass model — NEXT.**
- [ ] 2. Evaluate the existing checkpoint on the calibrated model.
- [ ] 3. Establish a stable XgoDuck seated target and feasible transition.
- [ ] 4. Fine-tune nominal SitStand transitions.
- [ ] 5. Adjust transition rewards only where diagnostics show a conflict.
- [ ] 6. Add robustness and validate deployment on hardware.

Status verified from repository files on 2026-11-20: **the mass update has not been applied**. The values below are a baseline and proposed calibration, not completed changes. No training run has been launched as part of this plan.

## 1. Update and validate the mass model

### Measurements and comparison

The final **275 g head + neck measurement supersedes the earlier 270 g head-only estimate**. Total robot mass includes the installed battery. Measurements are approximate.

| Quantity | Old/current simulator | Measured target | Difference |
|---|---:|---:|---:|
| Complete robot | 790.9 g | 870 g | +79.1 g |
| Complete head + neck | 252.9 g | 275 g | +22.1 g (+8.74%) |
| Everything outside head + neck, including battery | 538.0 g | 595 g | +57.0 g |
| Battery | Existing contribution unresolved | 100 g | Do not add 100 g without checking existing inclusion |

Baseline model: `src/mjlab_microduck/robot/xgoduck/robot_allcollisions_backlash.xml`, selected by `get_sitstand_spec()` in `src/mjlab_microduck/robot/xgoduck_constants.py`.

### Proposed head/neck allocation

Initially preserve the existing mass proportions, body-frame centers of mass, and geometry. Multiply each head/neck body's mass and inertia tensor by **275 / 252.9 ≈ 1.087386**. This is an initial identification assumption; it does not establish the true component mass distribution or inertia.

| Model body | Old mass | Proposed mass, approximately |
|---|---:|---:|
| `neck` | 45.312 g | 49.272 g |
| `neck_pitch` | 3.882 g | 4.221 g |
| `yaw_roll_motion` | 34.727 g | 37.761 g |
| `jaw_soft` | 168.979 g | 183.746 g |
| **Combined** | **252.900 g** | **275.000 g** |

The combined measured assembly is mapped to these four bodies for the initial calibration. Do not assign 275 g to `jaw_soft` alone.

### Trunk and battery allocation

The current trunk mass is **250 g**. After the head/neck correction, total mass would be **813 g**, leaving **57 g** to reach 870 g.

Inspect the source CAD/URDF and trunk inertial assumptions to determine how much of the 100 g battery is already included. There is no separately named battery body in the reviewed model. Unchanged mounting position does not establish whether its mass is already represented correctly.

If no better component information is available, a provisional model can keep other links unchanged and increase the trunk from **250 g to 307 g**. Clearly label this as a lumped residual-mass estimate. It matches total mass but does not identify the rear battery's contribution to trunk CoM and pitch inertia. Validate sensitivity to plausible rearward CoM/inertia changes before relying on it for transfer.

If representing the battery explicitly, subtract any existing battery contribution from the trunk before adding the measured battery at its mounting position. Combine inertias using the parallel-axis theorem where needed; avoid double counting.

### Implementation and completion checklist

- [ ] Establish a single calibration source for the SitStand mass/inertia values.
- [ ] Set the combined head/neck mass to 275 g and scale its inertias consistently.
- [ ] Resolve or explicitly document the provisional trunk/battery allocation; obtain 870 g total.
- [ ] Reconcile generation logic: `convert_from_urdf.py` currently normalizes total mass to 0.8 kg and has trunk assumptions differing from the checked-in model. Regeneration must preserve the chosen calibration.
- [ ] Keep unrelated task physics changes out of the initial SitStand experiment; document any intentionally shared model changes.
- [ ] Compile the exact model used by the task and verify total mass, component masses, valid inertias, and unchanged geometry.
- [ ] Record actual applied values and affected files here, then check off stage 1.

## 2. Evaluate the existing checkpoint on the calibrated model

Identify the `.pt` checkpoint, exported ONNX, and actual hardware runtime revision. Keep an unchanged baseline checkpoint for comparisons.

Test standing hold, descent, seated hold, and ascent separately. Record completion time, backward pitch and pitch rate, joint targets versus encoder positions, supporting contacts, and actuator effort. On hardware, also record voltage under load and available current/load telemetry.

Check whether the corrected simulator reproduces the observed backward failure. Good joint tracking followed by tipping suggests balance/contact problems; large tracking error before tipping suggests actuator limitations or runtime mismatch.

Before using `scripts/infer_policy.py` as a deployment rehearsal, correct its Microduck defaults for XgoDuck: model, HOME offsets, and actuator assumptions. Verify the actual hardware runtime independently:

- 50 Hz policy execution and action scale 1.0.
- 61D actor observations with correct joint order/signs, gravity convention, and previous-action semantics.
- Posture flag 0 = stand and 1 = sit; initially zero head/body commands.
- HOME offsets consistent with the XgoDuck training model.
- Firmware gains, voltage, delay, and action filtering consistent with training assumptions.
- ONNX export includes observation normalization.

**Completion gate:** a reproducible baseline for all four behaviors, with failure mode and runtime consistency documented.

## 3. Establish a stable seated target and feasible transition

The task currently retains Microduck's seated joint targets while retargeting trunk heights. Validate these against the calibrated XgoDuck model rather than assuming the inherited pose transfers.

Sweep nearby hip pitch, knee, ankle, and modest head/neck configurations using HLS1910 dynamics. Evaluate settled tilt and pitch rate, supporting contacts, actuator effort, and ability to initiate standing. Other body contacts may legitimately support sitting; do not require CoM over the feet throughout the entire motion.

A prior kinematic check found modeled CoM about 13 mm behind the rear-most foot mesh at the inherited seated target. This is a diagnostic clue, not proof of instability: it did not evaluate dynamic settling or the full active support polygon.

Investigate weight transfer before deep folding on descent and before leg extension on ascent. Do not prescribe a fixed lean direction or magnitude without checking the calibrated model.

After selecting a pose, remeasure settled seated height and update pose rewards, composite rewards, reset targets, and command height targets consistently.

**Completion gate:** stable seated hold under modest reset noise and a feasible rise within modeled actuator limits.

## 4. Fine-tune nominal SitStand transitions

Use an isolated XgoDuck SitStand experiment. Warm-start from the existing XgoDuck SitStand checkpoint, preserving actor/value weights and observation normalizers. Reset optimizer and training/curriculum counters deliberately. Ordinary resume restores optimizer state and curriculum progress and can immediately activate mature disturbance and penalty stages.

| Setting | Initial experiment |
|---|---|
| Terrain | Flat |
| Head commands | HOME; preserve command observation slots |
| Push disturbances | Disabled |
| Mass/CoM randomization | Narrow, centered on calibrated values |
| PPO learning rate | 1e-4, initially constant; compare 5e-5 if unstable |
| Entropy coefficient | Initial proposal 0.003; adjust from exploration behavior |
| Target ramp | Compare existing 2 s with 3 s; lengthen command dwell accordingly |
| Reset/command coverage | Both transition directions and both resting postures |
| Evaluation interval | Every 100–250 iterations |

Run a 64-environment, 5-iteration configuration smoke test before a longer run. Determine run length from transition success, not a preset iteration budget. Slowing an unstable transition alone does not fix balance.

**Completion gate:** reliable nominal descent and ascent, stable endpoint holds, and no persistent excessive actuator effort.

## 5. Adjust transition rewards only where necessary

Start with corrected mass and pose targets while preserving existing reward signs. Change reward structure only when diagnostics identify a conflict:

- Reduce joint-pose tracking pressure during transitions if interpolation prevents the necessary weight shift; retain endpoint accuracy.
- Consider direction-specific references if ascent and descent need different motion paths.
- Allow useful transient forward inclination while rewarding quiet, upright endpoints.
- Keep rise-speed and torque-rate penalties light until ascent is reliable; introduce stronger smoothing afterward.
- Add phase-aware backward-tip pressure if rearward collapse remains an exploit.
- Verify that penalty reward contributions have the intended sign.

Compare each change against the calibrated nominal baseline using separate descent/ascent metrics.

## 6. Add robustness and validate hardware

Widen physically consistent head and trunk/battery uncertainty around the calibrated values, then add measured voltage/delay variation, mild leaned-back seated starts, small backward disturbances, and head movement. Initial sensitivity sweeps may include head/neck assembly masses around 250–290 g; this is an exploratory range, not a measurement confidence interval. Evaluate mass combinations with explicit, consistent total-mass accounting.

Adapt current upstream Microduck's seated backward-tip events, head-command disturbances, and leaned-back neck-strain penalty only after nominal transitions work. Retune disturbance magnitudes and torque scales for HLS1910 hardware rather than copying XL330 values directly.

Reference: https://github.com/pollen-robotics/microduck_rl/blob/develop/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py

Measure stand hold, descent, sit hold, and ascent separately. Select checkpoints by success rate, endpoint tilt, transition time, tracking error, and actuator effort rather than aggregate reward alone.

Proposed acceptance targets:

- At least 99% success in each direction across 1,000 randomized simulation trials, with evaluation conditions recorded.
- Exported ONNX behavior consistent with checkpoint evaluation.
- Twenty consecutive supported hardware sit↔stand cycles without backward falls or sustained overload.

## Completion log

| Date | Completed work | Next action |
|---|---|---|
| 2026-11-20 | Reviewed configuration and upstream reference; recorded final mass measurements; verified mass calibration is still pending; created this plan. | Apply and validate stage 1 mass calibration, documenting trunk/battery assumptions. |
