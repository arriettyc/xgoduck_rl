# XgoDuck SitStand training plan — 2026-11-20

## Objective

Train reliable stand → sit → stand transitions on the measured XgoDuck hardware, without falling backward or requiring sustained actuator overload. The default standing position currently holds on hardware; descent and ascent fail with the heavier head/rear assembly.

## Progress and next action

- [x] Review local SitStand configuration, robot physics, inference conventions, and upstream Microduck reference.
- [x] Record hardware measurements: **870 g total**, **275 g complete head + neck**, **100 g battery**.
- [x] Confirm that head geometry and battery mounting position are intended to match this repository's setup.
- [x] **1. Update and validate the nominal mass model** (provisional lumped trunk/battery allocation).
- [ ] **2. Evaluate the existing checkpoint on the calibrated model — NEXT.**
- [ ] 3. Establish a stable XgoDuck seated target and feasible transition.
- [ ] 4. Fine-tune nominal SitStand transitions.
- [ ] 5. Adjust transition rewards only where diagnostics show a conflict.
- [ ] 6. Add robustness and validate deployment on hardware.

Status: **the nominal mass update is applied and validated**. SitStand's loader now applies the measured targets after loading the legacy XML. The battery's separate inertial contribution remains unresolved and is explicitly represented through a provisional lumped trunk model. No training run or checkpoint behavior evaluation has been performed as part of this stage.

## 1. Update and validate the mass model

### Measurements and comparison

The final **275 g head + neck measurement supersedes the earlier 270 g head-only estimate**. Total robot mass includes the installed battery. Measurements are approximate.

| Quantity | Old simulator | Applied nominal calibration / measured target | Difference |
|---|---:|---:|---:|
| Complete robot | 790.9 g | 870 g | +79.1 g |
| Complete head + neck | 252.9 g | 275 g | +22.1 g (+8.74%) |
| Everything outside head + neck, including battery | 538.0 g | 595 g | +57.0 g |
| Battery | Existing contribution unresolved | 100 g | Do not add 100 g without checking existing inclusion |

Baseline model: `src/mjlab_microduck/robot/xgoduck/robot_allcollisions_backlash.xml`, selected by `get_sitstand_spec()` in `src/mjlab_microduck/robot/xgoduck_constants.py`.

### Applied head/neck allocation

The loader preserves the existing mass proportions, body-frame centers of mass, and geometry. Each head/neck body's mass and inertia tensor is multiplied by **275 / 252.9 ≈ 1.087386** for the current XML. This is an initial identification assumption; it does not establish the true component mass distribution or inertia.

| Model body | Old mass | Applied mass, approximately |
|---|---:|---:|
| `neck` | 45.312 g | 49.272 g |
| `neck_pitch` | 3.882 g | 4.221 g |
| `yaw_roll_motion` | 34.727 g | 37.761 g |
| `jaw_soft` | 168.979 g | 183.746 g |
| **Combined** | **252.900 g** | **275.000 g** |

The combined measured assembly is mapped to these four bodies for the initial calibration. Do not assign 275 g to `jaw_soft` alone.

### Trunk and battery allocation

The old trunk mass was **250 g**. After the head/neck correction alone, total mass would be **813 g**, leaving **57 g** to reach 870 g. The applied provisional calibration puts that residual into the trunk: **307 g**, with mass and inertia scaled by **1.228**. Trunk CoM remains `(-0.012276, -0.00001265566, -0.01955284246)` m in its body frame.

Inspection of the source CAD CSV, URDF, converter, and checked-in XML did not establish how much of the 100 g battery was already included. There is no separately named battery body in the reviewed model. Unchanged mounting position does not establish whether its mass is already represented correctly. No separate battery mass is added.

This is a **lumped residual-mass estimate**, keeping the other links unchanged. It matches total mass but does not identify the rear battery's contribution to trunk CoM and pitch inertia. Validate sensitivity to plausible rearward CoM/inertia changes before relying on it for transfer.

If representing the battery explicitly, subtract any existing battery contribution from the trunk before adding the measured battery at its mounting position. Combine inertias using the parallel-axis theorem where needed; avoid double counting.

### Implementation and completion checklist

- [x] Establish a single calibration source: `src/mjlab_microduck/robot/xgoduck/sitstand_mass_calibration.json`.
- [x] Set the combined head/neck mass to 275 g and scale its inertias consistently.
- [x] Explicitly document the provisional 307 g trunk/battery allocation; obtain 870 g total.
- [x] Reconcile generation logic through a post-load calibration: `convert_from_urdf.py` retains the 0.8 kg legacy baseline, while `get_sitstand_spec()` reapplies the measured group/total targets after loading generated XML. Regeneration cannot silently remove the calibration; different regenerated link masses will change the residual trunk budget and require revalidation.
- [x] Keep calibration scoped to SitStand; registered Walk, StandUp, and Roulade retain their 790.9 g baseline.
- [x] Compile registered Flat/Rough, train/play SitStand model factories and verify mass targets, valid inertias, and preserved geometry.
- [x] Record actual applied values and affected files here, then check off stage 1.

Implementation: `src/mjlab_microduck/robot/xgoduck_mass_calibration.py` reads the calibration file, scales full or diagonal inertia tensors with mass, and assigns the remaining mass to the trunk. `xgoduck_constants.py:get_sitstand_spec()` applies it. Shared XML is not rewritten; opening raw XML directly still shows baseline masses. The converter documentation and repository README explain this distinction.

Validation: `.venv/bin/python scripts/validate_sitstand_mass.py -v` (four checks passed) checks measured totals; preserved link/geom/joint/site geometry and inertial CoM/axes; positive principal inertias and triangle inequalities; unchanged non-calibrated link masses; repeat application; diagonal/full tensor handling; invalid mass-budget rejection; and registered task isolation. This validates the nominal model, not policy performance. Existing startup mass randomization still produces variations around the nominal calibrated model.

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
| 2026-11-20 | Applied and validated SitStand-only calibration: 870 g total, 275 g head/neck, provisional 307 g trunk; scaled inertias and retained geometry/CoM positions. | Stage 2: identify the tested checkpoint/runtime and evaluate both holds and both transitions on the calibrated model. |
