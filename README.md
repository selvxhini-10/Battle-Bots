# Solemates: Sock Matchmaker with BracketBot :)

Solemates finds matching socks in an overhead image, plans a two-arm reunion, validates every Cartesian motion against the supplied BracketBot URDF, and escorts unmatched socks to the **Singles Club**.

## Run it

Python 3.10+, NumPy, and OpenCV are required.

```bash
cd /Users/raiyaminhas/Projects/Battle-Bots
python3 -m pip install -e .
./run_demo.sh
```

The launcher opens the annotated result in the default macOS image viewer. Use `./run_demo.sh --headless` to generate artifacts without opening a window.
Results are written to `demo_output/`:

- `scene.png`: synthetic camera frame
- `matches.png`: detected socks, grasp points, pairs, confidence, and singleton
- `run.json`: observations, all pair scores, accepted matches, IK solutions, and the full event trace

Other modes:

```bash
python3 -m solemates --image path/to/table.jpg --analyze-only
python3 -m solemates --camera 0 --analyze-only
python3 -m pip install -e '.[viewer]'
python3 -m solemates --synthetic --rrd
python3 -m unittest discover -s tests -v
```

## Implemented pipeline

```text
Synthetic scene, image, or camera
              |
              v
Background segmentation (OpenCV)
              |
              v
Color + texture + size + shape features
              |
              v
Globally optimal non-overlapping pairing
              |
              v
Grasp and dual-arm task planning
              |
              v
URDF FK + damped-least-squares position IK
              |
              v
Safe simulated backend + JSON/Rerun logging
```

The built-in scene contains two intended pairs and one singleton. The current test run detects all five, matches both pairs at roughly 93%, and IK-validates 33 arm movements.

## Matching

The detector assumes separated socks on the configured plain background. Each connected component becomes a `SockObservation` with:

- mask, bounding box, centroid, and area
- principal orientation from PCA
- interior grasp point from a distance transform
- 2D HSV color histogram
- Laplacian texture statistics
- compactness, extent, and aspect-ratio shape features

Pair confidence is:

```text
0.50 * color + 0.25 * pattern + 0.15 * size + 0.10 * shape
```

Dynamic programming chooses the maximum-total-score set of non-overlapping pairs. Candidates below `match_threshold` remain single. We do not greedily accept the first good-looking match.

Tune the background and matching settings in `config.json` for the real table and socks.

## Coordinates and grasping

The MVP linearly maps the camera image rectangle onto `table_bounds_m`. This is correct for the synthetic scene but only an approximation for a real camera. Before physical execution, replace it with a calibrated planar homography from at least four known table points.

For each sock, the planner creates:

```text
pregrasp -> grasp -> close -> vertical lift -> transport -> place -> open -> retreat
```

The gripper is oriented across the sock's principal fabric direction. If bare cloth is unreliable, a removable felt tab inside each cuff is a reasonable, documented grasp affordance.

## Kinematics

`solemates/kinematics.py` parses the supplied URDF directly. It supports fixed, revolute, continuous, and prismatic joints; calculates forward kinematics from `arm_base` to either end effector; and solves position targets with numerical damped-least-squares IK while clamping every joint to its URDF limits.

The solver covers seven independent joints per arm (`j0` through `j6`). Gripper mimic joints are not independent IK variables.

This lightweight solver is suitable for reachability checks and hackathon simulation. Physical control should use the organizers' supported controller and, if appropriate, constrained QP IK with velocity, collision, and hardware limits.

## State machine

```text
SCAN -> MATCH -> PICK_FIRST -> PICK_SECOND -> REUNITE -> PLACE_PAIR
                                                           |
                                                           v
                                                        VERIFY

remaining singleton -> HANDLE_SINGLE -> VERIFY
no remaining socks  -> CELEBRATE -> DONE
```

The simulated backend validates workspace bounds, solves IK for every move, tracks which sock each arm holds, and rejects impossible release/attachment sequences. Its verification events are task-state checks, not visual confirmation of cloth motion. The current launcher shows the perception result; it does not animate the 3D robot or simulate cloth physics.

Real closed-loop verification still needs to rescan after each pickup and placement and confirm that the sock moved from the source region to the destination. Retry with a new grasp point, then request human help after `max_retries`.

## Project structure

```text
config.json                 thresholds, workspace, destinations, backend
run_demo.sh                 synthetic demo launcher
solemates/
  app.py                    CLI and state-machine orchestration
  kinematics.py             URDF parser, FK, and numerical IK
  matching.py               scoring and global assignment
  models.py                 typed observations, poses, plans, events
  perception.py             segmentation and feature extraction
  planning.py               pixel mapping and grasp/destination poses
  robot.py                  simulated backend and guarded hardware adapter
  synthetic.py              deterministic five-sock scene
  visualization.py          annotated PNG and optional Rerun output
tests/
  test_kinematics.py
  test_matching.py
  test_pipeline.py
chopped_urdf_v2/            supplied self-contained robot model/viewer
```

## Hardware boundary

`BracketBotRobot` intentionally raises an error. A URDF describes the robot but cannot command motors. Before removing that guard, obtain and implement:

- supported joint or end-effector command API
- live joint feedback and units
- gripper commands and grasp feedback
- controller rate and trajectory format
- velocity/acceleration limits
- collision and workspace constraints
- watchdog, fault handling, and emergency stop

The hardware adapter must preserve this task-level interface:

```python
robot.move_end_effector(arm, pose)
robot.set_gripper(arm, closed)
robot.stop()
```

Running `--backend hardware` currently fails safely and explains what is missing.

## MVP and limits

MVP success is sorting five separated, flattened socks—two pairs plus one singleton—three times consecutively without unsafe motion or human intervention.

Not yet implemented:

- physics/contact or cloth deformation
- automatic camera homography calibration
- touching, overlapping, or tangled sock separation
- collision-aware trajectory planning
- visual post-action verification and retry
- physical motor control
- learned pattern embeddings

Good stretch goals are visual verification, uncertain-match human confirmation, depth-aware grasps, learned embeddings, overlapping socks, and folding matched pairs.

## Safety

- Confirm the physical emergency stop before every hardware session.
- Start without socks, then use one sock and one arm.
- Use low velocity and a conservative vertical travel height.
- Keep people outside the arm workspace while motion is enabled.
- Stop on camera loss, stale joint feedback, IK failure, unexpected contact, or workspace violation.
- Never send unvisualized or unbounded IK output to hardware.
