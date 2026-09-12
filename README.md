# Solemates: Sock Matchmaker with Bracket Bot :) 

Solemates uses BracketBot's two arms to identify matching socks, pick up one sock with each arm, reunite the pair, and place unmatched socks in a **Singles Club** basket.

## MVP

Given five separated, flattened socks on a marked table:

1. Detect every sock from an overhead camera.
2. Match two pairs by color, pattern, size, and shape.
3. Pick one sock from each pair with each arm.
4. Bring the pair together and place it in the Couples area.
5. Put the unmatched sock in the Singles Club.
6. Verify every move and record the run in Rerun.

Not in the MVP: tangled piles, folding, general-purpose grasping, or autonomous base navigation.

## System architecture

```text
Overhead RGB camera
        |
        v
OpenCV segmentation and feature extraction
        |
        v
Pair scoring and global matching
        |
        v
Grasp and task planner
        |
        v
QP inverse kinematics
        |
        v
Simulator or BracketBot hardware interface

Camera frames, masks, matches, targets, joints, and outcomes -> Rerun
```

## Stack

- Python 3.11+
- Supplied `chopped_urdf_v2` URDF and meshes
- NumPy: geometry and feature calculations
- OpenCV: camera input, calibration, segmentation, contours, histograms, and PCA
- SciPy: optional optimal pair assignment
- Pink + Pinocchio: QP inverse kinematics, unless an organizer-provided IK stack is required
- Rerun: 2D/3D visualization and `.rrd` recordings
- YAML: calibration, workspace, motion, and threshold configuration
- Organizer-provided motor API or ROS interface: physical execution

Start with classical vision. Add learned image embeddings only if similar patterns cannot be distinguished reliably.

## What the starter package provides

- Robot geometry and joint tree
- Joint axes and limits
- `root`, `arm_base`, `left_eef`, and `right_eef` frames
- 18 movable joints, including mimic gripper joints
- Standalone Rerun URDF viewer

It does **not** provide sock perception, grasp planning, collision-free trajectories, motor control, or grasp feedback. Obtain the physical robot API and emergency-stop procedure from the organizers before hardware testing.

## Workspace setup

- Fix the robot and overhead camera in place.
- Use a plain, high-contrast mat with marked Unsorted, Couples, and Singles regions.
- Begin with four to six visually distinctive adult socks, separated and flattened.
- Keep all motion inside a calibrated table rectangle.
- Approach and lift vertically; travel above a conservative safe height.
- Run at low speed with a person ready to stop the robot.

## Perception

### Calibration

Use four or more known mat points to compute a pixel-to-table transform. If the camera is fixed and the table is planar, a homography is sufficient. Store the transform and table height in `config.yaml`.

### Sock detection

For each camera frame:

1. Remove or threshold the known background.
2. Clean the binary mask with morphological operations.
3. Find connected components or contours.
4. Reject regions outside configured area limits.
5. Record each mask, centroid, outline, orientation, and confidence.

If socks touch, the MVP may ask the user to separate them. Overlap separation is a stretch goal.

### Features and matching

For each sock, calculate:

- HSV or Lab color histogram
- Texture/pattern descriptor
- Mask area and dimensions
- Outline/shape descriptor
- Principal orientation from PCA

Example pair score:

```text
score(i, j) = 0.45 * color_similarity
            + 0.30 * pattern_similarity
            + 0.15 * size_similarity
            + 0.10 * shape_similarity
```

Choose a globally consistent set of non-overlapping pairs. Accept a pair only above `match_threshold`; send remaining socks to Singles. Tune weights and the threshold on labeled test images rather than during the final demo.

## Grasp planning

For each sock mask:

1. Estimate its long axis and unobstructed regions.
2. Prefer the cuff or a thick region away from mask boundaries.
3. Convert the selected pixel to table coordinates.
4. Orient the gripper across the local fabric direction.
5. Generate pre-grasp, grasp, lift, transport, place, and retreat poses.

Represent a target as:

```python
GraspTarget(
    position=(x, y, z),
    yaw=yaw,
    approach_height=0.12,
    gripper_width=0.025,
)
```

If bare fabric is unreliable, place a removable felt tab inside each cuff and document it as an engineered grasp affordance.

## Motion sequence

```text
move above target -> orient -> descend -> close -> pause -> lift vertically
-> travel above destination -> descend -> open -> retreat
```

Develop with one arm first. The final reunion sequence is:

1. Right arm picks sock A.
2. Left arm picks sock B.
3. Both lift to safe height.
4. Both move to a central presentation pose.
5. Pause, then place the socks side by side.
6. Retreat and perform a small, prevalidated happy wiggle.

QP IK converts `left_eef` or `right_eef` targets into joint updates while enforcing URDF joint limits. Treat mimic gripper joints as dependent values. Validate every generated pose in Rerun before enabling physical execution.

## State machine

```text
SCAN -> MATCH -> PICK_FIRST -> PICK_SECOND -> REUNITE -> PLACE_PAIR
  ^                                                        |
  |                                                        v
RECOVER <------------------------ VERIFY <-----------------+

Remaining low-confidence sock -> HANDLE_SINGLE -> VERIFY
No remaining socks -> CELEBRATE -> DONE
```

After every action, rescan the table. A placement succeeds only if the sock disappears from its source region and appears near its destination. On failure: open, retreat, rescan, select a new grasp point, and retry. Stop and request human help after two failed attempts.

## Software layout

```text
solemates/
├── app.py
├── config.yaml
├── perception/
│   ├── camera.py
│   ├── calibration.py
│   ├── segmentation.py
│   └── features.py
├── matching/
│   └── pair_socks.py
├── manipulation/
│   ├── grasp_planner.py
│   ├── task_planner.py
│   ├── ik.py
│   └── motions.py
├── robot/
│   ├── interface.py
│   ├── simulated_robot.py
│   └── bracketbot_robot.py
├── visualization/
│   └── rerun_logger.py
└── tests/
    └── images/
```

Both robot backends should implement:

```python
robot.move_end_effector(arm, target_pose)
robot.set_gripper(arm, closed)
robot.get_joint_positions()
robot.stop()
```

This keeps perception and planning identical in simulation and on hardware.

## Configuration

```yaml
camera:
  device: 0
  homography: []

workspace:
  table_z: 0.0
  safe_height: 0.12
  bounds: [xmin, xmax, ymin, ymax]
  couples_pose: [x, y, z, yaw]
  singles_pose: [x, y, z, yaw]

matching:
  weights: {color: 0.45, pattern: 0.30, size: 0.15, shape: 0.10}
  match_threshold: 0.75

execution:
  backend: simulated
  speed_scale: 0.15
  max_retries: 2
```

Never store unexplained joint-angle arrays in application logic. Name reusable poses and document their coordinate frame.

## Rerun logging

Log at minimum:

- Raw and annotated camera frames
- Segmentation masks and sock IDs
- Pair scores and accepted matches
- Pixel and world-coordinate grasp points
- End-effector targets and planned paths
- Actual joint positions, when available
- State-machine transitions, retries, and outcomes

Save a complete `.rrd` recording for debugging and the project demo.

## Build order

1. **Scripted simulation:** hard-code two sock locations and animate pickup, reunion, placement, and celebration in the URDF viewer.
2. **Vision only:** correctly match four distinctive pairs across at least ten test layouts.
3. **Calibration:** clicking a camera pixel moves the simulated end effector above the corresponding table point.
4. **One-arm manipulation:** reliably pick and place one flattened sock.
5. **Closed loop:** detect, match, pick, place, rescan, and recover.
6. **Dual-arm demo:** add simultaneous presentation, Singles Club, and happy wiggle.

Keep independently demonstrable fallbacks: real vision with simulated manipulation, and manual coordinates with real manipulation.

## Tests and success criteria

- Unit-test similarity scores using labeled matching and non-matching pairs.
- Test calibration error across the entire workspace, not only at its center.
- Test IK reachability and joint limits for every named destination.
- Test empty scenes, odd sock counts, low-confidence matches, failed grasps, and camera loss.
- Run physical motions first without socks, then with one sock, then with both arms.
- MVP success: correctly sort five separated socks in three consecutive runs without unsafe motion or manual intervention.

## Stretch goals

- Similar-pattern matching with learned embeddings
- Touching or partially overlapping socks
- Depth-aware grasp selection
- Grasp-success sensing
- Mild untangling
- Folding matched pairs
- Active human clarification for uncertain matches
- Mobile-base collection from multiple stations

## Safety

- Confirm the emergency stop before every hardware session.
- Enforce joint, velocity, workspace, and table-height limits in software.
- Keep people outside the arm workspace while motion is enabled.
- Never execute unvisualized poses or unbounded IK output.
- Stop on lost camera input, stale joint feedback, IK failure, or unexpected contact.
