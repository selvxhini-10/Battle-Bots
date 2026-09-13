from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2

from .matching import find_pairs
from .models import Event, Pose, RunState, SockObservation
from .perception import segment_socks
from .planning import PixelTableTransform, destination_pose, make_grasp_plan
from .robot import BracketBotRobot, SimulatedRobot
from .smart import OptionalModelError, attach_clip_embeddings, segment_with_sam
from .synthetic import make_scene
from .visualization import annotate_scene, save_rerun


class SolematesApp:
    def __init__(self, config: dict, detector: str = "threshold", matcher: str = "classical"):
        self.config = config
        self.detector = detector
        self.matcher = matcher
        self.events: list[Event] = []
        self.step = 0
        execution = config["execution"]
        if execution["backend"] == "hardware":
            self.robot = BracketBotRobot()
        else:
            root = Path(__file__).parents[1]
            urdf = root / "chopped_urdf_v2/chopped_urdf_v2/urdf/chopped_urdf_v2.urdf"
            self.robot = SimulatedRobot(config["workspace"]["table_bounds_m"], self.emit, urdf)

    def emit(self, state: RunState, message: str, data: dict | None = None) -> None:
        self.step += 1
        self.events.append(Event(self.step, state, message, data or {}))

    def analyze(self, image) -> tuple[list[SockObservation], list, list[int], list]:
        self.emit(RunState.SCAN, "Scanning table")
        perception = self.config["perception"]
        if self.detector == "sam":
            socks = segment_with_sam(image, perception["sam"])
        else:
            socks = segment_socks(image, perception)
        if self.matcher == "clip":
            attach_clip_embeddings(image, socks, self.config["matching"]["clip"])
        self.emit(RunState.SCAN, f"Detected {len(socks)} socks", {"socks": [sock.summary() for sock in socks]})
        self.emit(RunState.MATCH, "Scoring possible couples")
        matching = self.config["matching"]
        weights = matching["clip_weights"] if self.matcher == "clip" else matching["weights"]
        pairs, singles, scores = find_pairs(socks, weights, matching["match_threshold"])
        self.emit(
            RunState.MATCH,
            f"Accepted {len(pairs)} pairs and {len(singles)} singles",
            {"pairs": [asdict(pair) for pair in pairs], "singles": singles},
        )
        return socks, pairs, singles, scores

    def execute(self, socks: list[SockObservation], pairs, singles: list[int]) -> None:
        workspace = self.config["workspace"]
        camera = self.config.get("camera", {})
        transform = PixelTableTransform(
            tuple(workspace["image_size"]), tuple(workspace["table_bounds_m"]),
            homography=camera.get("homography"),
            calibration_image_size=camera.get("image_size"),
        )
        by_id = {sock.id: sock for sock in socks}
        pair_destinations = workspace["couples_origins_m"]

        for pair_index, pair in enumerate(pairs):
            plans = [
                make_grasp_plan(by_id[pair.first_id], "right", transform, workspace),
                make_grasp_plan(by_id[pair.second_id], "left", transform, workspace),
            ]
            for state, plan in zip((RunState.PICK_FIRST, RunState.PICK_SECOND), plans):
                self.emit(state, f"Picking sock {plan.sock_id} with {plan.arm} arm", {"plan": _plan_json(plan)})
                self.robot.move_end_effector(plan.arm, plan.pregrasp)
                self.robot.move_end_effector(plan.arm, plan.grasp)
                self.robot.set_gripper(plan.arm, True)
                if isinstance(self.robot, SimulatedRobot):
                    self.robot.attach(plan.arm, plan.sock_id)
                self.robot.move_end_effector(plan.arm, plan.lift)

            self.emit(RunState.REUNITE, f"Reuniting socks {pair.first_id} and {pair.second_id}")
            reunion_y = -0.03
            safe_z = float(workspace["safe_height_m"])
            self.robot.move_end_effector("right", Pose(0.09, reunion_y, safe_z, 0.0))
            self.robot.move_end_effector("left", Pose(-0.09, reunion_y, safe_z, 0.0))

            self.emit(RunState.PLACE_PAIR, f"Placing couple {pair_index + 1}")
            origin = pair_destinations[min(pair_index, len(pair_destinations) - 1)]
            for slot, arm in enumerate(("right", "left")):
                destination = destination_pose(origin, slot, workspace)
                self.robot.move_end_effector(arm, Pose(destination.x, destination.y, safe_z, destination.yaw))
                self.robot.move_end_effector(arm, destination)
                self.robot.set_gripper(arm, False)
                if isinstance(self.robot, SimulatedRobot):
                    self.robot.release(arm)
                self.robot.move_end_effector(arm, Pose(destination.x, destination.y, safe_z, destination.yaw))
            self.emit(RunState.VERIFY, "Pair placement verified in simulation", {"pair": [pair.first_id, pair.second_id]})

        for slot, sock_id in enumerate(singles):
            plan = make_grasp_plan(by_id[sock_id], "left", transform, workspace)
            self.emit(RunState.HANDLE_SINGLE, f"Escorting sock {sock_id} to the Singles Club", {"plan": _plan_json(plan)})
            self.robot.move_end_effector("left", plan.pregrasp)
            self.robot.move_end_effector("left", plan.grasp)
            self.robot.set_gripper("left", True)
            if isinstance(self.robot, SimulatedRobot):
                self.robot.attach("left", sock_id)
            self.robot.move_end_effector("left", plan.lift)
            destination = destination_pose(workspace["singles_origin_m"], slot, workspace)
            self.robot.move_end_effector("left", Pose(destination.x, destination.y, workspace["safe_height_m"], 0.0))
            self.robot.move_end_effector("left", destination)
            self.robot.set_gripper("left", False)
            if isinstance(self.robot, SimulatedRobot):
                self.robot.release("left")
            self.emit(RunState.VERIFY, "Singles placement verified in simulation", {"sock": sock_id})

        self.emit(RunState.CELEBRATE, "All socks accounted for — happy wiggle!")
        self.emit(RunState.DONE, "Solemates run complete")


def _plan_json(plan) -> dict:
    return {
        "sock_id": plan.sock_id,
        "arm": plan.arm,
        "grasp": asdict(plan.grasp),
        "pregrasp": asdict(plan.pregrasp),
        "lift": asdict(plan.lift),
    }


def _load_image(args, config):
    if args.synthetic:
        image, labels = make_scene(*config["workspace"]["image_size"])
        return image, {"synthetic_labels": labels}
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"Could not read image: {args.image}")
        return image, {}
    camera = cv2.VideoCapture(args.camera)
    ok, image = camera.read()
    camera.release()
    if not ok:
        raise SystemExit(f"Could not capture from camera {args.camera}")
    return image, {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find every sock's solemate.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="run the built-in five-sock demo")
    source.add_argument("--image", type=Path, help="analyze a tabletop image")
    source.add_argument("--camera", type=int, help="capture one frame from a camera index")
    default_config = Path(__file__).parents[1] / "config.json"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    parser.add_argument("--backend", choices=("simulated", "hardware"), help="override configured backend")
    parser.add_argument("--detector", choices=("threshold", "sam"), help="sock mask source")
    parser.add_argument("--matcher", choices=("classical", "clip"), help="pair feature stack")
    parser.add_argument("--sam-checkpoint", type=Path, help="override SAM checkpoint path")
    parser.add_argument("--device", help="model device: auto, cpu, cuda, or mps")
    parser.add_argument("--analyze-only", action="store_true", help="skip simulated robot execution")
    parser.add_argument("--rrd", action="store_true", help="also write a Rerun recording when installed")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not (args.synthetic or args.image or args.camera is not None):
        args.synthetic = True
    config = json.loads(args.config.read_text())
    configured_checkpoint = Path(config["perception"]["sam"]["checkpoint"])
    if not configured_checkpoint.is_absolute():
        config["perception"]["sam"]["checkpoint"] = str(args.config.parent / configured_checkpoint)
    if args.backend:
        config["execution"]["backend"] = args.backend
    detector = args.detector or config["perception"].get("method", "threshold")
    matcher = args.matcher or config["matching"].get("method", "classical")
    if args.sam_checkpoint:
        config["perception"]["sam"]["checkpoint"] = str(args.sam_checkpoint)
    if args.device:
        config["perception"]["sam"]["device"] = args.device
        config["matching"]["clip"]["device"] = args.device
    if args.synthetic:
        # A real camera calibration does not describe the generated scene.
        config.pop("camera", None)
    image, metadata = _load_image(args, config)
    actual_size = [image.shape[1], image.shape[0]]
    config["workspace"]["image_size"] = actual_size

    app = SolematesApp(config, detector=detector, matcher=matcher)
    try:
        socks, pairs, singles, scores = app.analyze(image)
    except OptionalModelError as exc:
        raise SystemExit(f"Optional model setup error: {exc}") from exc
    execution_error = None
    status = "analyzed" if args.analyze_only else "completed"
    if not args.analyze_only:
        try:
            app.execute(socks, pairs, singles)
        except Exception as exc:
            status = "failed"
            state = app.events[-1].state if app.events else RunState.MATCH
            execution_error = {"type": type(exc).__name__, "message": str(exc)}
            app.emit(state, "Execution failed", {"error": execution_error})

    args.output.mkdir(parents=True, exist_ok=True)
    annotated = annotate_scene(image, socks, pairs, singles)
    cv2.imwrite(str(args.output / "scene.png"), image)
    cv2.imwrite(str(args.output / "matches.png"), annotated)
    report = {
        **metadata,
        "detector": detector,
        "matcher": matcher,
        "status": status,
        "error": execution_error,
        "sock_count": len(socks),
        "pairs": [asdict(pair) for pair in pairs],
        "singles": singles,
        "all_scores": [asdict(pair) for pair in scores],
        "events": [event.json() for event in app.events],
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2))
    if execution_error is not None:
        raise SystemExit(f"Execution failed ({execution_error['type']}): {execution_error['message']}\n"f"Artifacts: {args.output.resolve()}")
    wrote_rrd = args.rrd and save_rerun(args.output / "run.rrd", image, annotated, socks)

    print(f"Detected {len(socks)} socks")
    for pair in pairs:
        print(f"  couple: S{pair.first_id} + S{pair.second_id} ({pair.score:.1%})")
    for sock_id in singles:
        print(f"  single: S{sock_id}")
    print(f"Artifacts: {args.output.resolve()}")
    if args.rrd and not wrote_rrd:
        print("Rerun not installed; skipped run.rrd (install with: pip install -e '.[viewer]')")


if __name__ == "__main__":
    main()
