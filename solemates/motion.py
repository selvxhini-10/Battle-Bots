"""Build a complete, inspectable command sequence before simulated execution."""
from dataclasses import asdict, dataclass
import math

from .models import Pose, RunState
from .planning import destination_pose, make_grasp_plan, motion_heights


@dataclass
class Command:
    state: RunState
    action: str
    arm: str
    pose: Pose | None = None
    sock_id: int | None = None

    def json(self):
        return {**asdict(self), "state": self.state.value}


def build_motion_plan(socks, pairs, singles, transform, workspace):
    by_id = {sock.id: sock for sock in socks}
    assigned = [i for pair in pairs for i in (pair.first_id, pair.second_id)] + list(singles)
    if len(set(assigned)) != len(assigned) or set(assigned) != set(by_id):
        raise ValueError("Every observed sock must be assigned exactly once")
    if len(pairs) > len(workspace["couples_origins_m"]):
        raise ValueError("Not enough couples_origins_m for the matched pairs")
    arms = workspace["pair_arms"]
    if len(arms) != 2 or set(arms) != {"left", "right"} or workspace["single_arm"] not in arms:
        raise ValueError("Configure two distinct pair arms and a valid single arm")
    _, safe = motion_heights(workspace)
    commands = []

    def move(state, arm, pose):
        commands.append(Command(state, "move", arm, pose))

    def pick(sock_id, arm, state):
        plan = make_grasp_plan(by_id[sock_id], arm, transform, workspace)
        commands.append(Command(state, "open", arm))
        move(state, arm, plan.pregrasp)
        move(state, arm, plan.grasp)
        commands.append(Command(state, "pick", arm, sock_id=sock_id))
        move(state, arm, plan.lift)

    def place(sock_id, arm, pose, state):
        above = Pose(pose.x, pose.y, safe, pose.yaw)
        move(state, arm, above)
        move(state, arm, pose)
        commands.append(Command(state, "release", arm, sock_id=sock_id))
        move(state, arm, above)

    for index, pair in enumerate(pairs):
        ids = (pair.first_id, pair.second_id)
        for sock_id, arm, state in zip(ids, arms, (RunState.PICK_FIRST, RunState.PICK_SECOND)):
            pick(sock_id, arm, state)
        for arm in arms:
            xy = workspace["reunion_xy_m"][arm]
            move(RunState.REUNITE, arm, Pose(*xy, safe, float(workspace["placement_yaw_rad"])))
        for slot, (sock_id, arm) in enumerate(zip(ids, arms)):
            place(sock_id, arm, destination_pose(workspace["couples_origins_m"][index], slot, workspace), RunState.PLACE_PAIR)
    for slot, sock_id in enumerate(singles):
        arm = workspace["single_arm"]
        pick(sock_id, arm, RunState.HANDLE_SINGLE)
        place(sock_id, arm, destination_pose(workspace["singles_origin_m"], slot, workspace), RunState.HANDLE_SINGLE)

    xmin, xmax, ymin, ymax = workspace["table_bounds_m"]
    for command in commands:
        if command.pose is not None:
            p = command.pose
            if not all(math.isfinite(v) for v in (p.x, p.y, p.z, p.yaw)):
                raise ValueError(f"Planned pose must be finite: {p}")
            if not (xmin <= p.x <= xmax and ymin <= p.y <= ymax):
                raise ValueError(f"Planned {command.state.value} pose is outside table bounds: {p}")
    return commands
