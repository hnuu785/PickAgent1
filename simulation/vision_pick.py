"""Gemini camera-based fixed-table pick-and-place experiment (simulation only)."""
import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from pick_demo import SCENE, solve_position

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-robotics-er-2-preview"
WIDTH, HEIGHT = 640, 480
CUBE_DEFAULT = (0.20, 0.0)
ZONE_DEFAULT = (0.20, -0.18)


def pixel_to_world(model, data, y, x, plane_z):
    """Project a normalized image point onto a known horizontal surface."""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
    focal = HEIGHT / (2 * math.tan(math.radians(model.cam_fovy[camera_id]) / 2))
    u, v = x * WIDTH / 1000, y * HEIGHT / 1000
    ray_camera = np.array([(u - WIDTH / 2) / focal, -(v - HEIGHT / 2) / focal, -1.0])
    ray_world = data.cam_xmat[camera_id].reshape(3, 3) @ ray_camera
    origin = data.cam_xpos[camera_id]
    if abs(ray_world[2]) < 1e-8:
        raise ValueError("카메라 방향이 작업면과 평행합니다.")
    distance = (plane_z - origin[2]) / ray_world[2]
    if distance <= 0:
        raise ValueError("작업면이 카메라 뒤에 있습니다.")
    return (origin + distance * ray_world)[:2]


def in_range(point, xmin, xmax, ymin, ymax):
    return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax


def parse_and_validate(call):
    fields = ("cube_y", "cube_x", "zone_y", "zone_x")
    if call.name != "pick_and_place" or set(call.args or {}) != set(fields):
        raise ValueError("지원하지 않는 함수 호출입니다.")
    values = [call.args[field] for field in fields]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 1000 for v in values):
        raise ValueError("Gemini가 유효하지 않은 이미지 좌표를 반환했습니다.")
    return values


@dataclass(frozen=True)
class MotionResult:
    success: bool
    stage: str
    detail: str


@dataclass(frozen=True)
class GraspPlan:
    offset_xy: tuple[float, float] = (.005564, .00086)
    pick_z: float | None = None
    carry_z: float = .13
    release_z: float | None = None


def execute_motion(model, data, cube_xy, zone_xy, viewer=None, *,
                   body_name="toy", return_home=False, grip_closes=True, source_floor=None,
                   plan=None):
    plan = plan or GraspPlan()
    toy = model.body(body_name).id
    toy_geoms = set(np.flatnonzero(model.geom_bodyid == toy))
    drop = model.geom("drop_zone")
    fixed, moving = model.body("gripper").id, model.body("moving_jaw_so101_v1").id
    seed = np.array([-0.1064721, 0.51173375, -0.61329353, 0.56664408, -0.82321947, 0.0])
    home = data.ctrl[:6].copy()
    x, y = np.asarray(cube_xy) + np.asarray(plan.offset_xy)
    dx, dy = zone_xy
    closed = -.15 if grip_closes else .84835
    basket_bodies = set()
    pick_z, release_z = .019773, .13
    if source_floor is not None:
        floor = model.geom(source_floor)
        pick_z += data.geom_xpos[floor.id, 2] + floor.size[2]
        release_z = .085
        basket_bodies = {int(floor.bodyid[0]), int(drop.bodyid[0])}
    if plan.pick_z is not None:
        pick_z = plan.pick_z
    if plan.release_z is not None:
        release_z = plan.release_z
    robot_root = model.body("base").id
    geom_roots = model.body_rootid[model.geom_bodyid]
    stages = [("접근", [x, y, plan.carry_z], .84835), ("내려가기", [x, y, pick_z], .84835),
              ("집기", [x, y, pick_z], closed), ("들어 올리기", [x, y, plan.carry_z], closed),
              ("목표 지점으로 이동", [dx, dy, plan.carry_z], closed)]
    if basket_bodies:
        stages.append(("배출 바구니로 내리기", [dx, dy, release_z], closed))
    stages.append(("내려놓기", [dx, dy, release_z], 1.3))
    if basket_bodies:
        stages.append(("배출 바구니에서 나오기", [dx, dy, plan.carry_z], 1.3))

    def move(goal):
        start = data.ctrl[:6].copy()
        steps = int(3.25 / model.opt.timestep)
        for i in range(steps):
            if viewer is not None and not viewer.is_running():
                return "시뮬레이션 창이 닫혀 중단됐습니다."
            fraction = min((i + 1) / (steps * .7), 1)
            data.ctrl[:6] = start + (goal - start) * fraction
            mujoco.mj_step(model, data)
            if basket_bodies:
                for contact in data.contact:
                    first, second = geom_roots[contact.geom1], geom_roots[contact.geom2]
                    if (first == robot_root and second in basket_bodies
                            or second == robot_root and first in basket_bodies):
                        return "로봇팔이 바구니에 닿아 동작을 중단했습니다."
            if viewer is not None:
                viewer.sync()
                time.sleep(model.opt.timestep)
        return None

    for name, target, grip in stages:
        if name == "목표 지점으로 이동":
            touching = set()
            for contact in data.contact:
                if contact.geom1 in toy_geoms:
                    touching.add(model.geom_bodyid[contact.geom2])
                elif contact.geom2 in toy_geoms:
                    touching.add(model.geom_bodyid[contact.geom1])
            if data.xpos[toy, 2] < .05 or not {fixed, moving} <= touching:
                return MotionResult(False, "파지 검증", "대상 물체를 양쪽 집게로 들어 올리지 못했습니다.")
        goal, error = solve_position(model, seed, target)
        if error > .03:
            return MotionResult(False, name, f"목표에 접근할 수 없습니다 (오차 {error:.3f}m).")
        goal[5] = grip
        print(f"{name}: 목표 위치 {np.round(target, 4)}", flush=True)
        stop_reason = move(goal)
        if stop_reason:
            return MotionResult(False, name, stop_reason)
        seed = goal

    if return_home:
        print("복귀: 초기 관절 자세", flush=True)
        stop_reason = move(home)
        if stop_reason:
            return MotionResult(False, "복귀", stop_reason)
    actual_zone = data.geom_xpos[drop.id, :2]
    actual_cube = data.xpos[toy]
    # Use simulator state only to evaluate the outcome, never to choose the action.
    if basket_bodies:
        extents = []
        for geom_id in toy_geoms:
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            size = model.geom_size[geom_id]
            geom_type = model.geom_type[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = model.geom_dataid[geom_id]
                start = model.mesh_vertadr[mesh_id]
                stop = start + model.mesh_vertnum[mesh_id]
                vertices = np.einsum("ij,kj->ki", rotation, model.mesh_vert[start:stop])
                vertices += data.geom_xpos[geom_id]
                if not np.all(np.isfinite(vertices)):
                    return MotionResult(False, "배출 확인", "물체의 위치를 확인할 수 없습니다.")
                low, high = vertices.min(axis=0), vertices.max(axis=0)
                extents.append(((low + high) / 2, (high - low) / 2))
                continue
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = np.full(3, size[0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = size[0] + np.abs(rotation[:, 2]) * size[1]
            elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                extent = np.sqrt((rotation ** 2) @ (size ** 2))
            else:
                extent = np.abs(rotation) @ size
            extents.append((data.geom_xpos[geom_id].copy(), extent))
        inside = all(np.all(np.abs(center[:2] - actual_zone) + extent[:2] <= drop.size[:2])
                     for center, extent in extents)
        bottom = data.geom_xpos[drop.id, 2] + drop.size[2]
        object_bottom = min(center[2] - extent[2] for center, extent in extents)
        on_floor = abs(object_bottom - bottom) < .004
        stopped = np.linalg.norm(data.cvel[toy, 3:]) < .01
        success = bool(inside and on_floor and stopped)
    else:
        success = bool(np.linalg.norm(actual_cube[:2] - actual_zone) < .06 and actual_cube[2] < .04)
    print(f"최종 물체 위치: {np.round(actual_cube, 4)}, 실제 목표: {np.round(actual_zone, 4)}", flush=True)
    detail = "요청한 물체가 배출구에 도착했습니다." if success else "요청한 물체가 배출구에 도착하지 않았습니다."
    return MotionResult(success, "배출 확인", detail)


def run_motion(model, data, cube_xy, zone_xy, viewer):
    result = execute_motion(model, data, cube_xy, zone_xy, viewer)
    print(f"MuJoCo 검증 결과: {'성공' if result.success else '실패'} · {result.detail}", flush=True)
    return result.success


def main():
    parser = argparse.ArgumentParser(description="Gemini와 MuJoCo 카메라를 이용한 큐브 옮기기")
    parser.add_argument("request", nargs="?", default="주황색 큐브를 녹색 목표 지점에 옮겨줘")
    parser.add_argument("--cube", nargs=2, type=float, metavar=("X", "Y"), default=CUBE_DEFAULT)
    parser.add_argument("--zone", nargs=2, type=float, metavar=("X", "Y"), default=ZONE_DEFAULT)
    parser.add_argument("--headless", action="store_true", help="물리 동작을 창 없이 실행")
    args = parser.parse_args()
    if not in_range(args.cube, .18, .22, -.02, .02) or not in_range(args.zone, .16, .24, -.22, -.14):
        raise SystemExit("실험 범위 밖입니다. 큐브 x=.18~.22, y=-.02~.02 / 목표 x=.16~.24, y=-.22~-.14")
    if np.linalg.norm(np.array(args.cube) - np.array(args.zone)) < .10:
        raise SystemExit("큐브와 목표 지점을 더 떨어뜨려 주세요.")
    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_api_key_here":
        raise SystemExit(".env 파일에 GEMINI_API_KEY를 설정해 주세요.")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    data.qpos[:6] = [-1.3, 0, 0, 0, 0, 0]
    data.ctrl[:6] = data.qpos[:6]
    # The toy is the only free joint; look it up by its owning body.
    free = next(i for i in range(model.njnt) if model.jnt_bodyid[i] == model.body("toy").id)
    address = model.jnt_qposadr[free]
    data.qpos[address:address + 3] = [*args.cube, .015]
    model.geom_pos[model.geom("drop_zone").id, :2] = args.zone
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, camera="overhead")
        photo = Image.fromarray(renderer.render())
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    photo.save(output / "mujoco_camera.png")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)))
    coordinate = {"type": "number", "minimum": 0, "maximum": 1000}
    tool = types.Tool(function_declarations=[types.FunctionDeclaration(
        name="pick_and_place", description="사진에서 찾은 주황색 큐브와 녹색 원의 중심 좌표로 시뮬레이션 작업을 실행한다.",
        parameters_json_schema={"type": "object", "properties": {
            "cube_y": coordinate, "cube_x": coordinate, "zone_y": coordinate, "zone_x": coordinate},
            "required": ["cube_y", "cube_x", "zone_y", "zone_x"]})])
    config = types.GenerateContentConfig(
        tools=[tool], automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        system_instruction=("이것은 가상 로봇 실험이다. 사용자 요청이 이미지의 주황색 큐브를 녹색 원으로 옮기는 작업이면 "
            "사진에서 각 물체의 시각적 중심을 찾아 pick_and_place 함수를 호출하라. "
            "각 중심 좌표는 이미지 상단이 0, 하단이 1000인 y와 왼쪽이 0, 오른쪽이 1000인 x를 사용한다. "
            "요청을 할 수 없거나 물체가 보이지 않으면 호출하지 말고 설명하라. 물체의 위치를 임의로 가정하지 마라."))
    print(f"요청: {args.request}\nMuJoCo 카메라 이미지를 Gemini Robotics ER 2에 보내는 중...", flush=True)
    try:
        message = types.Content(role="user", parts=[
            types.Part.from_bytes(data=(output / "mujoco_camera.png").read_bytes(), mime_type="image/png"),
            types.Part.from_text(text=args.request)])
        response = client.models.generate_content(model=MODEL, contents=[message], config=config)
    except errors.APIError as exc:
        print(f"Gemini API 오류: {exc.code} ({exc.status})")
        return 1
    calls = response.function_calls or []
    if len(calls) != 1:
        print(f"실행할 수 없습니다: {response.text or '하나의 함수 호출을 받지 못했습니다.'}")
        return 1
    try:
        cy, cx, zy, zx = parse_and_validate(calls[0])
        cube = pixel_to_world(model, data, cy, cx, .03)
        zone = pixel_to_world(model, data, zy, zx, .002)
    except (ValueError, TypeError) as exc:
        print(f"위치 확인 실패: {exc}")
        return 1
    print(f"Gemini 함수 호출: {calls[0].name} {dict(calls[0].args)}", flush=True)
    print(f"이미지 좌표 → 로봇 좌표(m): 큐브={np.round(cube, 4)}, 목표={np.round(zone, 4)}", flush=True)
    marked = photo.copy()
    draw = ImageDraw.Draw(marked)
    for x, y, color in [(cx, cy, 'red'), (zx, zy, 'blue')]:
        px, py = x * WIDTH / 1000, y * HEIGHT / 1000
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
    marked.save(output / "mujoco_camera_detected.png")
    if not in_range(cube, .175, .225, -.025, .025) or not in_range(zone, .155, .245, -.225, -.135):
        print("인식된 위치가 이 실험의 제어 범위 밖입니다. 동작을 중단합니다.")
        return 1
    viewer = None
    try:
        if not args.headless:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(model, data)
        success = run_motion(model, data, cube, zone, viewer)
        return 0 if success else 1
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
