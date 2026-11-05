"""Gemini camera-based fixed-table pick-and-place experiment (simulation only)."""
import argparse
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


def run_motion(model, data, cube_xy, zone_xy, viewer):
    toy = model.body("toy").id
    toy_geom = model.geom("toy_geom").id
    fixed, moving = model.body("gripper").id, model.body("moving_jaw_so101_v1").id
    seed = np.array([-0.1064721, 0.51173375, -0.61329353, 0.56664408, -0.82321947, 0.0])
    x, y = cube_xy + np.array([.005564, .00086])
    dx, dy = zone_xy
    stages = [("접근", [x, y, .13], .84835), ("내려가기", [x, y, .019773], .84835),
              ("집기", [x, y, .019773], -.15), ("들어 올리기", [x, y, .13], -.15),
              ("목표 지점으로 이동", [dx, dy, .13], -.15),
              ("내려놓기", [dx, dy, .13], 1.3)]
    for name, target, grip in stages:
        if name == "목표 지점으로 이동":
            touching = set()
            for contact in data.contact:
                if contact.geom1 == toy_geom:
                    touching.add(model.geom_bodyid[contact.geom2])
                if contact.geom2 == toy_geom:
                    touching.add(model.geom_bodyid[contact.geom1])
            if data.xpos[toy, 2] < .05 or not {fixed, moving} <= touching:
                print("실패: 큐브를 양쪽 집게로 들어 올리지 못했습니다.", flush=True)
                return False
        goal, error = solve_position(model, seed, target)
        if error > .03:
            print(f"실패: {name} 목표에 접근할 수 없습니다 (오차 {error:.3f}m).", flush=True)
            return False
        goal[5] = grip
        start = data.ctrl[:6].copy()
        print(f"{name}: 목표 위치 {np.round(target, 4)}", flush=True)
        steps = int(3.25 / model.opt.timestep)
        for i in range(steps):
            if viewer is not None and not viewer.is_running():
                print("시뮬레이션 창이 닫혀 중단됐습니다.", flush=True)
                return False
            fraction = min((i + 1) / (steps * .7), 1)
            data.ctrl[:6] = start + (goal - start) * fraction
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                time.sleep(model.opt.timestep)
        seed = goal
    actual_zone = model.geom_pos[model.geom("drop_zone").id, :2]
    actual_cube = data.xpos[toy]
    # Use simulator state only to evaluate the outcome, never to choose the action.
    success = bool(np.linalg.norm(actual_cube[:2] - actual_zone) < .06 and actual_cube[2] < .04)
    print(f"최종 큐브 위치: {np.round(actual_cube, 4)}, 실제 목표: {np.round(actual_zone, 4)}", flush=True)
    print(f"MuJoCo 검증 결과: {'성공' if success else '실패'}", flush=True)
    return success


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
