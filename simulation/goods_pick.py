"""Run the natural-language goods demo with two baskets and its test scenarios."""
import argparse
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import httpx
import numpy as np
from PIL import Image, ImageDraw

from goods_agent import COLOR_NAMES, Selection, annotate, find_candidates, select_by_color, select_with_gemini


ROOT = Path(__file__).resolve().parents[1]
SCENE = Path(__file__).with_name("goods_scene.xml")
DOLL_SCENE = Path(__file__).with_name("doll_scene.xml")
MODEL = "gemini-robotics-er-2-preview"
HOME_POSE = (-1.3, 0, 0, 0, 0, 0)
DROP_ZONE = (.20, -.18)
WORKSPACE = (.17, .24, -.085, .085)
COLORS = ("red", "blue", "yellow")
RED, BLUE, YELLOW = (COLOR_NAMES[color] for color in COLORS)
POSITIONS_A = ((.20, 0), (.19, -.07), (.22, .045))
POSITIONS_B = (POSITIONS_A[1], POSITIONS_A[2], POSITIONS_A[0])
PALETTE = {"red": (.9, .07, .07), "blue": (.07, .25, .95), "yellow": (.95, .82, .07)}


@dataclass(frozen=True)
class Case:
    name: str
    request: str
    expected: str
    color: str
    positions: tuple = POSITIONS_A
    colors: tuple = COLORS
    grip_closes: bool = True


CASES = {
    case.name: case for case in (
        Case("a-blue", f"{BLUE} 굿즈 뽑아줘", "success", "blue"),
        Case("a-red", f"{RED} 굿즈 하나 줘", "success", "red"),
        Case("a-yellow", f"{YELLOW} 큐브를 배출구로 옮겨줘", "success", "yellow"),
        Case("b-blue", f"{BLUE} 굿즈 하나 줘", "success", "blue", POSITIONS_B),
        Case("b-red", f"{RED} 큐브를 배출구로 옮겨줘", "success", "red", POSITIONS_B),
        Case("b-yellow", f"{YELLOW} 굿즈 뽑아줘", "success", "yellow", POSITIONS_B),
        Case("missing", "초록 굿즈 뽑아줘", "not_found", "green"),
        Case("ambiguous", f"{BLUE} 굿즈 뽑아줘", "ambiguous", "blue", POSITIONS_B,
             ("blue", "blue", "yellow")),
        Case("out-of-range", f"{BLUE} 굿즈 뽑아줘", "out_of_range", "blue",
             (POSITIONS_A[0], (.33, -.03), POSITIONS_A[2])),
        Case("grasp-failure", f"{BLUE} 굿즈 뽑아줘", "grasp_failed", "blue", grip_closes=False),
    )
}


@dataclass(frozen=True)
class PickCommand:
    color: str
    source_xy_m: tuple[float, float]
    destination_xy_m: tuple[float, float] = DROP_ZONE


@dataclass(frozen=True)
class PickResult:
    success: bool
    status: str
    stage: str
    detail: str
    sim_seconds: float = 0
    verified_cube: str | None = None


class MockController:
    destination_xy_m = DROP_ZONE

    def __init__(self, succeed=True):
        self.succeed = succeed

    def pick(self, command):
        return PickResult(self.succeed, "success" if self.succeed else "grasp_failed", "Mock",
                          f"{COLOR_NAMES[command.color]} 굿즈의 집기 동작을 모의 실행했습니다."
                          if self.succeed else "모의 파지 실패입니다.")


class MuJoCoController:
    def __init__(self, case, viewer=False, dolls=False):
        import mujoco
        from vision_pick import HEIGHT, WIDTH, execute_motion, pixel_to_world

        self.mujoco = mujoco
        self.width, self.height = WIDTH, HEIGHT
        self.project = pixel_to_world
        self.execute = execute_motion
        self.case = case
        self.viewer = viewer
        self.model = mujoco.MjModel.from_xml_path(str(DOLL_SCENE if dolls else SCENE))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:6] = HOME_POSE
        self.data.ctrl[:6] = HOME_POSE
        mujoco.mj_forward(self.model, self.data)
        floor = self.model.geom("source_floor")
        self.source_surface_z = float(self.data.geom_xpos[floor.id, 2] + floor.size[2])
        self.destination_xy_m = tuple(map(float, self.data.geom("drop_zone").xpos[:2]))
        for slot, (position, color) in enumerate(zip(case.positions, case.colors), 1):
            address = self.model.joint(f"cube_{slot}_joint").qposadr[0]
            self.data.qpos[address:address + 3] = [*position, self.source_surface_z + .015]
            if dolls:
                self.data.qpos[address + 3:address + 7] = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
            for geom_id in range(self.model.ngeom):
                geom = self.model.geom(geom_id)
                if int(geom.bodyid[0]) == self.model.body(f"cube_{slot}").id and "_eye_" not in geom.name:
                    geom.rgba[:3] = PALETTE[color]
        mujoco.mj_forward(self.model, self.data)

    def photo(self, camera="overhead"):
        with self.mujoco.Renderer(self.model, self.height, self.width) as renderer:
            renderer.update_scene(self.data, camera=camera)
            return Image.fromarray(renderer.render())

    def workbench_position(self, candidate):
        return tuple(map(float, self.project(self.model, self.data, candidate.y, candidate.x,
                                             self.source_surface_z + .03)))

    def pick(self, command):
        x, y = command.source_xy_m
        xmin, xmax, ymin, ymax = WORKSPACE
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return PickResult(False, "out_of_range", "작업 범위 확인", "대상이 작업 범위 밖에 있습니다.")

        # Simulator identities are read only to verify the grasp and delivery.
        # The commanded arm trajectory uses the camera-projected coordinates.
        bodies = [f"cube_{slot}" for slot, color in enumerate(self.case.colors, 1) if color == command.color]
        nearest = min(bodies, key=lambda name: np.linalg.norm(self.data.body(name).xpos[:2] - (x, y)),
                      default=None)
        if nearest is None or np.linalg.norm(self.data.body(nearest).xpos[:2] - (x, y)) > .03:
            return PickResult(False, "location_mismatch", "검증 대상 확인", "검증할 큐브를 찾을 수 없습니다.")
        viewer = None
        try:
            if self.viewer:
                from mujoco import viewer as mj_viewer
                viewer = mj_viewer.launch_passive(self.model, self.data)
            result = self.execute(self.model, self.data, (x, y), command.destination_xy_m, viewer,
                                  body_name=nearest, return_home=True, grip_closes=self.case.grip_closes,
                                  source_floor="source_floor")
        finally:
            if viewer is not None:
                viewer.close()
        status = "success" if result.success else ("grasp_failed" if result.stage == "파지 검증" else "motion_failed")
        return PickResult(result.success, status, result.stage, result.detail, round(self.data.time, 3), nearest)


def mock_photo(case):
    image = Image.new("RGB", (640, 480), (210, 210, 210))
    draw = ImageDraw.Draw(image)
    for (x, y), color in zip(case.positions, case.colors):
        px, py = 320 + (x - .20) * 930, 240 - (y + .05) * 930
        draw.rectangle((px - 13, py - 13, px + 13, py + 13),
                       fill=tuple(round(value * 255) for value in PALETTE[color]))
    return image


def run_case(case, request, mode, output, client=None, viewer=False, dolls=False):
    output.mkdir(parents=True, exist_ok=True)
    # Each run begins from a fresh scene and writes evidence for that run only.
    for filename in ("camera.png", "detected.png", "final.png", "result.json"):
        (output / filename).unlink(missing_ok=True)
    controller = MockController(case.grip_closes) if mode == "mock" else MuJoCoController(case, viewer, dolls)
    selection = None
    command = None
    if mode == "motion":
        if case.color not in case.colors:
            raise ValueError("동작 단독 모드는 장면에 있는 색을 선택해야 합니다.")
        command = PickCommand(case.color, case.positions[case.colors.index(case.color)],
                              controller.destination_xy_m)
    else:
        image = mock_photo(case) if mode == "mock" else controller.photo()
        original = output / "camera.png"
        image.save(original)
        candidates = find_candidates(image, dolls=dolls)
        try:
            if mode == "gemini":
                selection = select_with_gemini(client, MODEL, request, original.read_bytes(), candidates)
            else:
                selection = select_by_color(request, candidates)
        except errors.APIError as exc:
            selection = Selection("api_error", f"Gemini API 오류 ({exc.code} {exc.status})")
        except httpx.TransportError as exc:
            selection = Selection("api_error", f"Gemini 연결 실패 ({type(exc).__name__})")
        annotate(image, candidates, selection).save(output / "detected.png")
        if selection.candidate is not None:
            candidate = selection.candidate
            if mode == "mock":
                xy = (.20 + (candidate.x / 1000 * image.width - 320) / 930,
                      -.05 - (candidate.y / 1000 * image.height - 240) / 930)
            else:
                xy = controller.workbench_position(candidate)
            command = PickCommand(candidate.color, xy, controller.destination_xy_m)

    if command is None:
        result = PickResult(False, selection.status, "대상 선택", selection.detail)
    else:
        print(f"대상 {command.color} · 작업대 좌표 {np.round(command.source_xy_m, 4)} m", flush=True)
        result = controller.pick(command)
        if mode in ("gemini", "color") and result.sim_seconds:
            controller.photo().save(output / "final.png")

    moved_when_blocked = case.expected in ("not_found", "ambiguous", "out_of_range") and result.sim_seconds > 0
    correct_color = case.expected != "success" or (command and command.color == case.color)
    record = {
        "case": case.name, "request": request, "mode": mode,
        "expected_status": case.expected if request == case.request else None,
        "passed": bool(result.status == case.expected and not moved_when_blocked and correct_color)
        if request == case.request else None,
        "selection": asdict(selection) if selection else None,
        "command": asdict(command) if command else None,
        "result": asdict(result),
    }
    (output / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{case.name}] {result.status} · {result.detail}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description="가상 SO-101 자연어 굿즈 뽑기")
    parser.add_argument("request", nargs="?", help="자연어 요청. 생략하면 선택한 케이스의 기본 요청을 사용합니다.")
    parser.add_argument("--case", choices=CASES, default="a-blue")
    parser.add_argument("--mode", choices=("gemini", "color", "motion", "mock"), default="gemini",
                        help="Gemini, 로컬 이미지 색상 인식, 관절 동작만, 또는 Mock")
    parser.add_argument("--evaluate", action="store_true", help="공통 시나리오 10개 실행")
    parser.add_argument("--viewer", action="store_true", help="로봇 동작 창 열기 (macOS는 mjpython 사용)")
    parser.add_argument("--output", type=Path, help="카메라 이미지와 JSON을 저장할 디렉터리")
    parser.add_argument("--dolls", action="store_true", help="큐브 대신 인형 모양 강체 사용")
    args = parser.parse_args()
    if args.evaluate and (args.request or args.viewer or args.mode not in ("gemini", "color")):
        parser.error("--evaluate는 개별 요청이나 뷰어 없이 gemini 또는 color 모드에서 사용하세요.")
    if args.viewer and args.mode == "mock":
        parser.error("Mock 모드에는 시뮬레이션 창이 없습니다.")
    if args.dolls and args.mode == "mock":
        parser.error("인형 장면은 MuJoCo 모드에서 실행하세요.")
    client = None
    if args.mode == "gemini":
        load_dotenv(ROOT / ".env")
        key = os.getenv("GEMINI_API_KEY")
        if not key or key == "your_api_key_here":
            parser.error(".env에 GEMINI_API_KEY를 설정해 주세요.")
        client = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)))
    output = args.output or ROOT / "outputs" / ("dolls" if args.dolls else "baskets") / args.mode
    try:
        if args.evaluate:
            results = [run_case(case, case.request, args.mode, output / case.name, client, dolls=args.dolls)
                       for case in CASES.values()]
            (output / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
            passed = sum(record["passed"] for record in results)
            print(f"공통 케이스 통과 {passed}/{len(results)}")
            return 0 if passed == len(results) else 1
        case = CASES[args.case]
        record = run_case(case, args.request or case.request, args.mode, output / case.name, client, args.viewer,
                          args.dolls)
        return 0 if record["result"]["success"] else 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
