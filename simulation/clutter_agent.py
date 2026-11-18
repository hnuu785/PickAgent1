"""Gemini-directed clutter experiment; MuJoCo state is used for execution feedback."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import httpx
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from vision_pick import GraspPlan, execute_motion, pixel_to_world

ROOT = Path(__file__).resolve().parents[1]
SCENE = Path(__file__).with_name('clutter_scene.xml')
MODEL = 'gemini-robotics-er-2-preview'
WIDTH, HEIGHT = 640, 480
HOME = [-1.3, 0, 0, 0, 0, 0]


def capture(model, data):
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, camera='overhead')
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera='overhead')
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera='overhead')
        segmentation = renderer.render().copy()
    return Image.fromarray(rgb), depth, segmentation


def select_point(model, data, depth, segmentation, point):
    if (not isinstance(point, (list, tuple)) or len(point) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v < 1000 for v in point)):
        raise ValueError('선택 좌표는 0 이상 1000 미만의 [y, x]여야 합니다.')
    y, x = point
    row, col = min(int(y * HEIGHT / 1000), HEIGHT - 1), min(int(x * WIDTH / 1000), WIDTH - 1)
    geom_id, kind = map(int, segmentation[row, col])
    if kind != int(mujoco.mjtObj.mjOBJ_GEOM) or not 0 <= geom_id < model.ngeom:
        raise ValueError('선택 지점에 상품이 보이지 않습니다.')
    body_id = int(model.geom_bodyid[geom_id])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ''
    if not name.startswith('goods_'):
        raise ValueError('선택 지점이 상품 표면이 아닙니다.')
    # Overhead camera looks straight down; MuJoCo depth is distance along its view axis.
    z = float(data.cam_xpos[model.camera('overhead').id, 2] - depth[row, col])
    xy = pixel_to_world(model, data, y, x, z)
    if not (.14 <= xy[0] <= .26 and -.105 <= xy[1] <= .105 and .005 <= z <= .16):
        raise ValueError('선택 지점이 시범 작업 범위 밖입니다.')
    return name, xy, z


def ask(client, request, image_path, history):
    coord = {'type': 'number', 'minimum': 0, 'maximum': 999}
    tool = {
        'type': 'function',
        'name': 'choose_next_action',
        'description': '다음 한 번의 이동 또는 작업 중단을 선택합니다.',
        'parameters': {'type': 'object', 'properties': {
            'action': {'type': 'string', 'enum': ['deliver_target', 'move_obstacle', 'stop']},
            'point_y': coord, 'point_x': coord,
            'reason': {'type': 'string'},
        }, 'required': ['action', 'reason']},
    }
    system_instruction = (
        '당신은 시뮬레이션 굿즈 뽑기 작업 계획자입니다. 사진과 사용자 요청을 보고 다음 행동 하나만 '
        'choose_next_action으로 반환하세요. 회색 바구니 안의 물체만 고릅니다. 초록 바구니는 배출구, '
        '파란 판은 방해물 임시 위치입니다. 사진에서 보이는 집기 쉬운 물체 표면을 선택하세요. '
        '목표를 집을 수 있으면 deliver_target, 목표를 가리는 물체를 먼저 옮겨야 하면 move_obstacle을 '
        '선택합니다. 좌표는 이미지 위에서 아래 y, 왼쪽에서 오른쪽 x이며 0~999입니다. '
        '목표가 보이지 않아 특정할 수 없거나 후보가 모호하거나 진행할 수 없으면 stop을 선택하고 '
        '이유를 설명하세요. 이미지에 없는 물체나 가려진 위치를 추측하지 마세요. '
        '이동 결과를 받은 뒤에는 새 사진으로 다시 판단하세요.')
    response = client.interactions.create(
        model=MODEL,
        api_version='v1beta',
        input=[
            {'type': 'image', 'data': image_path, 'mime_type': 'image/png'},
            {'type': 'text', 'text': (
                f'사용자 요청: {request}\n'
                f'이전 동작 결과: {json.dumps(history, ensure_ascii=False)}')},
        ],
        system_instruction=system_instruction,
        tools=[tool],
        generation_config={'tool_choice': {
            'allowed_tools': {'mode': 'any', 'tools': ['choose_next_action']},
        }},
    )
    calls = [step for step in response.steps if step.type == 'function_call']
    if len(calls) != 1 or calls[0].name != 'choose_next_action':
        raise ValueError('Gemini가 다음 행동을 하나 반환하지 않았습니다: ' + str(response.output_text or ''))
    action = dict(calls[0].arguments or {})
    if action.get('action') not in ('deliver_target', 'move_obstacle', 'stop') or not isinstance(action.get('reason'), str):
        raise ValueError('Gemini의 행동 형식이 올바르지 않습니다.')
    return action


def main():
    parser = argparse.ArgumentParser(description='20개 스캔 굿즈 장면에서 Gemini가 다음 행동을 선택하는 실험')
    parser.add_argument('request', nargs='?', default='장난감 버스 하나를 배출구로 옮겨줘')
    parser.add_argument('--viewer', action='store_true', help='MuJoCo 창 표시 (macOS: mjpython)')
    parser.add_argument('--preview', action='store_true', help='API 호출 없이 장면만 표시')
    parser.add_argument('--max-moves', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_moves <= 10:
        parser.error('--max-moves는 1~10이어야 합니다.')
    load_dotenv(ROOT / '.env')
    key = os.getenv('GEMINI_API_KEY')
    if not args.preview and (not key or key == 'your_api_key_here'):
        parser.error('.env에 GEMINI_API_KEY를 설정해 주세요.')
    output = ROOT / 'outputs' / 'clutter' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    data.qpos[:6] = HOME
    data.ctrl[:6] = HOME
    for _ in range(1200):
        mujoco.mj_step(model, data)
    viewer = None
    client = None
    history = []
    result = {'request': args.request, 'status': 'incomplete', 'history': history}
    try:
        if args.viewer or args.preview:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(model, data)
        if args.preview:
            image, _, _ = capture(model, data)
            image.save(output / 'preview.png')
            print(f'장면 저장: {output / "preview.png"}', flush=True)
            while viewer.is_running():
                mujoco.mj_step(model, data); viewer.sync(); time.sleep(model.opt.timestep)
            result['status'] = 'preview'
            return 0
        client = genai.Client(api_key=key, http_options=types.HttpOptions(
            api_version='v1beta', timeout=60000,
            retry_options=types.HttpRetryOptions(attempts=1)))
        for step in range(args.max_moves):
            image, depth, segmentation = capture(model, data)
            path = output / f'{step:02d}_camera.png'
            image.save(path)
            action = ask(client, args.request, path, history)
            print(f'Gemini {step+1}: {json.dumps(action, ensure_ascii=False)}', flush=True)
            if action['action'] == 'stop':
                history.append({'decision': action})
                result['status'] = 'stopped_by_agent'
                break
            point = [action.get('point_y'), action.get('point_x')]
            try:
                body, xy, surface_z = select_point(model, data, depth, segmentation, point)
            except ValueError as exc:
                history.append({'decision': action, 'result': {'success': False, 'detail': str(exc)}})
                result['status'] = 'invalid_selection'
                break
            marked = image.copy(); draw = ImageDraw.Draw(marked)
            px, py = point[1] * WIDTH / 1000, point[0] * HEIGHT / 1000
            draw.ellipse((px-6, py-6, px+6, py+6), outline='magenta', width=3)
            marked.save(output / f'{step:02d}_selected.png')
            zone = 'drop_zone' if action['action'] == 'deliver_target' else 'staging_zone'
            destination = data.geom(zone).xpos[:2].copy()
            # The point is a visible surface; this simple grasp assumes a roughly 3 cm thick torso.
            plan = GraspPlan(offset_xy=(0.0, 0.0),
                pick_z=max(.015, surface_z - .015), carry_z=.18, release_z=.095)
            motion = execute_motion(model, data, xy, destination, viewer, body_name=body,
                return_home=True, source_floor='source_floor', plan=plan,
                destination_geom=zone, stop_on_other_goods=False,
                continue_after_failed_grasp=True)
            history.append({'decision': action, 'selected_body_for_simulation_feedback': body,
                            'surface_xyz_m': [*map(float, xy), surface_z], 'result': asdict(motion)})
            print(f'{motion.stage}: {motion.detail}', flush=True)
            if not motion.success:
                result['status'] = 'motion_failed'
                break
            if action['action'] == 'deliver_target':
                result['status'] = 'delivered_identity_unverified'
                break
        if result['status'] == 'incomplete':
            result['status'] = 'move_limit'
        image, _, _ = capture(model, data)
        image.save(output / 'final.png')
        return 0 if result['status'] == 'delivered_identity_unverified' else 1
    except (errors.APIError, httpx.HTTPError, ValueError) as exc:
        result['status'] = 'error'
        result['error'] = f'{type(exc).__name__}: {exc}'
        print(result['error'], flush=True)
        return 1
    except Exception as exc:
        result['status'] = 'error'
        result['error'] = f'{type(exc).__name__}: {exc}'
        print(result['error'], flush=True)
        return 1
    finally:
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(f'결과: {output}', flush=True)
        if client is not None: client.close()
        if viewer is not None: viewer.close()


if __name__ == '__main__':
    raise SystemExit(main())
