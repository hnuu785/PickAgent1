"""SO-101 pick experiment. Use --headless to check without a viewer."""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

SCENE = Path(__file__).with_name('pick_scene.xml')


def solve_position(model, seed, target):
    """Approximate joint angles for the gripper reference point, without orientation constraints."""
    data = mujoco.MjData(model)
    data.qpos[:6] = seed
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
    for _ in range(400):
        mujoco.mj_forward(model, data)
        error = np.asarray(target) - data.site_xpos[site]
        if np.linalg.norm(error) < 0.003:
            break
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, site)
        j = jac[:, :5]
        step = j.T @ np.linalg.solve(j @ j.T + 0.005 * np.eye(3), error)
        data.qpos[:5] = np.clip(data.qpos[:5] + np.clip(step, -0.08, 0.08),
                                 model.jnt_range[:5, 0], model.jnt_range[:5, 1])
    return data.qpos[:6].copy(), float(np.linalg.norm(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
    toy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'toy')
    toy_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'toy_geom')
    fixed = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
    moving = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'moving_jaw_so101_v1')
    # Seed selects an arm posture in which both fingers can surround the cube.
    seed = np.array([-0.1064721, 0.51173375, -0.61329353,
                     0.56664408, -0.82321947, 0.0])
    # Each point is the desired position of the gripper reference, in metres.
    stages = [('approach', [0.205564, 0.00086, 0.13], 0.84835),
              ('lower', [0.205564, 0.00086, 0.019773], 0.84835),
              ('grasp', [0.205564, 0.00086, 0.019773], -0.15),
              ('lift', [0.205564, 0.00086, 0.13], -0.15),
              ('move to drop zone', [0.20, -0.18, 0.13], -0.15),
              ('release', [0.20, -0.18, 0.13], 1.3)]
    try:
        for name, target, grip in stages:
            if name == 'move to drop zone':
                contact_bodies = set()
                for contact in data.contact:
                    if contact.geom1 == toy_geom:
                        contact_bodies.add(model.geom_bodyid[contact.geom2])
                    elif contact.geom2 == toy_geom:
                        contact_bodies.add(model.geom_bodyid[contact.geom1])
                if data.xpos[toy, 2] < 0.05 or not {fixed, moving} <= contact_bodies:
                    print('파지 실패: 물체를 양쪽 집게로 들고 있지 않아 이동을 중단합니다.', flush=True)
                    return 1
            goal, error = solve_position(model, seed, target)
            if error > 0.03:
                print(f'{name}: 목표 위치를 찾지 못했습니다 (오차 {error:.3f}m).', flush=True)
                return 1
            goal[5] = grip
            start = data.ctrl[:6].copy()
            print(f'{name}: 목표 {target}', flush=True)
            steps = int(3.25 / model.opt.timestep)
            for i in range(steps):
                if viewer and not viewer.is_running():
                    print('시뮬레이션 창이 닫혀 작업이 중단됐습니다.', flush=True)
                    return 1
                fraction = min((i + 1) / max(steps * .7, 1), 1)
                data.ctrl[:6] = start + (goal - start) * fraction
                mujoco.mj_step(model, data)
                if viewer:
                    viewer.sync()
                    time.sleep(model.opt.timestep)
            seed = goal
            print('  물체 위치:', np.round(data.xpos[toy], 3), flush=True)
        delivered = (np.linalg.norm(data.xpos[toy, :2] - [0.20, -0.18]) < .06
                     and data.xpos[toy, 2] < .04)
        print('배출구 주변 내려놓기 성공:', delivered)
        return 0 if delivered else 1
    finally:
        if viewer:
            viewer.close()


if __name__ == '__main__':
    raise SystemExit(main())
