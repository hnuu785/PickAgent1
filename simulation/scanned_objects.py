"""Preview scanned objects or try a fixed-position grasp in MuJoCo."""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

from vision_pick import GraspPlan, execute_motion


SCENE = Path(__file__).with_name("scanned_objects_scene.xml")
OBJECTS = {"orca": "범고래", "rhino": "코뿔소", "bus": "장난감 버스"}
GRASP_PLANS = {
    "orca": GraspPlan(offset_xy=(0, 0), pick_z=.02, carry_z=.18, release_z=.14),
    "rhino": GraspPlan(offset_xy=(0, .01), pick_z=.02, carry_z=.18, release_z=.14),
    "bus": GraspPlan(offset_xy=(0, -.01), pick_z=.02, carry_z=.18, release_z=.14),
}


def main():
    parser = argparse.ArgumentParser(description="실물 스캔 물체가 담긴 바구니 시뮬레이션")
    parser.add_argument("--pick", choices=OBJECTS, default="bus", help="집을 물체 (기본값: bus)")
    parser.add_argument("--preview", action="store_true", help="물체를 옮기지 않고 장면만 보기")
    parser.add_argument("--headless", action="store_true", help="집기 동작을 창 없이 시험")
    args = parser.parse_args()
    if args.headless and args.preview:
        parser.error("--preview는 시뮬레이션 창에서 사용하세요.")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    data.qpos[:6] = [-1.3, 0, 0, 0, 0, 0]
    data.ctrl[:6] = data.qpos[:6]
    for _ in range(400):
        mujoco.mj_step(model, data)

    viewer = None
    try:
        if not args.headless:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(model, data)

        exit_code = 0
        if not args.preview:
            # Fixed-position physics experiment, using simulator state as input.
            source = data.body(args.pick).xpos[:2].copy()
            destination = data.geom("drop_zone").xpos[:2].copy()
            print(f"{OBJECTS[args.pick]} 집기 시험 · 물체 위치 {np.round(source, 4)}m", flush=True)
            result = execute_motion(model, data, source, destination, viewer,
                                    body_name=args.pick, return_home=True, source_floor="source_floor",
                                    plan=GRASP_PLANS[args.pick])
            print(f"{result.stage} · {result.detail}", flush=True)
            exit_code = 0 if result.success else 1
            if viewer is None:
                return exit_code

        print("결과를 확인한 뒤 MuJoCo 창을 닫으면 종료합니다.", flush=True)
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
        return exit_code
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
