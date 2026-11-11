# Gemini와 MuJoCo 카메라로 큐브 옮기기

## 2주차 · 여러 굿즈 중 하나를 골라 배출구로 옮기기

프로젝트 루트에서 가상환경을 활성화하고 `requirements.txt`의 패키지를 설치한다. Gemini 실행에는 `.env`의 `GEMINI_API_KEY`가 필요하다.

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python simulation/goods_pick.py "파란 굿즈 뽑아줘"
```

시뮬레이션에는 빨강·파랑·노랑 큐브, 상부 카메라, 녹색 고정 배출구가 있다. Gemini가 요청한 색과 카메라 이미지상의 중심을 함수 호출로 반환한다. 프로그램은 이미지에서 찾은 후보와 모델 좌표를 비교하고, 후보 중심을 작업대 좌표로 변환한다. 제어기는 해당 위치를 향해 움직이고, 요청한 큐브가 양쪽 집게에 잡혔는지와 배출구에 도착했는지를 확인한다.

```bash
# 정상 6건, 예외 4건을 각각 초기 상태에서 실행
python simulation/goods_pick.py --evaluate

# API 없이 같은 장면과 케이스 실행
python simulation/goods_pick.py --mode color --evaluate

# 카메라와 API 없이 관절·그리퍼·배출·복귀 확인
python simulation/goods_pick.py --mode motion --case a-blue

# MuJoCo 없이 성공·실패 응답 확인
python simulation/goods_pick.py --mode mock
python simulation/goods_pick.py --mode mock --case grasp-failure

# 후보가 둘일 때의 질문과 위치를 다시 지정한 요청
python simulation/goods_pick.py --case ambiguous
python simulation/goods_pick.py --case ambiguous "왼쪽 파란 굿즈 뽑아줘"

# macOS에서 로봇 동작 창 열기
mjpython simulation/goods_pick.py --viewer
```

Gemini와 로컬 색상 인식 모드는 창을 띄우지 않아도 카메라 렌더링 환경이 필요하다. `--mode color`는 이 장면의 고정 카메라와 단색 큐브에 맞춘 로컬 검증 모드다. `grasp-failure`는 집게가 닫히지 않는 조건을 설정해 실패를 재현한다. 단일 실행은 새 장면에서 시작하며, 성공 시 종료 코드 0, 예외 안내나 실패 시 1을 반환한다. `--evaluate`는 모든 케이스가 각자의 기대 결과를 충족할 때 종료 코드 0을 반환한다.

카메라 원본 `camera.png`, 후보와 Gemini 좌표를 표시한 `detected.png`, 동작 직후의 `final.png`, 실행 정보 `result.json`은 `outputs/week2/<모드>/<케이스>/`에 저장된다. 동일 경로에서 다시 실행하면 해당 케이스의 파일이 갱신된다. 전체 검증은 모드 디렉터리에 `summary.json`을 저장한다. 기존 결과를 보관하려면 `--output`으로 별도 경로를 지정한다.

## 1주차 · 단일 큐브와 목표 지점 찾기

프로젝트 루트에서 가상환경을 활성화하고 `.env`의 `GEMINI_API_KEY`를 설정한다.

```bash
source .venv/bin/activate
mjpython simulation/vision_pick.py
```

큐브와 목표 지점의 위치를 바꾸거나 자연어 요청을 전달할 수 있다. 좌표는 로봇 작업대 기준 미터 단위다.

```bash
mjpython simulation/vision_pick.py "주황색 큐브를 녹색 원으로 옮겨줘" --cube 0.21 0.01 --zone 0.22 -0.16
```

실험 가능한 배치 범위는 큐브 `x=0.18~0.22`, `y=-0.02~0.02`, 녹색 목표 지점 `x=0.16~0.24`, `y=-0.22~-0.14`다. 큐브와 목표 지점은 최소 0.10m 떨어뜨린다. 범위는 지원하는 입력의 제한이며, 모든 위치에서 파지 성공을 보장하지 않는다.

`--headless`는 로봇 동작 창을 열지 않는다. 이 경우에도 카메라 이미지를 생성하려면 MuJoCo 렌더링이 가능한 환경이 필요하다.

```bash
python simulation/vision_pick.py --cube 0.19 -0.01 --zone 0.18 -0.20 --headless
```

MuJoCo 카메라 이미지가 Gemini Robotics ER 2로 전송된다. Gemini가 이미지에서 큐브와 녹색 원의 중심을 찾아 함수 호출에 담으면, 고정 카메라의 위치와 각도 및 알려진 물체 높이를 이용해 작업대 좌표로 변환한다. 변환 좌표로 집기와 내려놓기 목표를 계산하고 물리 시뮬레이션으로 동작한다. 원본과 인식 지점을 표시한 이미지는 각각 `outputs/mujoco_camera.png`, `outputs/mujoco_camera_detected.png`에 저장되며 다음 실행 시 덮어쓴다.

이 실험은 고정 카메라, 단일 큐브, 평평한 작업대와 고정된 물체 크기를 전제로 한다. Gemini가 모든 관절 동작을 직접 만드는 것은 아니다. 로봇 제어기는 기존의 파지 자세와 단계별 동작을 활용해 이미지에서 찾은 위치에 맞는 관절 각도를 계산한다. 성공 여부는 시뮬레이터 내부의 접촉과 최종 위치로 평가한다. 실제 로봇이나 다양한 물체에서 검증된 코드는 아니다.
