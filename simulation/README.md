# Gemini와 MuJoCo 카메라로 큐브 옮기기

## 실제 스캔 물체가 담긴 바구니

Google Scanned Objects에서 가져온 범고래 봉제인형, 코뿔소 피규어, 장난감 버스를 상품 바구니에 배치했다. 실물 모델을 그리퍼와 바구니에 맞게 각각 16%, 43%, 60%로 축소했다. 배출 바구니의 벽은 작업대 기준 높이 8cm다. 각 물체는 고유한 질감과 충돌 메쉬를 가지며 강체로 움직인다.

```bash
conda activate pickagent
mjpython simulation/scanned_objects.py
```

`mjpython simulation/scanned_objects.py`는 버스를 집어 8cm 높이의 배출 바구니로 옮긴다. 코뿔소를 옮기려면 `mjpython simulation/scanned_objects.py --pick rhino`를 실행한다. `--preview`를 추가하면 물체가 정지한 장면을 볼 수 있다. 동작이 끝난 뒤에는 창을 닫을 때까지 결과 화면이 유지된다.

각 물체를 시뮬레이터에서 얻은 고정 위치와 물체별로 조정한 집기 지점으로 움직인다. 현재 배치에서 버스와 코뿔소를 세 번씩 배출하는 데 성공했다. `--pick orca`는 범고래를 집으려 하지만 로봇팔이 높은 배출 바구니 벽에 닿아 중단된다. 자연어·카메라 대상 인식은 아직 연결하지 않았다. 봉제인형도 강체 메쉬라 눌리거나 휘지 않는다.

모델 원본과 라이선스는 [scanned_objects/README.md](../models/scanned_objects/README.md)에 기록되어 있다.

## 인형 모양 물체 집기

상품 바구니에 머리, 몸통, 팔, 다리가 있는 인형 세 개를 넣고 집어 볼 수 있다. 인형의 각 부분에는 충돌과 질량을 적용했고, 배출 바구니 안에 인형 전체가 들어가야 성공으로 판정한다.

```bash
# API 없이 카메라 색상 인식으로 파란 인형 집기
mjpython simulation/goods_pick.py --dolls --mode color --viewer "파란 인형 뽑아줘"

# Gemini 이미지 인식과 자연어 요청을 함께 사용
mjpython simulation/goods_pick.py --dolls --viewer "파란 인형 뽑아줘"

# 두 배치의 정상 요청 6건과 예외 상황 4건을 확인
python simulation/goods_pick.py --dolls --mode color --evaluate
```

인형은 천이 변형되는 봉제인형이 아니라 여러 도형을 이어 만든 단단한 물체다. 색상과 놓인 방향이 일정하며, 크기·마찰·파지 자세는 실제 인형과 다를 수 있다. 결과는 `outputs/dolls/`에 저장된다.

## 2주차 · 여러 굿즈 중 하나를 골라 배출구로 옮기기

프로젝트 루트에서 가상환경을 활성화하고 `requirements.txt`의 패키지를 설치한다. Gemini 실행에는 `.env`의 `GEMINI_API_KEY`가 필요하다.

```bash
conda activate pickagent
python -m pip install -r requirements.txt
python simulation/goods_pick.py "파란 굿즈 뽑아줘"
```

평평한 작업대에 빨강·파랑·노랑 큐브가 담긴 회색 상품 바구니와 빈 녹색 배출 바구니를 놓았다. 두 바구니의 바닥과 벽에는 충돌이 적용된다. Gemini가 요청한 색과 카메라 이미지상의 중심을 함수 호출로 반환하면, 프로그램은 이미지에서 찾은 후보와 모델 좌표를 비교하고 바구니 바닥 높이를 반영해 작업대 좌표로 변환한다.

로봇팔은 큐브를 들어 상품 바구니 테두리 위로 옮긴 뒤 배출 바구니 쪽으로 낮춰 내려놓는다. 팔이 바구니에 닿으면 해당 동작을 중단한다. 큐브 전체가 배출 바구니 안에 들어가 바닥에 안착해야 성공으로 판정한다.

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

# macOS에서 API 키 없이 바구니 데모 보기
mjpython simulation/goods_pick.py --mode color --viewer
```

Gemini와 로컬 색상 인식 모드는 창을 띄우지 않아도 카메라 렌더링 환경이 필요하다. `--mode color`는 이 장면의 고정 카메라와 단색 큐브에 맞춘 로컬 검증 모드다. `grasp-failure`는 집게가 닫히지 않는 조건을 설정해 실패를 재현한다. 단일 실행은 새 장면에서 시작하며, 성공 시 종료 코드 0, 예외 안내나 실패 시 1을 반환한다. `--evaluate`는 모든 케이스가 각자의 기대 결과를 충족할 때 종료 코드 0을 반환한다.

바구니 데모의 카메라 원본 `camera.png`, 후보와 Gemini 좌표를 표시한 `detected.png`, 동작 직후의 `final.png`, 실행 정보 `result.json`은 `outputs/baskets/<모드>/<케이스>/`에 저장된다. 동일 경로에서 다시 실행하면 해당 케이스의 파일이 갱신된다. 전체 검증은 모드 디렉터리에 `summary.json`을 저장한다. 실행별 결과를 보관하려면 `--output`으로 별도 경로를 지정한다.

## 1주차 · 단일 큐브와 목표 지점 찾기

프로젝트 루트에서 가상환경을 활성화하고 `.env`의 `GEMINI_API_KEY`를 설정한다.

```bash
conda activate pickagent
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
