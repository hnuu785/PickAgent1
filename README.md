# PickAgent 개발 환경

## 설치

Miniforge 또는 Conda와 Git을 설치한 뒤 프로젝트 루트에서 실행한다. `lerobot/`은 별도의 Git 저장소이며 PickAgent 저장소에는 포함하지 않는다. 이미 해당 폴더가 있으면 clone 명령은 생략한다.

```bash
git clone https://github.com/huggingface/lerobot.git lerobot
conda create -n pickagent python=3.12 ffmpeg -c conda-forge -y
conda activate pickagent
python -m pip install -r requirements.txt
```

`requirements.txt`는 프로젝트 패키지와 함께 로컬 LeRobot 소스를 editable 모드로 설치하며, SO-101의 Feetech 모터와 로봇 실행에 필요한 추가 패키지를 포함한다. 설치 후 다음 명령으로 확인한다.

```bash
python -c "import lerobot; print(lerobot.__version__)"
python -c "import lerobot.scripts.lerobot_find_port; import serial; import rustypot; print('SO-101 motor dependencies OK')"
ffmpeg -version
```

`lerobot-find-port`는 도움말 명령이 아니라 USB 케이블을 분리하고 Enter를 누르는 대화형 포트 확인 도구다. 로봇을 조립하고 연결할 때 실행한다.

## 다시 사용할 때

```bash
conda activate pickagent
```

종료할 때는 `conda deactivate`를 실행한다. 시뮬레이션 실행 방법은 [simulation/README.md](simulation/README.md)를 참고한다.
