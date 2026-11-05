"""Let Gemini request the existing fixed-scene MuJoCo pick-and-place action."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-robotics-er-2-preview"
DEFAULT_REQUEST = "주황색 정육면체를 집어서 녹색 목표 지점에 놓아줘."


def main():
    parser = argparse.ArgumentParser(description="Gemini가 MuJoCo 로봇에게 작업 요청")
    parser.add_argument("request", nargs="?", default=DEFAULT_REQUEST, help="자연어 작업 요청")
    parser.add_argument("--headless", action="store_true", help="화면 없이 실행")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_api_key_here":
        raise SystemExit(".env 파일에 GEMINI_API_KEY를 설정해 주세요.")

    client = genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)))
    tool = types.Tool(function_declarations=[types.FunctionDeclaration(
        name="pick_and_place_cube",
        description="가상의 주황색 큐브를 집어 녹색 목표 지점에 내려놓는다.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "enum": ["orange_cube"]},
                "destination": {"type": "string", "enum": ["green_zone"]},
            },
            "required": ["object_name", "destination"],
        },
    )])
    config = types.GenerateContentConfig(
        tools=[tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        system_instruction=(
            "가상 MuJoCo 장면에는 주황색 큐브 1개와 녹색 목표 지점 1개가 있다. "
            "사용자가 그 큐브를 녹색 목표 지점으로 옮기라고 요청하면 함수를 호출하라. "
            "없는 물체나 다른 장소를 요청하면 함수를 호출하지 말고 불가능하다고 설명하라. "
            "함수 결과를 받기 전에는 작업이 성공했다고 주장하지 마라."
        ),
    )
    print(f"요청: {args.request}\nGemini Robotics ER 2에 요청하는 중...", flush=True)
    try:
        user_message = types.Content(role="user", parts=[types.Part.from_text(text=args.request)])
        response = client.models.generate_content(model=MODEL, contents=[user_message], config=config)
        calls = response.function_calls or []
        if not calls:
            print(f"Gemini 응답: {response.text or '함수 호출이 반환되지 않았습니다.'}")
            return 1
        if len(calls) != 1:
            print("지원하지 않는 여러 함수 호출이 반환되어 실행을 중단합니다.")
            return 1
        call = calls[0]
        arguments = dict(call.args or {})
        if call.name != "pick_and_place_cube" or arguments != {
            "object_name": "orange_cube", "destination": "green_zone"
        }:
            print("지원하지 않는 작업이 요청되어 실행을 중단합니다.")
            return 1

        print(f"Gemini 함수 호출: {call.name} {arguments}", flush=True)
        interpreter = Path(sys.executable).parent / ("python" if args.headless else "mjpython")
        command = [str(interpreter), str(ROOT / "simulation" / "pick_demo.py")]
        if args.headless:
            command.append("--headless")
        completed = subprocess.run(command, cwd=ROOT, check=False)
        success = completed.returncode == 0
        result = {"success": success, "detail": "큐브를 목표 지점에 내려놓음" if success else "시뮬레이션 실패 또는 중단"}
        print(f"MuJoCo 검증 결과: {'성공' if success else '실패 또는 중단'}", flush=True)

        # Keep the original model content, including any function-call metadata.
        if response.candidates and response.candidates[0].content:
            tool_result = types.Content(role="user", parts=[
                types.Part.from_function_response(name=call.name, response=result)
            ])
            try:
                final = client.models.generate_content(
                    model=MODEL,
                    contents=[user_message, response.candidates[0].content, tool_result],
                    config=config,
                )
                if final.text:
                    print(f"Gemini 결과 설명: {final.text}")
            except errors.APIError as exc:
                print(f"Gemini 결과 설명을 받지 못했습니다: {exc.code} ({exc.status})")
        return 0 if success else 1
    except errors.APIError as exc:
        print(f"Gemini API 오류: {exc.code} ({exc.status})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
