import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str = Field(description="Name of the detected object")
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax] on a 0-1000 scale")


class Detections(BaseModel):
    boxes: list[Detection]


def main():
    parser = argparse.ArgumentParser(description="Gemini로 이미지 속 물체 찾기")
    parser.add_argument("target", nargs="?", default="파란색 자동차", help="찾을 물체")
    parser.add_argument("--image", type=Path, default=Path(__file__).parent / "examples/toys.png")
    parser.add_argument("--model", choices=("flash", "robotics"), default="flash",
                        help="사용할 모델: flash (기본값) 또는 robotics (Gemini Robotics ER 2)")
    args = parser.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_api_key_here":
        raise SystemExit(".env 파일에 GEMINI_API_KEY를 설정해 주세요.")

    client = genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)))
    with Image.open(args.image) as source:
        photo = source.convert("RGB")
    model = "gemini-robotics-er-2-preview" if args.model == "robotics" else "gemini-3.1-flash-lite"
    print(f"{model} 모델로 '{args.target}' 찾는 중...", flush=True)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[photo, f"Find only objects matching this description: {args.target}. "
                      "Return an empty boxes list if none match. Each box must be "
                      "[ymin, xmin, ymax, xmax] normalized to 0-1000."],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=Detections),
        )
        detections = Detections.model_validate_json(response.text)
    except errors.APIError as exc:
        raise SystemExit(f"Gemini API 오류: {exc.code} ({exc.status})") from None

    draw = ImageDraw.Draw(photo)
    width, height = photo.size
    for detection in detections.boxes:
        if len(detection.box_2d) != 4:
            raise SystemExit("Gemini가 잘못된 좌표를 반환했습니다. 다시 시도해 주세요.")
        ymin, xmin, ymax, xmax = detection.box_2d
        if not (0 <= xmin < xmax <= 1000 and 0 <= ymin < ymax <= 1000):
            raise SystemExit("Gemini가 범위 밖의 좌표를 반환했습니다. 다시 시도해 주세요.")
        box = (round(xmin * width / 1000), round(ymin * height / 1000),
               round(xmax * width / 1000), round(ymax * height / 1000))
        draw.rectangle(box, outline="lime", width=6)
        print(f"찾은 물체: {detection.label}, 위치 (픽셀): {box}")
    if not detections.boxes:
        print("일치하는 물체를 찾지 못했습니다.")
        return
    filename = "detected_robotics.png" if args.model == "robotics" else "detected.png"
    output = Path(__file__).parent / "outputs" / filename
    output.parent.mkdir(exist_ok=True)
    photo.save(output)
    print(f"결과 이미지: {output}")


if __name__ == "__main__":
    main()
