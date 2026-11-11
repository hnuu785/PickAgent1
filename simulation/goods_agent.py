"""Interpret a goods request and validate it against the current camera image."""
from dataclasses import dataclass
import math
import re

import cv2
from google.genai import types
import numpy as np
from PIL import ImageDraw


COLOR_NAMES = {"red": "빨간", "blue": "파란", "yellow": "노란", "green": "초록"}
COLOR_WORDS = {
    "red": r"빨간|빨강|붉은|적색|\bred\b",
    "blue": r"파란|파랑|푸른|청색|\bblue\b",
    "yellow": r"노란|노랑|황색|\byellow\b",
    "green": r"초록|녹색|\bgreen\b",
}


@dataclass(frozen=True)
class Candidate:
    color: str
    y: float
    x: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Selection:
    status: str
    detail: str
    candidate: Candidate | None = None
    model_point: tuple[float, float] | None = None


def find_candidates(image):
    """Find solid-color cubes in the controlled MuJoCo scene."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = pixels.transpose(2, 0, 1)
    masks = {
        "red": (red > 70) & (red > 1.55 * green) & (red > 1.55 * blue),
        "blue": (blue > 70) & (blue > 1.45 * red) & (blue > 1.45 * green),
        "yellow": (red > 85) & (green > 75) & (green > .65 * red) & (blue < .55 * green),
    }
    height, width = pixels.shape[:2]
    candidates = []
    for color, mask in masks.items():
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        for index in range(1, count):
            x, y, w, h, area = map(int, stats[index])
            # At this camera height a 30 mm cube spans about 28 pixels.
            # The arm also has yellow details, but their shapes differ.
            if 20 <= w <= 36 and 20 <= h <= 36 and area >= .65 * w * h:
                cx, cy = map(float, centers[index])
                candidates.append(Candidate(color, cy * 1000 / height, cx * 1000 / width,
                                            (x, y, x + w - 1, y + h - 1)))
    return candidates


def requested_colors(request):
    return [color for color, pattern in COLOR_WORDS.items() if re.search(pattern, request, re.I)]


def left_or_right(request):
    left = re.search(r"왼쪽|\bleft\b", request, re.I)
    right = re.search(r"오른쪽|\bright\b", request, re.I)
    if bool(left) != bool(right):
        return "left" if left else "right"
    return None


def choose_candidate(request, candidates, color, model_point=None):
    if color not in COLOR_NAMES:
        return Selection("unsupported", "원하는 굿즈의 색을 알려주세요.")
    explicit = requested_colors(request)
    if explicit and explicit != [color]:
        return Selection("ambiguous", "어떤 색의 굿즈 하나를 원하시는지 알려주세요.")
    matching = sorted((item for item in candidates if item.color == color), key=lambda item: item.x)
    if not matching:
        return Selection("not_found", f"{COLOR_NAMES[color]} 굿즈를 찾지 못했습니다.")
    side = left_or_right(request)
    if len(matching) > 1 and (not side or matching[-1].x - matching[0].x < 20):
        return Selection("ambiguous", f"{COLOR_NAMES[color]} 굿즈가 여러 개입니다. 어느 위치의 굿즈인가요?")
    target = matching[-1] if side == "right" else matching[0]
    if model_point is not None:
        nearest = min(matching, key=lambda item: math.dist(model_point, (item.y, item.x)))
        if nearest != target or math.dist(model_point, (target.y, target.x)) > 65:
            return Selection("location_mismatch", "Gemini가 반환한 위치와 카메라에서 찾은 대상이 일치하지 않습니다.",
                             model_point=model_point)
    return Selection("selected", "요청한 굿즈를 찾았습니다.", target, model_point)


def select_by_color(request, candidates):
    colors = requested_colors(request)
    if len(colors) != 1:
        return Selection("ambiguous", "원하는 굿즈의 색 하나를 알려주세요.")
    return choose_candidate(request, candidates, colors[0])


def select_with_gemini(client, model, request, image_bytes, candidates):
    coordinate = {"type": "number", "minimum": 0, "maximum": 1000}
    tool = types.Tool(function_declarations=[types.FunctionDeclaration(
        name="locate_goods",
        description="요청한 색과 사진에서 찾은 해당 굿즈의 중심을 보고한다.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": list(COLOR_NAMES)},
                "center_y": coordinate,
                "center_x": coordinate,
            },
            "required": ["color"],
        },
    )])
    config = types.GenerateContentConfig(
        tools=[tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        system_instruction=(
            "이 가상 인형뽑기에서는 요청한 큐브 하나를 고정 배출구로 옮긴다. "
            "요청이 굿즈 하나를 고르는 것일 때 locate_goods를 정확히 한 번 호출하라. "
            "색은 사용자가 요청한 색이며, 위치는 현재 사진에서 실제로 보이는 대상의 중심이다. "
            "center_y는 위쪽 0 아래쪽 1000, center_x는 왼쪽 0 오른쪽 1000이다. "
            "대상이 없거나 여러 후보 중 어느 것인지 불명확하면 color만 보고하고 좌표는 생략하라. "
            "사용자가 좌우를 구분하면 사진에서 해당하는 후보의 좌표를 보고하라. "
            "굿즈 선택이 아닌 요청은 함수 없이 설명하라."
        ),
    )
    message = types.Content(role="user", parts=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        types.Part.from_text(text=request),
    ])
    response = client.models.generate_content(model=model, contents=[message], config=config)
    calls = response.function_calls or []
    if len(calls) != 1 or calls[0].name != "locate_goods":
        return Selection("unsupported", response.text or "지원되는 굿즈 선택 요청을 입력해 주세요.")
    args = dict(calls[0].args or {})
    color = args.get("color")
    if not isinstance(color, str) or set(args) not in ({"color"}, {"color", "center_y", "center_x"}):
        return Selection("invalid_response", "Gemini의 도구 호출 형식이 올바르지 않습니다.")
    if "center_y" not in args:
        selection = choose_candidate(request, candidates, color)
        if selection.candidate:
            return Selection("invalid_response", "Gemini가 대상의 위치를 반환하지 않았습니다.")
        return selection
    point = (args["center_y"], args["center_x"])
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1000 for value in point):
        return Selection("invalid_response", "Gemini가 유효하지 않은 이미지 좌표를 반환했습니다.")
    return choose_candidate(request, candidates, color, point)


def annotate(image, candidates, selection):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for index, item in enumerate(candidates, 1):
        chosen = item == selection.candidate
        draw.rectangle(item.box, outline="white" if chosen else "black", width=3)
        draw.text((item.box[0], item.box[1] - 13), f"{index}. {item.color}", fill="black")
        if chosen:
            x, y = item.x * image.width / 1000, item.y * image.height / 1000
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="white")
    if selection.model_point is not None:
        y, x = selection.model_point
        x, y = x * image.width / 1000, y * image.height / 1000
        draw.line((x - 7, y, x + 7, y), fill="magenta", width=2)
        draw.line((x, y - 7, x, y + 7), fill="magenta", width=2)
    return marked
