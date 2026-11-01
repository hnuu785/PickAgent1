import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv(Path(__file__).with_name('.env'))
api_key = os.getenv('GEMINI_API_KEY')
if not api_key or api_key == 'your_api_key_here':
    raise SystemExit('먼저 .env 파일에 GEMINI_API_KEY를 설정해 주세요.')

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=20000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)
print('Gemini에 요청을 보내는 중입니다. 최대 20초 정도 기다려 주세요.', flush=True)
try:
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents='안녕! 로봇 프로젝트를 시작하는 팀에게 짧게 인사해 줘.',
    )
except errors.APIError as exc:
    print(f'Gemini API 오류: {exc.code} ({exc.status}). 잠시 후 다시 시도해 주세요.')
    raise SystemExit(1) from None
except Exception as exc:
    print(f'요청에 실패했습니다 ({type(exc).__name__}). 네트워크와 API 키 설정을 확인해 주세요.')
    raise SystemExit(1) from None
print(response.text)
