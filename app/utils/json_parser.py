import json
import re

def parse_json(text: str) -> dict:
    try:
        json_str = re.search(r"\{.*\}", text, re.S).group()
        return json.loads(json_str)
    except Exception:
        return {
            "action": "response",
            "content": "LLM 출력 파싱 실패"
        }
