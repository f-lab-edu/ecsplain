from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """
You are an assistant that helps manage a codebase.

Given:
- A high-level project description
- A git diff of the current staged changes

Your task:
- Analyze what was changed.
- Recommend what the developer should do NEXT.

Guidelines:
- Answer in Korean.
- Focus on **practical, short-term next actions** right after this commit.
- Consider:
  - 어떤 테스트를 돌려야 하는지 (unit/integration/e2e 등)
  - 어떤 문서/README/주석/API 문서를 업데이트해야 하는지
  - 리팩터링이나 공통화가 필요한 부분
  - 배포/환경 설정 관련해서 확인해야 할 것
- 너무 추상적인 조언 말고, 구체적인 행동 단위로 설명해줘.

Output format (Markdown):

## 이번 변경 요약
- ...

## 추천 후속 작업
- [ ] 할 일 1
- [ ] 할 일 2
- [ ] 할 일 3

## 참고 메모 (선택)
- 필요한 경우에만 추가 설명
"""


def get_repo_root() -> Path:
    """현재 레포의 루트 경로를 반환."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def get_staged_diff(repo_root: Path) -> str:
    """현재 staged 된 diff(git diff --cached)를 문자열로 가져온다."""
    result = subprocess.run(
        ["git", "diff", "--cached"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def load_project_context(repo_root: Path) -> str:
    """
    PROJECT_CONTEXT.md 또는 PROJECT_CONTEXT.example.md가 있으면 읽어서 반환.
    없으면 빈 문자열.
    """
    for name in ("PROJECT_CONTEXT.md", "PROJECT_CONTEXT.example.md"):
        p = repo_root / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    return ""


def ask_llm(system_prompt: str, user_prompt: str) -> str:
    """
    OpenAI Chat Completion 호출.
    환경변수:
      - OPENAI_API_KEY (필수)
      - NEXT_ACTIONS_MODEL (선택, 기본 gpt-4.1-mini)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ OPENAI_API_KEY 환경변수가 설정되어 있지 않아 LLM을 호출하지 못했습니다."

    model = os.environ.get("NEXT_ACTIONS_MODEL", "gpt-4.1-mini")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def main() -> None:
    try:
        repo_root = get_repo_root()
    except Exception as e:
        print(f"[next_actions] git 레포 루트를 찾는 데 실패했습니다: {e}")
        return

    diff = get_staged_diff(repo_root)
    if not diff.strip():
        print("[next_actions] staged diff 없음 (git add 후에 실행됩니다).")
        return

    context = load_project_context(repo_root)

    user_prompt = f"""
프로젝트 설명:
----------------
{context}

아래는 이번 커밋의 staged git diff입니다.
이 변경을 기준으로, 개발자가 지금 바로 하면 좋을 후속 작업들을 추천해줘.

git diff:
----------------
```diff
{diff}"""

    text = ask_llm(SYSTEM_PROMPT, user_prompt)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path = repo_root / ".git" / "NEXT_ACTIONS.md"
    out_path.write_text(f"<!-- generated at {ts} -->\n\n{text}\n", encoding="utf-8")

    print("\n===== LLM 기반 후속 작업 추천 =====\n")
    print(text)
    print("\n결과는 .git/NEXT_ACTIONS.md 에도 저장되었습니다.\n")


if __name__ == "__main__":
    main()
