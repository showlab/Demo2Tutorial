# critic_impl.py
import json
import logging
import os

from .openai_compat import chat_completion_create

logger = logging.getLogger(__name__)

LLM_RUBRIC = """
You are a strict, software-agnostic tutorial critic. Evaluate ONLY from the draft JSON content.
Score 0..1 on four dimensions:
1) Coverage: each chapter has actions; includes a closing/confirmation step (enter/apply/ok/save/finish/submit);
   at least 2 chapters unless the task is trivially short.
2) Granularity: steps are atomic with action verbs (click/press/type/drag/select/set/open/choose/toggle/confirm/enter/apply/save/ok/finish/submit);
   instruction length is not too short/long; include explicit confirmation after parameter edits.
3) Orderliness: time order is consistent (t_start/t_end monotonic, minimal overlaps); avoid reverting to earlier phase
   (don’t change parameters after a final confirmation unless stated as correction).
4) Learnability: include UI anchors (panel/menu/field/button/toolbar/window/shortcut/dropdown/value/slider/tab/checkbox/input),
   and show at least one concrete value when editing parameters.

If you think overall >= 0.86 AND coverage >= 0.85 AND granularity >= 0.80 AND orderliness >= 0.85 AND learnability >= 0.80,
reply EXACTLY:
Looks good.
ELSE,
Return STRICT JSON with this schema ONLY WHEN THE SCORES DO NOT PASS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!:
{
  "scores": {
    "coverage": <float 0..1>,
    "granularity": <float 0..1>,
    "orderliness": <float 0..1>,
    "learnability": <float 0..1>,
    "overall": <float 0..1>
  },
  "suggestions": [<short actionable suggestion strings>]
}

Do NOT include any extra keys or text. Keep comments/suggestions concise.
Examine the scores you provide, I repeat, if the scores passed the threshold JUST REPLY 'Looks good',  ONLY return the json if they do NOT pass the threshold.
"""


def evaluate_tut_draft_llm(
    draft_path: str, model: str = "gpt-4.1", iter: int = 0
) -> str:
    """
    LLM-based, software-agnostic evaluation.
    Reads a tut_draft JSON file and returns a STRICT JSON string as per rubric.
    If iter > 5, returns "ALL GOOD" (就这样吧).
    """
    # 如果迭代超过5次，直接返回"就这样吧"
    if iter > 5:
        return "ALL GOOD"

    draft = json.load(open(draft_path, "r", encoding="utf-8"))
    draft_str = json.dumps(draft, ensure_ascii=False, indent=2)

    from openai import OpenAI

    logger.info("Critic: evaluating draft (iter %d) via %s", iter, model)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    total_critic_tokens = 0

    def ask_once() -> str:
        nonlocal total_critic_tokens
        resp = chat_completion_create(
            client,
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": LLM_RUBRIC},
                {"role": "user", "content": "Here is the tutorial draft JSON:"},
                {"role": "user", "content": draft_str},
                {
                    "role": "user",
                    "content": "Now evaluate and return ONLY the required content.",
                },
            ],
        )
        if resp.usage:
            total_critic_tokens += resp.usage.total_tokens
            logger.info("Critic: token usage %d", resp.usage.total_tokens)
        return resp.choices[0].message.content

    txt = ask_once()

    # 首先检查是否是"Looks good."这样的通过消息
    if "looks good" in txt.lower().strip():
        return "ALL GOOD"

    try:
        # 保证严格 JSON
        parsed = json.loads(txt)

        # 额外的后处理：检查分数是否达标，如果达标强制返回"ALL GOOD"
        scores = parsed.get("scores", {})
        overall = scores.get("overall", 0)
        coverage = scores.get("coverage", 0)
        granularity = scores.get("granularity", 0)
        orderliness = scores.get("orderliness", 0)
        learnability = scores.get("learnability", 0)

        # 阈值判断
        if (
            overall >= 0.8
            and coverage >= 0.8
            and granularity >= 0.80
            and orderliness >= 0.8
            and learnability >= 0.80
        ):
            return "ALL GOOD"

        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        # 自修复：如果第一轮不是合法 JSON，再请求一次"仅输出 JSON"
        repair_resp = chat_completion_create(
            client,
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Output VALID JSON ONLY. No prose."},
                {"role": "user", "content": txt},
            ],
        )
        if repair_resp.usage:
            total_critic_tokens += repair_resp.usage.total_tokens
            logger.info("Critic: repair token usage %d", repair_resp.usage.total_tokens)
        repair = repair_resp.choices[0].message.content
        parsed2 = json.loads(repair)  # 若再失败会抛异常，方便你在上层捕获

        # 同样检查修复后的JSON分数
        scores = parsed2.get("scores", {})
        overall = scores.get("overall", 0)
        coverage = scores.get("coverage", 0)
        granularity = scores.get("granularity", 0)
        orderliness = scores.get("orderliness", 0)
        learnability = scores.get("learnability", 0)

        if (
            overall >= 0.85
            and coverage >= 0.85
            and granularity >= 0.80
            and orderliness >= 0.85
            and learnability >= 0.80
        ):
            return "ALL GOOD"

        return json.dumps(parsed2, ensure_ascii=False, indent=2)
