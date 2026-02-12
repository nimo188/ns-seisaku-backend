# app.py（履歴対応 + スムーズ質問生成）
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ====== 設定 ======
BASE_DIR = Path(__file__).resolve().parent

PROJECT_CSV_PATH = os.getenv("PROJECT_CSV_PATH", str(BASE_DIR / "project_list_nagaoka.csv"))
MEMBER_CSV_PATH = os.getenv("MEMBER_CSV_PATH", str(BASE_DIR / "northsand_member_list.csv"))

TOP_K = int(os.getenv("PROJECT_SEARCH_TOP_K", "5"))
MIN_SCORE = float(os.getenv("PROJECT_SEARCH_MIN_SCORE", "0.18"))

# --- プロジェクト履歴CSV列 ---
COL_EMP = "社員"
COL_NAME = "プロジェクト名"
COL_DESC = "プロジェクト概要"
COL_SITE = "現場"
COL_NOTION = "Notionリンク"

# --- メンバーCSV列（固定） ---
MEM_NAME = "氏名"
MEM_ROLE = "職位"
MEM_LOC = "ロケーション"
MEM_SITE = "現場"
MEM_SUMMARY = "経歴要約"
MEM_NOTION = "Notionリンク"


def _read_csv_flexible(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def _norm_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("　", " ")
    s = re.sub(r"\s+", "", s)
    return s.strip().lower()


def _to_md_link(url: str, label: str = "Notionリンク") -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if u.startswith("[") and "](" in u and u.endswith(")"):
        return u
    return f"[{label}]({u})"


def _to_md_links(urls: List[str], label: str = "Notionリンク") -> str:
    uniq: List[str] = []
    seen = set()
    for u in urls:
        uu = (u or "").strip()
        if not uu or uu in seen:
            continue
        seen.add(uu)
        uniq.append(uu)

    if not uniq:
        return ""
    if len(uniq) == 1:
        return _to_md_link(uniq[0], label=label)
    return "\n".join([_to_md_link(u, label=f"{label}{i+1}") for i, u in enumerate(uniq)])


@lru_cache(maxsize=1)
def _load_and_build_project_index(csv_path: str):
    df = _read_csv_flexible(csv_path)

    if COL_NOTION not in df.columns:
        df[COL_NOTION] = ""

    for c in [COL_EMP, COL_NAME, COL_DESC, COL_SITE, COL_NOTION]:
        if c not in df.columns:
            raise ValueError(f"プロジェクトCSVに必要な列がありません: {c}. columns={list(df.columns)}")
        df[c] = df[c].fillna("")

    corpus = (df[COL_NAME] + " " + df[COL_DESC] + " " + df[COL_SITE]).tolist()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X = vectorizer.fit_transform(corpus)
    return df, vectorizer, X


@lru_cache(maxsize=1)
def _load_member_index(csv_path: str) -> Dict[str, Dict[str, str]]:
    df = _read_csv_flexible(csv_path)

    for c in [MEM_NAME, MEM_ROLE, MEM_LOC, MEM_SITE, MEM_SUMMARY, MEM_NOTION]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    idx: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        name = str(row.get(MEM_NAME, "") or "")
        key = _norm_name(name)
        if not key:
            continue
        idx[key] = {
            MEM_NAME: name,
            MEM_ROLE: str(row.get(MEM_ROLE, "") or ""),
            MEM_LOC: str(row.get(MEM_LOC, "") or ""),
            MEM_SITE: str(row.get(MEM_SITE, "") or ""),
            MEM_SUMMARY: str(row.get(MEM_SUMMARY, "") or ""),
            MEM_NOTION: str(row.get(MEM_NOTION, "") or ""),
        }
    return idx


def _lookup_member(emp_name: str) -> Optional[Dict[str, str]]:
    idx = _load_member_index(MEMBER_CSV_PATH)
    return idx.get(_norm_name(emp_name))


@tool
def search_projects_and_people(query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> Dict[str, Any]:
    """
    1) プロジェクト履歴から類似案件（projects）を返す
    2) projects に登場した社員をメンバー一覧CSVで突合し、人物情報（people）を返す
    """
    df, vectorizer, X = _load_and_build_project_index(PROJECT_CSV_PATH)

    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    idx_sorted = sims.argsort()[::-1][: max(top_k, 1)]

    projects: List[Dict[str, Any]] = []
    people_map: Dict[str, Dict[str, Any]] = {}

    for i in idx_sorted:
        score = float(sims[i])
        if score < min_score:
            continue

        row = df.iloc[int(i)]
        emp = str(row.get(COL_EMP, "") or "")
        proj_name = str(row.get(COL_NAME, "") or "")
        proj_desc = str(row.get(COL_DESC, "") or "")
        proj_site = str(row.get(COL_SITE, "") or "")
        proj_notion = str(row.get(COL_NOTION, "") or "")

        projects.append(
            {
                "score": round(score, 4),
                "社員": emp,
                "プロジェクト名": proj_name,
                "プロジェクト概要": proj_desc,
                "現場": proj_site,
                "Notionリンク": proj_notion,
            }
        )

        key = _norm_name(emp)
        if not key:
            continue

        if key not in people_map:
            mem = _lookup_member(emp)
            member_notion = (mem.get(MEM_NOTION, "") if mem else "").strip()

            people_map[key] = {
                "社員": emp,
                "氏名": (mem.get(MEM_NAME) if mem else emp),
                "職位": (mem.get(MEM_ROLE, "") if mem else ""),
                "ロケーション": (mem.get(MEM_LOC, "") if mem else ""),
                "現場": (mem.get(MEM_SITE, "") if mem else ""),
                "経歴要約": (mem.get(MEM_SUMMARY, "") if mem else ""),
                "Notionリンク": member_notion,
                "related_projects": [],
                "_notion_candidates": [],
            }

        if proj_name and proj_name not in people_map[key]["related_projects"]:
            people_map[key]["related_projects"].append(proj_name)

        if proj_notion:
            people_map[key]["_notion_candidates"].append(proj_notion)

    projects = projects[:3]
    top_people_keys = {_norm_name(p["社員"]) for p in projects if _norm_name(p["社員"])}

    people: List[Dict[str, Any]] = []
    for k, v in people_map.items():
        if k not in top_people_keys:
            continue

        notion_urls: List[str] = []
        if v.get("Notionリンク", "").strip():
            notion_urls.append(v["Notionリンク"])
        notion_urls.extend(v.get("_notion_candidates", []))
        v["Notionリンク"] = _to_md_links(notion_urls, label="Notionリンク")
        v.pop("_notion_candidates", None)
        people.append(v)

    return {"projects": projects, "people": people}


# ★ 追加：ざっくり判定（簡易ヒューリスティック）
def _is_vague_question(q: str) -> bool:
    qq = (q or "").strip()
    if len(qq) < 12:
        return True
    vague_markers = [
        "教えて", "どうすれば", "困って", "相談", "何が", "どんな", "できますか",
        "うまくいかない", "エラー", "原因", "わからない"
    ]
    hit = sum(1 for m in vague_markers if m in qq)
    # マーカーが多くても、固有情報(例: サービス名/数値/ログっぽさ)があれば曖昧扱いを弱める
    has_specific = bool(re.search(r"[A-Za-z0-9_/:-]{4,}", qq)) or ("AWS" in qq) or ("OpenSearch" in qq)
    return (hit > 0) and (not has_specific)


# ★ SYSTEM_PROMPT 改修：質問文生成＆追加質問
SYSTEM_PROMPT = """
あなたは社内のプロジェクト履歴（CSV）とメンバー一覧（CSV）を参照して回答するアシスタントです。
必ず search_projects_and_people ツールを使って検索してください。

重要:
- ツールに渡す query は「ユーザーの最新の質問（latest_question）」を使うこと。会話履歴全文を query に入れない。
- 会話履歴（chat_history）は、用語の省略・前提・制約の補完、追加ヒアリングの重複回避にのみ使う。

出力ルール:
- ツール結果の projects が空なら、必ず「類似プロジェクトがありませんでした。」とだけ返す。

- projects がある場合、必ず次の順番・構成で出力する（順番厳守）:

【類似プロジェクト（プロジェクト履歴）】
上位1〜3件について、各件で必ず
- 社員
- プロジェクト名
- プロジェクト概要
- 現場
を箇条書きで出す。

【このプロジェクト履歴にマッチした人はこのような人です】
people を社員ごとに箇条書きで出し、必ず
- 氏名
- 職位
- ロケーション
- 現場
- 経歴要約
- Notionリンク（クリックできるリンクとして表示する）
を出す。
（補足として related_projects を1行で添えてよい）

【次に聞くなら（コピペ用）】
類似PJがヒットした場合、質問者がそのままTeams等で送れる「短い質問文」を3案、箇条書きで作成する。
- 宛先（氏名）と、相手の関連プロジェクト名を自然に引用する
- ユーザーの最新質問（latest_question）を要約して入れる
- 依頼形（〜教えていただけますか／〜可能でしょうか）で、1メッセージで完結する
- 不足情報がある場合は「前提が揃った版（プレースホルダ付き）」にする

【追加で確認したいこと】
latest_question が「ざっくり」なら、先にユーザーへ追加質問を2〜5個だけ箇条書きで返す。
- 追加質問は、回答者に聞く前にユーザー自身が埋めるべき情報（目的/制約/環境/期限/期待アウトプット/既に試したこと/エラー内容）に寄せる
- 追加質問を出した場合でも【次に聞くなら（コピペ用）】はプレースホルダ付きで出す

文章は簡潔に。
"""


def convert_event(event) -> Optional[Dict[str, Any]]:
    try:
        if not hasattr(event, "get"):
            return None

        inner = event.get("event")
        if not inner:
            return None

        cbd = inner.get("contentBlockDelta")
        if cbd:
            delta = (cbd.get("delta") or {})
            text = delta.get("text")
            if text:
                return {"type": "text", "data": text}

        cbs = inner.get("contentBlockStart")
        if cbs:
            start = (cbs.get("start") or {})
            tool_use = start.get("toolUse")
            if tool_use:
                tool_name = tool_use.get("name", "unknown")
                return {"type": "tool_use", "tool_name": tool_name}

        return None
    except Exception:
        return None


agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    tools=[search_projects_and_people],
    model="jp.anthropic.claude-haiku-4-5-20251001-v1:0",
)

app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload, context):
    payload = payload or {}
    latest_question = (payload.get("prompt") or "").strip()

    # ★ 追加：履歴受け取り（[{role, content}]）
    history = payload.get("history") or []
    if not isinstance(history, list):
        history = []

    # 履歴を安全に整形（role/contentの最低限保証）
    lines: List[str] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant"):
            continue
        if not content:
            continue
        prefix = "ユーザー" if role == "user" else "アシスタント"
        lines.append(f"{prefix}: {content}")

    chat_history_text = "\n".join(lines[-40:])  # 念のため上限

    vague_flag = _is_vague_question(latest_question)

    # ★ 変更：agentに渡す入力は「履歴＋最新質問＋ざっくり判定」
    # ※ツール検索は latest_question を使うようSYSTEM_PROMPTで明示済み
    combined_prompt = f"""あなたへの入力データです。指示に従って回答してください。

<chat_history>
{chat_history_text}
</chat_history>

<latest_question>
{latest_question}
</latest_question>

<is_vague_question>
{str(vague_flag).lower()}
</is_vague_question>
"""

    emitted_any = False
    async for event in agent.stream_async(combined_prompt):
        converted = convert_event(event)
        if converted:
            emitted_any = True
            yield converted

    if not emitted_any:
        result = agent(combined_prompt)
        text = getattr(result, "message", None) or str(result)
        yield {"type": "text", "data": text}


if __name__ == "__main__":
    app.run()
