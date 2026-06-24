"""
AI 分析模块 —— 调用 OpenAI 兼容 API，对词频统计结果进行智能解读和对话。
"""

import json
from typing import List, Tuple, Optional, Dict, Any

import requests


# 默认 API 配置
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """你是一个专业的文本分析助手。用户会提供一篇文章的词频统计结果，请你：
1. 根据高频词汇推断文章的主题和核心内容
2. 分析词汇之间的关联，找出关键概念
3. 用简洁清晰的中文给出分析结论
4. 如果用户追问，结合词频数据给出有针对性的回答

回答时注意：
- 直接给出分析，不要重复用户提供的数据
- 控制在 200 字以内
- 语气专业但平易近人"""


def _build_context(freq_data: List[Tuple[str, int]], top_n: int = 30) -> str:
    """将词频数据构建为 LLM 可理解的文本上下文。"""
    if not freq_data:
        return "（无词频数据）"

    total_unique = len(freq_data)
    total_occurrences = sum(f for _, f in freq_data)
    top_words = freq_data[:top_n]

    lines = [
        f"词频统计概要：",
        f"- 去重词汇总数：{total_unique}",
        f"- 总词次数：{total_occurrences}",
        f"- 平均词频：{round(total_occurrences / max(total_unique, 1), 1)}",
        f"",
        f"Top-{len(top_words)} 高频词：",
    ]
    for i, (word, count) in enumerate(top_words, 1):
        lines.append(f"  {i}. {word}（{count}次）")

    return "\n".join(lines)


def _call_api(
    messages: List[Dict[str, str]],
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = 30,
) -> str:
    """调用 OpenAI 兼容 Chat Completions API。"""
    # 兼容完整路径的端点（如豆包 Ark），否则拼接 /v1/chat/completions
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        url = base
    else:
        url = f"{base}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_summary(
    freq_data: List[Tuple[str, int]],
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    根据词频数据生成 AI 分析摘要。

    参数：
        freq_data: 词频列表 [(word, count), ...]
        api_key: API 密钥
        base_url: API 端点
        model: 模型名称

    返回：
        AI 生成的分析摘要文本
    """
    context = _build_context(freq_data)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"请根据以下词频统计结果，分析文章的主题和核心内容：\n\n{context}"
            ),
        },
    ]
    return _call_api(messages, api_key, base_url, model)


def chat_query(
    freq_data: List[Tuple[str, int]],
    question: str,
    chat_history: List[Dict[str, str]],
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    处理用户追问，结合词频数据和对话历史给出回答。

    参数：
        freq_data: 词频列表
        question: 用户当前问题
        chat_history: 之前的对话历史
        api_key: API 密钥
        base_url: API 端点
        model: 模型名称

    返回：
        AI 回复文本
    """
    context = _build_context(freq_data)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 加入词频上下文（作为第一条 user 消息）
    messages.append({
        "role": "user",
        "content": f"以下是一篇文章的词频统计结果，请记住这些数据以便回答后续问题：\n\n{context}",
    })
    messages.append({
        "role": "assistant",
        "content": "已了解词频数据，请提问。",
    })

    # 加入历史对话（跳过第一轮上下文注入）
    for msg in chat_history:
        messages.append(msg)

    # 加入当前问题
    messages.append({"role": "user", "content": question})

    return _call_api(messages, api_key, base_url, model)
