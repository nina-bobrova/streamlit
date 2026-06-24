"""
网页抓取模块 —— 使用 requests + BeautifulSoup 提取中文网页正文。
"""

import re
import requests
from bs4 import BeautifulSoup


def fetch_text(url: str, timeout: int = 10) -> str:
    """
    抓取指定 URL 的网页正文文本。

    处理流程：
    1. 发送 GET 请求，自动检测编码。
    2. 移除 <script>, <style>, <nav>, <footer>, <header> 等非正文标签。
    3. 提取可见文本，清洗多余空白。

    参数：
        url (str): 文章网页地址。
        timeout (int): 请求超时秒数。

    返回：
        str: 清洗后的正文文本；若失败则返回空字符串。

    异常：
        requests.RequestException: 网络请求失败时在调用方捕获。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    # 自动检测编码（避免乱码）
    if resp.apparent_encoding and resp.apparent_encoding.lower() != "ascii":
        resp.encoding = resp.apparent_encoding
    elif resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
        # 回退方案：尝试从 HTML meta 标签中提取 charset
        match = re.search(
            r'<meta[^>]+charset=["\']?([^"\'>\s]+)', resp.text[:4096], re.IGNORECASE
        )
        if match:
            resp.encoding = match.group(1)
        else:
            resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "html.parser")

    # 移除噪声标签
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # 提取纯文本，用空格连接，合并空白行
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text: str) -> str:
    """
    判断文本中中文和英文的占比，返回 'zh' / 'en' / 'mixed'。
    简单按中文字符比例阈值来判定。
    """
    if not text:
        return "en"

    chinese_chars = len(re.findall(r"[一-鿿]", text))
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    total = chinese_chars + english_words
    if total == 0:
        return "en"
    ratio = chinese_chars / total
    if ratio > 0.6:
        return "zh"
    elif ratio < 0.3:
        return "en"
    else:
        return "mixed"
