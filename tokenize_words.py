"""
分词与词频统计模块 —— 使用 jieba 进行中文分词，Counter 进行词频计数。
"""

import re
import os
from collections import Counter
from typing import List, Tuple, Optional, Set

import jieba


# ---------- 默认停用词 ----------
_DEFAULT_STOPWORDS: Set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "所", "为", "所以", "因为", "但是", "然而", "而且", "虽然", "如果",
    "可以", "还是", "这个", "那个", "这些", "那些", "什么", "怎么", "怎样",
    "哪", "哪里", "哪个", "如何", "为什么", "是否", "已经", "正在", "将",
    "把", "被", "让", "从", "以", "之", "与", "及", "或", "但", "而",
    "且", "因", "虽", "于", "则", "其", "中", "对", "等", "能", "更",
    "又", "再", "才", "刚", "已", "曾", "将", "要", "应", "可", "便",
    "向", "用", "啊", "吧", "吗", "呢", "哦", "嗯", "哈", "呀", "哇",
    "么", "嘛", "噢", "呵", "啦", "咯", "哎", "嗨", "喂", "嘿", "哎哟",
    "的", "地", "得", "大", "小", "多", "少", "来", "去", "出", "进",
    "做", "作", "从事", "进行", "时间", "年", "月", "日", "时", "分", "秒",
    "该", "此", "每", "各", "某", "另", "别", "本", "前", "后", "左", "右",
    "内", "外", "旁", "附近", "点", "号", "第", "位", "种", "类", "样",
    "段", "篇", "条", "项", "次", "回", "遍", "下", "过", "完", "着", "了",
}


def _load_stopwords(filepath: str) -> Set[str]:
    """从文件加载停用词，每行一个词。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def tokenize_and_count(
    text: str,
    min_freq: int = 2,
    top_n: int = 100,
    stopwords_file: Optional[str] = None,
    extra_stopwords: Optional[List[str]] = None,
) -> List[Tuple[str, int]]:
    """
    对文本进行分词、过滤停用词、统计词频。

    参数：
        text (str): 输入文本。
        min_freq (int): 最低词频阈值，低于此值的词会被过滤。
        top_n (int): 返回前 N 个高频词（0 表示返回全部）。
        stopwords_file (Optional[str]): 自定义停用词文件路径。
        extra_stopwords (Optional[List[str]]): 额外需要过滤的词语。

    返回：
        List[Tuple[str, int]]: 按词频降序排列的 (词语, 频次) 列表。
    """
    if not text or not text.strip():
        return []

    # 合并停用词
    stopwords = _DEFAULT_STOPWORDS.copy()
    if stopwords_file and os.path.exists(stopwords_file):
        stopwords.update(_load_stopwords(stopwords_file))
    if extra_stopwords:
        stopwords.update(extra_stopwords)

    # 提取词汇正则：中文单字太短无意义，过滤掉单字
    words_raw: List[str] = []
    # 对中文，jieba 分词
    seg_list = jieba.cut(text)
    for word in seg_list:
        word = word.strip()
        if not word:
            continue
        # 过滤纯数字、纯符号、单字、停用词
        if len(word) < 2:
            continue
        if re.fullmatch(r"[\d\.\+\-\/\%\s]+", word):
            continue
        if word in stopwords:
            continue
        # 仅保留中文字符或英文单词
        if re.search(r"[一-鿿]|[a-zA-Z]", word):
            words_raw.append(word)

    # 统计词频
    counter = Counter(words_raw)

    # 过滤低频词
    if min_freq > 1:
        counter = Counter({k: v for k, v in counter.items() if v >= min_freq})

    # 按词频降序
    sorted_words = counter.most_common(top_n if top_n > 0 else None)

    return sorted_words


def get_top_words(
    freq_list: List[Tuple[str, int]], n: int = 20
) -> List[Tuple[str, int]]:
    """获取词频列表中的前 n 个高频词。"""
    return freq_list[:n]
