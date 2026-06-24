"""
Streamlit 中文网页文本词频分析与可视化系统
============================================
用户输入文章 URL -> 自动抓取正文 -> 分词 -> 统计词频 -> 多图表交互展示。
"""

import sys
import os
import base64
from typing import List, Tuple, Dict, Any

# 从环境变量读取 API Key（未设置时使用占位符，需用户自行配置）
_ENV_DOUBAO_KEY = os.environ.get("DOUBAO_API_KEY", "your-doubao-api-key")
_ENV_DOUBAO_TURBO_MODEL = os.environ.get("DOUBAO_TURBO_MODEL", "ep-xxxxxxxxxxxx-xxxxx")
_ENV_DOUBAO_MINI_MODEL = os.environ.get("DOUBAO_MINI_MODEL", "ep-xxxxxxxxxxxx-xxxxx")

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch import fetch_text, detect_language
from tokenize_words import tokenize_and_count
from ai_analysis import generate_summary, chat_query


# ======================== 页面配置 ========================
st.set_page_config(
    page_title="中文词频分析与可视化系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    hr { margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ======================== 常量 ========================
STOPWORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwords_cn.txt")

AI_PROVIDERS = {
    "doubao": {
        "display": "Doubao-Seed-2.1-turbo",
        "api_key": _ENV_DOUBAO_KEY,
        "base_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "model": _ENV_DOUBAO_TURBO_MODEL,
    },
    "mini": {
        "display": "Doubao-Seed-2.0-mini",
        "api_key": _ENV_DOUBAO_KEY,
        "base_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "model": _ENV_DOUBAO_MINI_MODEL,
    },
}

CHART_TYPES: Dict[str, str] = {
    "wordcloud": "词云图",
    "bar":       "柱状图",
    "line":      "折线图",
    "pie":       "饼图",
    "scatter":   "散点图",
    "funnel":    "漏斗图",
    "radar":     "雷达图",
    "treemap":   "矩形树图",
}


# ======================== 工具函数 ========================
def get_table_download_link(df: pd.DataFrame, filename: str = "word_freq.csv") -> str:
    """生成 CSV 下载链接的 HTML。"""
    csv_data = df.to_csv(index=False, encoding="utf-8-sig")
    b64 = base64.b64encode(csv_data.encode("utf-8-sig")).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" '
        f'download="{filename}" '
        f'style="text-decoration:none;padding:8px 16px;'
        f'background-color:#4CAF50;color:white;border-radius:4px;">'
        f'下载 CSV</a>'
    )
    return href


# ======================== pyecharts 图表渲染 ========================
try:
    from pyecharts import options as opts
    from pyecharts.charts import (
        Bar, Line, Pie, Scatter, Funnel, Radar, WordCloud, TreeMap,
    )
    PYECHARTS_AVAILABLE = True
except ImportError:
    PYECHARTS_AVAILABLE = False


def _render_pyecharts_chart(chart) -> None:
    """将 pyecharts 图表渲染为 HTML 并嵌入 Streamlit。"""
    from streamlit.components.v1 import html as st_html
    st_html(chart.render_embed(), height=520, scrolling=False)


def render_wordcloud(data: List[Tuple[str, int]]) -> None:
    """词云图"""
    if not PYECHARTS_AVAILABLE:
        st.warning("pyecharts 未安装，无法渲染图表")
        return
    wordcloud_data = [(word, freq) for word, freq in data]
    wc = (
        WordCloud()
        .add(
            series_name="词频",
            data_pair=wordcloud_data,
            word_size_range=[16, 100],
            shape="circle",
            textstyle_opts=opts.TextStyleOpts(font_family="Microsoft YaHei"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词云图", subtitle="词频越高，字体越大"),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
        )
    )
    _render_pyecharts_chart(wc)


def render_bar(data: List[Tuple[str, int]], title: str = "词频柱状图") -> None:
    """柱状图"""
    if not PYECHARTS_AVAILABLE:
        return
    words = [item[0] for item in data]
    freqs = [item[1] for item in data]

    bar = (
        Bar()
        .add_xaxis(words)
        .add_yaxis("词频", freqs, color="rgb(64,144,247)")
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title, subtitle=f"共 {len(data)} 个词汇"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
            yaxis_opts=opts.AxisOpts(name="词频"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=True, position="top"),
            markpoint_opts=opts.MarkPointOpts(
                data=[opts.MarkPointItem(type_="max", name="最高")]
            ),
        )
    )
    _render_pyecharts_chart(bar)


def render_line(data: List[Tuple[str, int]]) -> None:
    """折线图"""
    if not PYECHARTS_AVAILABLE:
        return
    words = [item[0] for item in data]
    freqs = [item[1] for item in data]

    line = (
        Line()
        .add_xaxis(words)
        .add_yaxis(
            "词频",
            freqs,
            is_smooth=True,
            color="rgb(255,107,107)",
            linestyle_opts=opts.LineStyleOpts(width=3),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.15),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词频趋势折线图"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
            yaxis_opts=opts.AxisOpts(name="词频"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
    )
    _render_pyecharts_chart(line)


def render_pie(data: List[Tuple[str, int]]) -> None:
    """饼图"""
    if not PYECHARTS_AVAILABLE:
        return
    pie_data = [(word, freq) for word, freq in data[:15]]

    pie = (
        Pie()
        .add(
            series_name="词频占比",
            data_pair=pie_data,
            radius=["35%", "70%"],
            rosetype="radius",
            label_opts=opts.LabelOpts(
                is_show=True,
                formatter="{b}: {d}%",
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词频占比饼图"),
            legend_opts=opts.LegendOpts(
                orient="vertical", pos_top="middle", pos_right="5%"
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{b}: {c} ({d}%)"
            ),
        )
    )
    _render_pyecharts_chart(pie)


def render_scatter(data: List[Tuple[str, int]]) -> None:
    """散点图"""
    if not PYECHARTS_AVAILABLE:
        return

    words = [item[0] for item in data]
    freqs = [item[1] for item in data]

    scatter = (
        Scatter()
        .add_xaxis(words)
        .add_yaxis(
            "词频散点",
            freqs,
            symbol_size=14,
            itemstyle_opts=opts.ItemStyleOpts(color="rgb(147,112,219)"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词频散点图", subtitle="按词汇排名分布"),
            xaxis_opts=opts.AxisOpts(
                name="词汇", axislabel_opts=opts.LabelOpts(rotate=-45)
            ),
            yaxis_opts=opts.AxisOpts(name="词频"),
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{b}: 词频 {c}"
            ),
        )
    )
    _render_pyecharts_chart(scatter)


def render_funnel(data: List[Tuple[str, int]]) -> None:
    """漏斗图"""
    if not PYECHARTS_AVAILABLE:
        return
    funnel_data = [(word, freq) for word, freq in data[:20]]

    funnel = (
        Funnel()
        .add(
            series_name="词频",
            data_pair=funnel_data,
            gap=2,
            label_opts=opts.LabelOpts(position="inside", formatter="{b}: {c}"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词频漏斗图"),
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{b}: {c}次"
            ),
        )
    )
    _render_pyecharts_chart(funnel)


def render_radar(data: List[Tuple[str, int]]) -> None:
    """雷达图"""
    if not PYECHARTS_AVAILABLE:
        return
    top6 = data[:6]
    if len(top6) < 3:
        st.info("至少需要 3 个词汇才能渲染雷达图")
        return

    words = [item[0] for item in top6]
    freqs = [item[1] for item in top6]

    max_freq = max(freqs) if freqs else 1
    schema = [
        opts.RadarIndicatorItem(name=w, max_=round(max_freq * 1.2))
        for w in words
    ]

    radar = (
        Radar()
        .add_schema(schema=schema, shape="polygon")
        .add(
            series_name="词频",
            data=[freqs],
            color="rgb(255,99,71)",
            areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
            linestyle_opts=opts.LineStyleOpts(width=2),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="雷达图", subtitle="Top-6 高频词（维度有限，仅展示前 6 个词）"
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
        )
    )
    _render_pyecharts_chart(radar)


def render_treemap(data: List[Tuple[str, int]]) -> None:
    """矩形树图"""
    if not PYECHARTS_AVAILABLE:
        return

    treemap_data = [opts.TreeItem(name=w, value=v) for w, v in data[:30]]

    tm = (
        TreeMap()
        .add(
            series_name="词频",
            data=treemap_data,
            leaf_depth=1,
            label_opts=opts.LabelOpts(position="inside", formatter="{b}\n{c}"),
            upper_label_opts=opts.LabelOpts(is_show=True, position="inside"),
            levels=[
                opts.TreeMapLevelsOpts(
                    treemap_itemstyle_opts=opts.TreeMapItemStyleOpts(
                        border_color="#fff", border_width=2, gap_width=1
                    )
                )
            ],
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="矩形树图", subtitle="面积越大，词频越高"),
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{b}: {c}次"
            ),
        )
    )
    _render_pyecharts_chart(tm)


# ======================== 图表调度器 ========================
RENDERERS: Dict[str, Any] = {
    "wordcloud": render_wordcloud,
    "bar":       render_bar,
    "line":      render_line,
    "pie":       render_pie,
    "scatter":   render_scatter,
    "funnel":    render_funnel,
    "radar":     render_radar,
    "treemap":   render_treemap,
}


# ======================== UI 布局 ========================
def render_sidebar(
    freq_data: List[Tuple[str, int]]
) -> Tuple[str, int, int]:
    """渲染侧边栏控件。"""
    with st.sidebar:
        st.markdown("""
        <style>
        .sb-chart  { background: #e8f5e9; padding: 16px; border-radius: 8px; margin-bottom: 12px; }
        .sb-filter { background: #e3f2fd; padding: 16px; border-radius: 8px; margin-bottom: 12px; }
        .sb-ai     { background: #fce4ec; padding: 16px; border-radius: 8px; margin-bottom: 12px; }
        .sb-title  { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-chart"><div class="sb-title">图表设置</div>', unsafe_allow_html=True)
        chart_type = st.selectbox(
            "选择图表类型",
            options=list(CHART_TYPES.keys()),
            format_func=lambda x: CHART_TYPES[x],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-filter"><div class="sb-title">过滤设置</div>', unsafe_allow_html=True)

        max_freq = max([f for _, f in freq_data]) if freq_data else 50
        slider_max = max(2, min(max_freq + 5, 100))

        min_freq = st.slider(
            "最低词频阈值",
            min_value=1,
            max_value=slider_max,
            value=2,
            step=1,
            help="过滤掉出现次数低于此阈值的词汇",
        )

        top_n = st.slider(
            "展示词汇数量 (Top-N)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="图表和数据表格中显示的词汇数量",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-ai"><div class="sb-title">AI 分析设置</div>', unsafe_allow_html=True)
        st.session_state["ai_enabled"] = st.checkbox("启用 AI 分析", value=st.session_state["ai_enabled"])

        if st.session_state["ai_enabled"]:
            st.session_state["ai_provider"] = st.selectbox(
                "AI 引擎",
                options=list(AI_PROVIDERS.keys()),
                format_func=lambda x: AI_PROVIDERS[x]["display"],
                index=list(AI_PROVIDERS.keys()).index(st.session_state["ai_provider"]),
            )
        st.markdown('</div>', unsafe_allow_html=True)

    return chart_type, min_freq, top_n


def render_stats_bar(freq_data: List[Tuple[str, int]]) -> None:
    """渲染顶部统计指标栏。"""
    cols = st.columns(4)
    with cols[0]:
        st.metric("总词汇数（去重）", len(freq_data))
    with cols[1]:
        total_occurrences = sum(f for _, f in freq_data)
        st.metric("总词次数", total_occurrences)
    with cols[2]:
        avg_freq = round(total_occurrences / max(len(freq_data), 1), 1)
        st.metric("平均词频", avg_freq)
    with cols[3]:
        top_word = freq_data[0][0] if freq_data else "-"
        st.metric("最高频词", top_word)


def render_data_table(freq_data: List[Tuple[str, int]], top_n: int) -> pd.DataFrame:
    """渲染数据表格并返回 DataFrame。"""
    df = pd.DataFrame(freq_data[:top_n], columns=["词汇", "词频"])
    df.index = range(1, len(df) + 1)
    df.index.name = "排名"

    st.markdown("### 词频统计表")
    st.dataframe(
        df.style.background_gradient(subset=["词频"], cmap="Blues"),
        width="stretch",
        height=400,
    )
    return df


# ======================== 主入口 ========================
def main() -> None:
    """Streamlit 主入口函数。"""
    st.title("中文网页文本词频分析与可视化系统")
    st.markdown(
        "输入任意文章 URL，系统自动抓取正文、分词、统计词频，"
        "并通过词云、柱状图、折线图等多种图表进行交互式展示。"
    )

    # 初始化 session_state
    if "input_url" not in st.session_state:
        st.session_state["input_url"] = ""
    if "do_analyze" not in st.session_state:
        st.session_state["do_analyze"] = False
    if "pending_url" not in st.session_state:
        st.session_state["pending_url"] = None
    if "pending_demo_text" not in st.session_state:
        st.session_state["pending_demo_text"] = None
    if "ai_provider" not in st.session_state:
        st.session_state["ai_provider"] = "doubao"
    if "ai_summary" not in st.session_state:
        st.session_state["ai_summary"] = ""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "ai_enabled" not in st.session_state:
        st.session_state["ai_enabled"] = True
    if "_clear_chat_input" not in st.session_state:
        st.session_state["_clear_chat_input"] = False

    # 处理示例按钮发来的待填入 URL（必须在 text_input widget 创建之前设置）
    if st.session_state["pending_url"] is not None:
        st.session_state["input_url"] = st.session_state["pending_url"]
        st.session_state["pending_url"] = None

    # ---------- URL 输入 ----------
    col1, col2 = st.columns([4, 1])
    with col1:
        url = st.text_input(
            "文章 URL",
            key="input_url",
            placeholder="https://news.sina.com.cn/c/2024-01-01/doc-xxxxx.shtml",
            label_visibility="visible",
        )
    with col2:
        if st.button("开始分析", type="primary", width="stretch"):
            st.session_state["do_analyze"] = True

    # 示例文本（内置，无需联网即可演示）
    DEMO_TEXTS = {
        "人工智能": (
            "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个重要分支，"
            "主要研究如何使计算机能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。"
            "人工智能的研究领域包括机器学习、自然语言处理、计算机视觉、语音识别、机器人技术、"
            "专家系统、知识表示与推理、规划与决策等。机器学习是人工智能的核心驱动力，"
            "深度学习作为机器学习的一个子集，通过多层神经网络实现自动特征提取和模式识别，"
            "在图像分类、语音识别、自然语言理解等任务中取得了突破性进展。自然语言处理技术"
            "使得计算机能够理解、生成和翻译人类语言，广泛应用于智能助手、机器翻译、情感分析"
            "等场景。计算机视觉技术让机器能够识别和理解图像与视频内容，在自动驾驶、医疗影像"
            "诊断、安防监控等领域发挥着重要作用。人工智能正在深刻改变人们的生活和工作方式，"
            "推动各行各业的数字化转型。在医疗领域，人工智能辅助诊断系统能够帮助医生提高诊断"
            "准确率；在教育领域，个性化学习系统能够根据学生特点定制学习方案；在金融领域，"
            "智能风控系统能够实时监测和预防欺诈行为。但人工智能也带来了隐私保护、就业影响、"
            "算法偏见等伦理和社会问题，需要社会各界共同关注和应对。未来，人工智能将继续向着"
            "更通用、更可靠、更安全的方向发展，为人类创造更大的价值。"
        ),
        "Python编程": (
            "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年首次发布。"
            "它以简洁、易读的语法著称，强调代码的可读性和开发效率。Python支持多种编程范式，"
            "包括面向对象编程、函数式编程、过程式编程等。作为一种解释型语言，Python具有"
            "动态类型系统和自动内存管理功能。Python拥有丰富的标准库和第三方生态系统，"
            "涵盖了网络编程、数据库操作、图形界面开发、科学计算、数据分析等广泛领域。"
            "在数据科学领域，NumPy提供了高效的多维数组运算，Pandas简化了数据处理和分析流程，"
            "Matplotlib和Seaborn则用于数据可视化。在机器学习领域，Scikit-learn、TensorFlow、"
            "PyTorch等框架使得模型训练和部署变得简单高效。在Web开发方面，Django和Flask"
            "等框架支持快速构建可靠的网络应用。Python也被广泛用于自动化脚本、网络爬虫、"
            "系统管理等场景。Python的社区活跃且包容，拥有海量的学习资源和开源项目。"
            "Python的设计哲学强调优美胜于丑陋，显式胜于隐式，简单胜于复杂，复杂胜于凌乱。"
            "Python解释器易于扩展，可以使用C或C++编写扩展模块。Python的跨平台特性使得"
            "同一套代码可以在Windows、Linux、macOS等不同操作系统上运行。随着人工智能"
            "和大数据技术的蓬勃发展，Python已成为数据科学家和机器学习工程师的首选语言之一。"
        ),
        "机器学习": (
            "机器学习是人工智能的一个核心领域，研究如何通过经验自动改进计算机算法的性能。"
            "机器学习算法通过从数据中学习规律和模式，使计算机能够在没有明确编程指令的"
            "情况下做出预测或决策。机器学习主要分为监督学习、无监督学习、半监督学习和"
            "强化学习四大类型。监督学习使用带有标签的训练数据，常见算法包括线性回归、"
            "逻辑回归、支持向量机、决策树、随机森林和神经网络等。无监督学习处理没有标签"
            "的数据，用于聚类分析、降维和异常检测等任务。强化学习通过智能体与环境的交互"
            "来学习最优策略，在游戏AI、机器人控制和自动驾驶中有着广泛应用。深度学习的"
            "出现极大地推动了机器学习的发展，卷积神经网络在图像处理领域表现优异，循环神经"
            "网络和Transformer架构在序列数据处理方面取得了显著成果。迁移学习、联邦学习、"
            "对比学习等前沿技术不断拓展机器学习的边界。特征工程是机器学习中至关重要的环节，"
            "好的特征能够显著提升模型性能。模型评估与选择同样关键，交叉验证、混淆矩阵、"
            "ROC曲线等工具帮助评估模型质量。超参数调优和正则化技术用于防止过拟合。"
            "在实际应用中，机器学习已广泛应用于推荐系统、搜索引擎、广告投放、信用评估、"
            "医疗诊断、语音助手等场景。数据质量、模型可解释性和公平性是机器学习实践中"
            "需要重点关注的挑战。"
        ),
    }

    st.caption("试试这些示例文本：")
    demo_cols = st.columns(len(DEMO_TEXTS))
    for i, (label, demo_text) in enumerate(DEMO_TEXTS.items()):
        with demo_cols[i]:
            if st.button(label, key=f"demo_{i}", width="stretch"):
                st.session_state["pending_demo_text"] = demo_text
                st.session_state["do_analyze"] = True
                st.rerun()

    target_url = url.strip()
    do_analyze = st.session_state["do_analyze"]
    has_results = "freq_data_raw" in st.session_state and st.session_state["freq_data_raw"]

    # 既没有触发分析，也没有缓存结果 → 展示首页
    if not do_analyze and not has_results:
        if not target_url:
            st.info("请在上方输入文章 URL，然后点击「开始分析」按钮")
        _show_intro()
        return

    # ---------- 执行分析（仅在点击"开始分析"或示例按钮时触发）----------
    if do_analyze:
        st.session_state["do_analyze"] = False

        if "pending_demo_text" in st.session_state and st.session_state["pending_demo_text"]:
            raw_text = st.session_state.pop("pending_demo_text")
            st.success(f"已加载示例文本（{len(raw_text):,} 个字符）")
        elif target_url:
            with st.spinner("正在抓取网页内容..."):
                try:
                    raw_text = fetch_text(target_url, timeout=15)
                except Exception as e:
                    st.error(f"网页抓取失败：{e}")
                    st.info(
                        "可能的原因：\n"
                        "- 网络连接不稳定\n"
                        "- URL 无法访问\n"
                        "- 网站有反爬机制\n\n"
                        "请检查 URL 是否正确，或尝试其他链接。"
                    )
                    return

            if not raw_text or len(raw_text) < 20:
                st.error("抓取到的文本内容过短，可能不是有效的文章页面。")
                return

            st.success(f"成功抓取 {len(raw_text):,} 个字符的文本内容")
        else:
            st.warning("请输入有效的 URL")
            return

        lang = detect_language(raw_text)
        st.caption(f"检测到语言类型：{lang}（zh=中文，en=英文，mixed=中英混合）")

        with st.spinner("正在进行中文分词..."):
            try:
                freq_data = tokenize_and_count(
                    raw_text,
                    min_freq=1,
                    top_n=0,
                    stopwords_file=STOPWORDS_FILE,
                )
            except Exception as e:
                st.error(f"分词失败：{e}")
                return

        if not freq_data:
            st.error("未能提取到有效词汇，请检查网页内容是否包含中文文本。")
            return

        st.session_state["freq_data_raw"] = freq_data
        st.session_state["raw_text"] = raw_text
        # 新数据 → 清除旧的 AI 结果
        st.session_state["ai_summary"] = ""
        st.session_state["chat_history"] = []
        st.session_state["_clear_chat_input"] = True

    # ---------- 展示结果（有缓存数据时始终渲染）----------
    freq_data = st.session_state.get("freq_data_raw")
    if not freq_data:
        st.warning("暂无分析数据")
        return

    # ---------- 侧边栏 ----------
    chart_type, min_freq, top_n = render_sidebar(freq_data=freq_data)

    # ---------- 应用过滤 ----------
    filtered_data = [(w, f) for w, f in freq_data if f >= min_freq]
    display_data = filtered_data[:top_n]

    if not display_data:
        st.warning("当前过滤条件下没有词汇，请降低最低词频阈值。")
        return

    # ---------- 统计指标 ----------
    render_stats_bar(filtered_data)

    # ---------- AI 分析摘要 ----------
    if st.session_state.get("ai_enabled") and AI_PROVIDERS.get(st.session_state.get("ai_provider", "")):
        with st.expander("AI 分析解读", expanded=False):
            if st.button("生成分析摘要", key="btn_ai_summary"):
                with st.spinner("AI 正在分析词频数据..."):
                    try:
                        provider = AI_PROVIDERS[st.session_state["ai_provider"]]
                        st.session_state["ai_summary"] = generate_summary(
                            freq_data,
                            api_key=provider["api_key"],
                            base_url=provider["base_url"],
                            model=provider["model"],
                        )
                    except Exception as e:
                        st.session_state["ai_summary"] = f"（AI 分析失败：{e}）"

            if st.session_state.get("ai_summary"):
                st.markdown(st.session_state["ai_summary"])
                if st.button("重新生成", key="btn_ai_regenerate"):
                    st.session_state["ai_summary"] = ""
                    st.rerun()
            else:
                st.caption("点击上方按钮让 AI 分析当前词频数据")

    # ---------- AI 对话 ----------
    if st.session_state.get("ai_enabled") and AI_PROVIDERS.get(st.session_state.get("ai_provider", "")):
        st.markdown("### AI 对话")

        # 显示历史对话
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

        # 处理清除输入框的请求（在 widget 创建之前删除 key）
        if st.session_state.get("_clear_chat_input"):
            st.session_state.pop("chat_input", None)
            st.session_state["_clear_chat_input"] = False

        # 输入区
        col_input, col_btn, col_clear = st.columns([6, 1, 1])
        with col_input:
            user_question = st.text_input(
                "输入问题",
                key="chat_input",
                placeholder="基于分析结果提问...",
                label_visibility="collapsed",
            )
        with col_btn:
            send_clicked = st.button("发送", key="btn_send", width="stretch")
        with col_clear:
            if st.button("清空", key="btn_clear_chat", width="stretch"):
                st.session_state["chat_history"] = []
                st.rerun()

        if send_clicked and user_question.strip():
            st.session_state["chat_history"].append({"role": "user", "content": user_question})
            with st.spinner("AI 思考中..."):
                try:
                    provider = AI_PROVIDERS[st.session_state["ai_provider"]]
                    reply = chat_query(
                        freq_data,
                        question=user_question,
                        chat_history=st.session_state["chat_history"],
                        api_key=provider["api_key"],
                        base_url=provider["base_url"],
                        model=provider["model"],
                    )
                    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"AI 回复失败：{e}")
                    st.session_state["chat_history"].pop()
            # 标记需要清除输入框（在下次渲染前由 pending 机制处理）
            st.session_state["_clear_chat_input"] = True
            st.rerun()

    # ---------- 图表区 ----------
    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.markdown(f"### {CHART_TYPES[chart_type]}")
        with st.spinner("正在渲染图表..."):
            try:
                renderer = RENDERERS.get(chart_type)
                if renderer:
                    renderer(display_data)
                else:
                    st.error(f"未知图表类型：{chart_type}")
            except Exception as e:
                st.error(f"图表渲染失败：{e}")
                st.info("请尝试切换图表类型或调整过滤参数。")

    with col_table:
        df = render_data_table(filtered_data, top_n)

        st.markdown("### 数据导出")
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
        st.caption("下载词频统计 CSV 文件，可用 Excel 打开。")

    st.caption("基于 Streamlit + jieba + pyecharts 构建")


def _show_intro() -> None:
    """展示系统介绍和使用说明。"""
    st.markdown("## 系统介绍")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ### 功能特性
            - 自动抓取：输入 URL 自动提取网页正文
            - 智能分词：基于 jieba 的中文分词引擎
            - 停用词过滤：内置中文停用词表
            - 8 种图表：词云、柱状图、折线图、饼图、散点图、漏斗图、雷达图、矩形树图
            - 实时过滤：滑块控制最低词频和展示数量
            - 数据导出：一键下载词频统计 CSV
            """
        )
    with col2:
        st.markdown(
            """
            ### 使用步骤
            1. 在顶部输入框粘贴文章 URL
            2. 点击「开始分析」按钮
            3. 等待抓取和分词完成
            4. 在左侧边栏切换图表类型
            5. 拖动滑块过滤低频词
            6. 下载词频数据 CSV

            ### 技术栈
            | 层次 | 技术 |
            |------|------|
            | Web | Streamlit |
            | 抓取 | requests + BS4 |
            | 分词 | jieba |
            | 图表 | pyecharts |
            | 数据 | pandas |
            """
        )


if __name__ == "__main__":
    main()
