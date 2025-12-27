import streamlit as st
import requests
from bs4 import BeautifulSoup
import jieba
from collections import Counter
import re
from pyecharts import options as opts
from pyecharts.charts import WordCloud, Bar, Line, Pie, Radar, Scatter, HeatMap, TreeMap
from streamlit_echarts import st_pyecharts
import numpy as np

# 页面配置
st.set_page_config(page_title="URL文本词频分析系统", layout="wide")

# --------------------------
# 1. 工具函数定义
# --------------------------
def fetch_url_content(url):
    """
    抓取指定URL的文本内容
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # 抛出HTTP请求异常
        response.encoding = response.apparent_encoding  # 自动识别编码
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 移除脚本、样式等无关标签
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # 提取文本内容
        text = soup.get_text()
        # 清理多余空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        st.error(f"URL内容抓取失败：{str(e)}")
        return None

def text_segment_and_word_freq(text, stop_words=None, min_freq=1):
    """
    文本分词与词频统计
    :param text: 原始文本
    :param stop_words: 停用词列表
    :param min_freq: 最低词频过滤
    :return: 词频字典、排序后的词频列表
    """
    if stop_words is None:
        # 基础中文停用词
        stop_words = {"的", "了", "是", "在", "和", "有", "我", "你", "他", "她", "它",
                      "们", "就", "都", "这", "那", "不", "也", "还", "将", "会", "要",
                      "之", "与", "及", "或", "对于", "关于", "通过", "为了", "随着"}
    
    # 分词
    words = jieba.lcut(text)
    # 过滤条件：非空、长度大于1、非停用词、非纯数字
    valid_words = [
        word for word in words
        if len(word) > 1 and word not in stop_words and not word.isdigit()
    ]
    # 统计词频
    word_freq = Counter(valid_words)
    # 过滤低频词
    word_freq_filtered = {word: cnt for word, cnt in word_freq.items() if cnt >= min_freq}
    # 按词频排序
    sorted_word_freq = sorted(word_freq_filtered.items(), key=lambda x: x[1], reverse=True)
    return word_freq_filtered, sorted_word_freq

# --------------------------
# 2. 侧边栏配置（图表筛选+参数设置）
# --------------------------
st.sidebar.title("功能筛选面板")

# 图表类型选择（至少7种，实际提供8种）
chart_types = st.sidebar.multiselect(
    "选择要展示的图表类型",
    options=[
        "词云图", "柱状图（前20词频）", "折线图（前20词频）", 
        "饼图（前10词频）", "雷达图（前8词频）", "散点图（前20词频）",
        "热力图（词频分布）", "矩形树图（前20词频）"
    ],
    default=["词云图", "柱状图（前20词频）"]
)

# 低频词过滤交互设置
min_frequency = st.sidebar.slider(
    "过滤低频词（最低词频）",
    min_value=1,
    max_value=20,
    value=2,
    step=1,
    help="设置最小词频，低于该值的词汇将被过滤"
)

# 额外参数设置
st.sidebar.divider()
st.sidebar.subheader("额外配置")
top_n = st.sidebar.number_input(
    "词频排名展示数量",
    min_value=10,
    max_value=50,
    value=20,
    step=1
)

# --------------------------
# 3. 主页面核心逻辑
# --------------------------
st.title("URL文章文本词频分析与可视化")
st.divider()

# 文本输入框（用户输入文章URL）
url_input = st.text_input(
    "请输入文章URL地址",
    placeholder="例如：https://www.example.com/article.html",
    help="支持大部分静态网页文章抓取"
)

# 提交按钮
if st.button("开始分析", type="primary", disabled=not url_input):
    # 1. 抓取URL内容
    with st.spinner("正在抓取URL内容..."):
        article_text = fetch_url_content(url_input)
        if not article_text:
            st.stop()
    
    # 2. 分词与词频统计
    with st.spinner("正在进行分词与词频统计..."):
        word_freq_dict, sorted_word_freq_list = text_segment_and_word_freq(
            article_text,
            min_freq=min_frequency
        )
        if not sorted_word_freq_list:
            st.warning("未筛选到符合条件的词汇，请降低最低词频重试")
            st.stop()
    
    # 3. 提取前N词汇（用于展示和绘图）
    top_n_words = sorted_word_freq_list[:top_n]
    top_n_words_names = [item[0] for item in top_n_words]
    top_n_words_counts = [item[1] for item in top_n_words]
    
    # 展示词频排名前20（及用户设置的top_n）
    st.subheader(f"词频排名前 {len(top_n_words)} 词汇")
    top_n_df = {
        "词汇": top_n_words_names,
        "词频": top_n_words_counts
    }
    # 改用st.table，避免pyarrow依赖报错，同时无Emoji乱码
    st.table(top_n_df)
    
    st.divider()
    st.subheader("可视化图表展示")
    col1, col2 = st.columns(2)  # 分栏展示，优化布局
    
    # 4. 按选择的图表类型绘制（pyecharts + streamlit_echarts）
    for chart_type in chart_types:
        # 词云图
        if chart_type == "词云图":
            with col1:
                st.caption("词云图（词频越高，字体越大）")
                wordcloud = (
                    WordCloud()
                    .add(
                        series_name="词频分布",
                        data_pair=sorted_word_freq_list[:100],  # 取前100个词汇绘制词云
                        word_size_range=[12, 60],
                        shape="circle"
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="文章核心词汇词云"),
                        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c}")
                    )
                )
                st_pyecharts(wordcloud, height="400px")
        
        # 柱状图（前20词频）
        elif chart_type == "柱状图（前20词频）":
            with col2:
                st.caption(f"柱状图（前 {len(top_n_words)} 词汇词频）")
                bar = (
                    Bar()
                    .add_xaxis(top_n_words_names)
                    .add_yaxis("词频", top_n_words_counts)
                    .reversal_axis()  # 横向柱状图，更易查看词汇
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title=f"前 {len(top_n_words)} 词汇词频柱状图"),
                        xaxis_opts=opts.AxisOpts(name="词频"),
                        yaxis_opts=opts.AxisOpts(name="词汇"),
                        tooltip_opts=opts.TooltipOpts(trigger="item")
                    )
                )
                st_pyecharts(bar, height="400px")
        
        # 折线图（前20词频）
        elif chart_type == "折线图（前20词频）":
            with col1:
                st.caption(f"折线图（前 {len(top_n_words)} 词汇词频趋势）")
                line = (
                    Line()
                    .add_xaxis(top_n_words_names)
                    .add_yaxis("词频", top_n_words_counts, markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max"), opts.MarkPointItem(type_="min")]))
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title=f"前 {len(top_n_words)} 词汇词频折线图"),
                        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
                        yaxis_opts=opts.AxisOpts(name="词频"),
                        tooltip_opts=opts.TooltipOpts(trigger="item")
                    )
                )
                st_pyecharts(line, height="400px")
        
        # 饼图（前10词频）
        elif chart_type == "饼图（前10词频）":
            with col2:
                top_10_words = sorted_word_freq_list[:10]
                top_10_names = [item[0] for item in top_10_words]
                top_10_counts = [item[1] for item in top_10_words]
                st.caption("饼图（前10词汇词频占比）")
                pie = (
                    Pie()
                    .add(
                        "",
                        list(zip(top_10_names, top_10_counts)),
                        radius=["30%", "75%"],
                        rosetype="radius"
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="前10词汇词频饼图"),
                        legend_opts=opts.LegendOpts(orient="vertical", pos_top="10%", pos_left="left")
                    )
                    .set_series_opts(tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c} ({d}%)"))
                )
                st_pyecharts(pie, height="400px")
        
        # 雷达图（前8词频）
        elif chart_type == "雷达图（前8词频）":
            with col1:
                top_8_words = sorted_word_freq_list[:8]
                top_8_names = [item[0] for item in top_8_words]
                top_8_counts = [item[1] for item in top_8_words]
                st.caption("雷达图（前8词汇词频对比）")
                radar = (
                    Radar()
                    .add_schema(
                        schema=[opts.RadarIndicatorItem(name=name, max_=max(top_8_counts)) for name in top_8_names],
                        shape="polygon"
                    )
                    .add("词频", [top_8_counts])
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="前8词汇词频雷达图"),
                        tooltip_opts=opts.TooltipOpts(trigger="item")
                    )
                )
                st_pyecharts(radar, height="400px")
        
        # 散点图（前20词频）
        elif chart_type == "散点图（前20词频）":
            with col2:
                st.caption(f"散点图（前 {len(top_n_words)} 词汇词频）")
                scatter = (
                    Scatter()
                    .add_xaxis(top_n_words_names)
                    .add_yaxis("词频", top_n_words_counts, symbol_size=10)
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title=f"前 {len(top_n_words)} 词汇词频散点图"),
                        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
                        yaxis_opts=opts.AxisOpts(name="词频"),
                        tooltip_opts=opts.TooltipOpts(trigger="item")
                    )
                )
                st_pyecharts(scatter, height="400px")
        
        # 热力图（词频分布）
        elif chart_type == "热力图（词频分布）":
            with col1:
                st.caption("热力图（词汇词频分布）")
                # 构造热力图数据（二维矩阵）
                heatmap_data = []
                row_num = 5
                col_num = len(top_n_words) // row_num + 1
                # 填充数据
                for i in range(row_num):
                    for j in range(col_num):
                        idx = i * col_num + j
                        if idx < len(top_n_words_counts):
                            heatmap_data.append([i, j, top_n_words_counts[idx]])
                        else:
                            heatmap_data.append([i, j, 0])
                # 绘制热力图
                heatmap = (
                    HeatMap()
                    .add_xaxis(list(range(row_num)))
                    .add_yaxis("词频", list(range(col_num)), heatmap_data)
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="词汇词频热力图"),
                        visualmap_opts=opts.VisualMapOpts(min_=min(top_n_words_counts), max_=max(top_n_words_counts)),
                        tooltip_opts=opts.TooltipOpts(trigger="item")
                    )
                )
                st_pyecharts(heatmap, height="400px")
        
        # 矩形树图（前20词频）
        elif chart_type == "矩形树图（前20词频）":
            with col2:
                st.caption(f"矩形树图（前 {len(top_n_words)} 词汇词频）")
                treemap_data = [{"name": item[0], "value": item[1]} for item in top_n_words]
                treemap = (
                    TreeMap()
                    .add("词频", treemap_data, roam=False, label_opts=opts.LabelOpts(show=True, font_size=10))
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title=f"前 {len(top_n_words)} 词汇词频矩形树图"),
                        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c}")
                    )
                )
                st_pyecharts(treemap, height="400px")

# 无URL输入时的提示
if not url_input:
    st.info("请在上方输入框中填写文章URL，然后点击开始分析按钮进行处理")
