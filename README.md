# 中文网页文本词频分析与可视化系统

基于 Streamlit 的中文词频分析工具，支持 URL 抓取、jieba 分词、8 种 pyecharts 图表、AI 大模型分析。

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 http://localhost:8501

## AI 分析配置

系统支持豆包（Doubao）大模型进行词频智能分析。使用前需配置 API Key：

```bash
# Linux / Mac
export DOUBAO_API_KEY="your-key"
export DOUBAO_TURBO_MODEL="ep-xxxxxxxxxxxx-xxxxx"
export DOUBAO_MINI_MODEL="ep-xxxxxxxxxxxx-xxxxx"

# Windows
set DOUBAO_API_KEY=your-key
set DOUBAO_TURBO_MODEL=ep-xxxxxxxxxxxx-xxxxx
set DOUBAO_MINI_MODEL=ep-xxxxxxxxxxxx-xxxxx
```

未配置时 AI 功能仍可见但调用会报错。

## 项目结构

```
word-freq-app/
├── app.py                  # Streamlit 主入口
├── fetch.py                # 网页抓取模块
├── tokenize_words.py       # 分词与词频统计
├── ai_analysis.py          # AI 分析模块
├── stopwords_cn.txt        # 中文停用词表
├── requirements.txt        # 依赖清单
├── .env.example            # 环境变量示例
└── images/                 # 演示截图
```

## 技术栈

| 层次 | 技术 |
|------|------|
| Web | Streamlit |
| 抓取 | requests + BeautifulSoup4 |
| 分词 | jieba |
| 图表 | pyecharts |
| 数据 | pandas |
| AI | 豆包 Ark API |
