\# 校务通



轻量 RAG 校园教务知识库，支持 PDF、TXT 文档上传、文档切片、

Chroma 本地向量检索、Ollama 本地 Qwen 问答及原文来源展示。



\## 技术栈



\- Python

\- LangChain

\- Chroma

\- Ollama

\- Qwen

\- Streamlit



\## 本地运行



\### 安装依赖



pip install -r requirements.txt



\### 下载本地模型



ollama pull qwen2.5:3b

ollama pull bge-m3



\### 启动项目



python -m streamlit run app.py

