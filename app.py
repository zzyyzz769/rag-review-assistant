"""校务通 Streamlit 前端。

启动命令：streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from rag_core import CampusRAG


st.set_page_config(
    page_title="校务通",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def get_rag(
    llm_model: str,
    embedding_model: str,
    base_url: str,
) -> CampusRAG:
    """缓存 RAG 实例，避免 Streamlit 每次交互都重新初始化。"""
    return CampusRAG(
        llm_model=llm_model,
        embedding_model=embedding_model,
        ollama_base_url=base_url,
    )


def show_sources(sources: list[dict]) -> None:
    """在回答下方展示检索到的原文片段。"""
    if not sources:
        return

    with st.expander(
        "查看原文来源片段",
        expanded=False,
    ):
        for item in sources:
            page = (
                f" · 第 {item['page']} 页"
                if item.get("page")
                else ""
            )
            chunk = (
                f" · 切片 {item['chunk']}"
                if item.get("chunk")
                else ""
            )

            st.markdown(
                f"**[资料{item['id']}] "
                f"{item['source']}{page}{chunk}**"
            )
            st.text(item["content"])


st.title("🎓 校务通")
st.caption(
    "轻量 RAG 校园教务知识库 · "
    "文档、向量、检索和回答均在本机处理"
)

with st.sidebar:
    st.header("本地模型")

    llm_model = st.text_input(
        "Qwen 生成模型",
        value="qwen2.5:3b",
    )
    embedding_model = st.text_input(
        "嵌入模型",
        value="bge-m3",
    )
    base_url = st.text_input(
        "Ollama 地址",
        value="http://localhost:11434",
    )
    top_k = st.slider(
        "每次检索片段数",
        min_value=2,
        max_value=8,
        value=4,
    )

    try:
        rag = get_rag(
            llm_model,
            embedding_model,
            base_url,
        )
    except Exception as exc:
        st.error(f"初始化失败：{exc}")
        st.stop()

    st.divider()
    st.header("知识库管理")

    uploaded_files = st.file_uploader(
        "上传教务 PDF / TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button(
        "保存上传文件并重建索引",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning(
                "请先选择至少一个 PDF 或 TXT 文件。"
            )
        else:
            try:
                for uploaded in uploaded_files:
                    # 只保留文件名，避免路径穿越。
                    # 同名文件直接覆盖。
                    filename = Path(uploaded.name).name
                    target_path = rag.data_dir / filename
                    target_path.write_bytes(
                        uploaded.getvalue()
                    )

                with st.spinner(
                    "正在读取、切片并向量化，请稍候……"
                ):
                    stats = rag.build_index()

                st.success(
                    f"入库完成：{stats['files']} 个文件，"
                    f"{stats['chunks']} 个文本切片。"
                )

            except Exception as exc:
                st.error(f"索引构建失败：{exc}")

    if st.button(
        "按现有文档重新构建",
        use_container_width=True,
    ):
        try:
            with st.spinner("正在重建索引……"):
                stats = rag.build_index()

            st.success(
                f"重建完成，共 {stats['chunks']} 个文本切片。"
            )

        except Exception as exc:
            st.error(f"重建失败：{exc}")

    files = rag.list_files()

    if files:
        st.caption("当前原始文档")

        for file in files:
            st.write(f"📄 {file.name}")
    else:
        st.info("尚未上传文档。")

    try:
        st.metric(
            "向量库切片数",
            rag.chunk_count(),
        )
    except Exception:
        st.caption("向量库尚未初始化")


if "messages" not in st.session_state:
    st.session_state.messages = []


# 显示历史对话及其来源。
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_sources(
                message.get("sources", [])
            )


question = st.chat_input(
    "例如：本科生申请缓考需要哪些材料和步骤？"
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索并生成回答……"):
                result = rag.ask(
                    question,
                    top_k=top_k,
                )

            st.markdown(result["answer"])
            show_sources(result["sources"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )

        except Exception as exc:
            error_message = (
                f"处理失败：{exc}\n\n"
                "请检查 Ollama 是否已启动、模型是否已下载，"
                "以及知识库是否已完成索引。"
            )

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )