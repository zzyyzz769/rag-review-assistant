"""校务通 Streamlit 前端。启动：python -m streamlit run app.py"""

import html

import streamlit as st

from rag_core import GROUPS, ROLE_ACCESS, ROLE_NAMES, CampusRAG


st.set_page_config(page_title="校务通", page_icon="🎓", layout="wide")


@st.cache_resource
def get_rag(llm_model: str, embedding_model: str, base_url: str) -> CampusRAG:
    return CampusRAG(
        llm_model=llm_model,
        embedding_model=embedding_model,
        ollama_base_url=base_url,
    )


def role_text(role: str) -> str:
    return ROLE_NAMES.get(role, role)


def highlight_document(text: str, snippet: str) -> str:
    """转义全部文档内容，仅把命中的原文片段包装为黄色 mark。"""
    safe_text = html.escape(text)
    safe_snippet = html.escape(snippet.strip())
    if safe_snippet and safe_snippet in safe_text:
        safe_text = safe_text.replace(
            safe_snippet, f"<mark>{safe_snippet}</mark>", 1
        )
    else:
        safe_text = (
            "<p><mark>未能在当前页面自动定位完整切片，以下展示对应原文。</mark></p>"
            + safe_text
        )
    return (
        "<div style='white-space:pre-wrap;line-height:1.8;max-height:560px;"
        "overflow:auto;padding:16px;border:1px solid #ddd;border-radius:8px;"
        "background:#fafafa;color:#222'>"
        + safe_text
        + "</div>"
    )


# 登录只使用默认模型配置；登录后可在侧边栏调整。
rag = get_rag("qwen2.5:3b", "bge-m3", "http://localhost:11434")
if "user" not in st.session_state:
    st.session_state.user = None

st.title("🎓 校务通")
st.caption("权限分组知识库 · 本地 RAG · 原文高亮 · 全程审计")

if not st.session_state.user:
    st.subheader("登录校务通")
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", type="primary")
    if submitted:
        user = rag.authenticate(username, password)
        if user:
            st.session_state.user = user
            st.session_state.pop("qa_csv", None)
            st.rerun()
        else:
            st.error("用户名或密码错误。")
    with st.expander("首次运行的本地演示账户"):
        st.code(
            "学生：student / Student@123\n"
            "辅导员：counselor / Counselor@123\n"
            "管理员：admin / Admin@123"
        )
        st.warning("演示密码仅用于初次登录，请登录后立即修改。")
    st.stop()

user = st.session_state.user

with st.sidebar:
    st.success(f"{user['username']} · {role_text(user['role'])}")
    allowed_names = "、".join(GROUPS[g] for g in ROLE_ACCESS[user["role"]])
    st.caption(f"当前可访问：{allowed_names}")

    with st.expander("本地模型设置"):
        llm_model = st.text_input("Qwen 生成模型", value="qwen2.5:3b")
        embedding_model = st.text_input("嵌入模型", value="bge-m3")
        base_url = st.text_input("Ollama 地址", value="http://localhost:11434")
        st.caption("更换嵌入模型后，需要重新向量化全部现有文档。")
        rag = get_rag(llm_model, embedding_model, base_url)

    top_k = st.slider("每次检索片段数", 2, 8, 4)
    try:
        st.metric("权限范围内向量切片", rag.accessible_chunk_count(user))
    except Exception:
        st.caption("向量库尚未初始化")

    st.divider()
    if st.button("生成我的问答 CSV", use_container_width=True):
        try:
            st.session_state.qa_csv = rag.export_qa_csv(user)
            st.success("CSV 已生成，仅包含当前账户的问答。")
        except Exception as exc:
            st.error(f"导出失败：{exc}")
    if st.session_state.get("qa_csv"):
        st.download_button(
            "下载问答记录 CSV",
            data=st.session_state.qa_csv,
            file_name="校务通_问答记录.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("修改密码"):
        with st.form("password_form"):
            old_password = st.text_input("原密码", type="password")
            new_password = st.text_input("新密码（至少 8 位）", type="password")
            change = st.form_submit_button("确认修改")
        if change:
            try:
                rag.change_password(user, old_password, new_password)
                st.success("密码已修改。")
            except Exception as exc:
                st.error(str(exc))

    if st.button("退出登录", use_container_width=True):
        st.session_state.clear()
        st.rerun()


def show_sources(sources: list[dict], record_id: int) -> None:
    if not sources:
        return
    with st.expander("查看来源 / 打开原文高亮"):
        for source in sources:
            page = f" · 第 {source['page']} 页" if source.get("page") else ""
            st.markdown(
                f"**[资料{source['id']}] {source['group_name']}/"
                f"{source['source']}{page} · 切片 {source.get('chunk', '')}**"
            )
            st.caption(source["content"])
            if st.button(
                "打开来源文档并标黄",
                key=f"source_{record_id}_{source['id']}",
            ):
                try:
                    opened = rag.open_source(
                        user,
                        source["group"],
                        source["source"],
                        source.get("page"),
                    )
                    st.session_state.preview = {
                        "title": f"{source['group_name']}/{source['source']}",
                        "location": opened["location"],
                        "html": highlight_document(opened["text"], source["content"]),
                    }
                except Exception as exc:
                    st.error(f"原文打开失败：{exc}")


def render_documents() -> None:
    docs = rag.list_documents(user)
    if not docs:
        st.info("当前权限范围内还没有文档。")
        return
    rows = [
        {
            "分组": d["group_name"],
            "文件名": d["filename"],
            "大小（KB）": d["size_kb"],
            "修改时间": d["modified"],
        }
        for d in docs
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


tab_names = ["教务问答", "我的知识库"]
if user["role"] == "admin":
    tab_names += ["文档管理", "审计日志"]
tabs = st.tabs(tab_names)

with tabs[0]:
    records = list(reversed(rag.get_qa_records(user, limit=30)))
    if not records:
        st.info("请输入问题开始咨询。系统只会检索当前角色有权访问的知识库。")
    for record in records:
        with st.chat_message("user"):
            st.markdown(record["question"])
        with st.chat_message("assistant"):
            st.caption(record["created_at"])
            st.markdown(record["answer"])
            show_sources(record["sources"], int(record["id"]))

    if st.session_state.get("preview"):
        preview = st.session_state.preview
        st.subheader(f"原文高亮：{preview['title']}")
        st.caption(preview["location"])
        st.markdown(preview["html"], unsafe_allow_html=True)
        if st.button("关闭原文预览"):
            st.session_state.pop("preview", None)
            st.rerun()

    question = st.chat_input("例如：本科生申请缓考需要哪些材料和步骤？")
    if question:
        try:
            with st.spinner("正在权限范围内检索并生成回答……"):
                rag.ask(user, question, top_k=top_k)
            st.session_state.pop("qa_csv", None)
            st.rerun()
        except Exception as exc:
            st.error(f"处理失败：{exc}")

with tabs[1]:
    st.subheader("当前角色可访问的文档")
    render_documents()

if user["role"] == "admin":
    with tabs[2]:
        st.subheader("批量导入")
        group_labels = {name: key for key, name in GROUPS.items()}
        target_group = group_labels[
            st.selectbox("上传到权限分组", list(group_labels.keys()))
        ]
        uploads = st.file_uploader(
            "可一次选择多个 PDF / TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if st.button("保存并逐篇向量化", type="primary"):
            if not uploads:
                st.warning("请先选择文件。")
            else:
                success = 0
                for upload in uploads:
                    try:
                        filename = rag.save_document(
                            user, target_group, upload.name, upload.getvalue()
                        )
                        count = rag.index_document(user, target_group, filename)
                        st.success(f"{filename}：已写入 {count} 个切片")
                        success += 1
                    except Exception as exc:
                        st.error(f"{upload.name} 导入失败：{exc}")
                st.info(f"本次成功导入 {success}/{len(uploads)} 个文件。")

        st.divider()
        st.subheader("单文档管理")
        docs = rag.list_documents(user)
        if not docs:
            st.info("暂无文档。")
        else:
            doc_map = {
                d["doc_id"]: d for d in docs
            }
            selected_id = st.selectbox(
                "选择要单独操作的文档",
                list(doc_map.keys()),
                format_func=lambda key: (
                    f"{doc_map[key]['group_name']}/{doc_map[key]['filename']}"
                ),
            )
            selected = doc_map[selected_id]
            col1, col2 = st.columns(2)
            with col1:
                if st.button("仅重新向量化此文档", use_container_width=True):
                    try:
                        with st.spinner("正在更新当前文档的向量……"):
                            count = rag.index_document(
                                user, selected["group"], selected["filename"]
                            )
                        st.success(f"已完成，当前文档共 {count} 个切片。")
                    except Exception as exc:
                        st.error(f"重建失败：{exc}")
            with col2:
                confirm = st.checkbox("确认同时删除原文件和对应向量")
                if st.button("删除此文档", use_container_width=True, disabled=not confirm):
                    try:
                        rag.delete_document(user, selected["group"], selected["filename"])
                        st.success("文档及其向量已删除。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")

    with tabs[3]:
        st.subheader("角色访问审计日志")
        role_filter_label = st.selectbox(
            "按角色筛选", ["全部角色", "学生", "辅导员", "管理员"]
        )
        label_to_role = {None: None, "全部角色": None, **{v: k for k, v in ROLE_NAMES.items()}}
        logs = rag.get_audit_logs(user, role=label_to_role[role_filter_label], limit=300)
        if logs:
            rows = [
                {
                    "时间": row["created_at"],
                    "用户名": row["username"],
                    "角色": role_text(row["role"]),
                    "操作": row["action"],
                    "资源": row["resource"],
                    "详情": row["detail"],
                }
                for row in logs
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("暂无审计记录。")