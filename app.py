"""校务通 Streamlit 前端。启动：python -m streamlit run app.py"""

import hashlib
import html

import streamlit as st
import streamlit.components.v1 as components

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
            safe_snippet, f"<mark id='active-highlight'>{safe_snippet}</mark>", 1
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
        llm_model = st.selectbox(
            "Qwen 生成模型",
            ["qwen2.5:3b", "qwen:1.8b"],
            index=0,
        )
        if llm_model == "qwen:1.8b":
            st.caption("qwen:1.8b 更容易重复或漏答，建议使用 qwen2.5:3b。")
        embedding_model = st.text_input("嵌入模型", value="bge-m3")
        base_url = st.text_input("Ollama 地址", value="http://localhost:11434")
        st.caption("更换嵌入模型后，需要重新向量化全部现有文档。")
        rag = get_rag(llm_model, embedding_model, base_url)

    top_k = st.slider("最多显示来源数", 1, 5, 3)
    retrieval_mode = st.radio(
        "检索模式",
        ["hybrid", "vector", "bm25"],
        format_func=lambda value: {
            "hybrid": "混合检索", "vector": "纯向量检索", "bm25": "纯 BM25"
        }[value],
        horizontal=True,
    )
    use_rerank = st.checkbox("启用 BGE Rerank 重排", value=False)
    if use_rerank:
        st.caption("首次启用需下载重排模型，之后本地运行。")

    try:
        stats = rag.get_stats(user)
        row1 = st.columns(2)
        row1[0].metric("文档", stats["documents"])
        row1[1].metric("切片", stats["chunks"])
        row2 = st.columns(2)
        row2[0].metric("敏感命中", stats["sensitive_hits"])
        row2[1].metric("今日问答", stats["today_qa"])
    except Exception:
        st.caption("统计面板尚未初始化")

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
        st.caption("已嵌入当前账户的零宽溯源水印")

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
    with st.expander(f"答案依据（{len(sources)} 条，已提取关键句）"):
        for source in sources:
            page = f"第 {source['page']} 页" if source.get("page") else "全文"
            st.markdown(
                f"**[资料{source['id']}] {source['source']} · {page}**"
            )
            matched = source.get("matched") or []
            source_note = []
            if matched:
                source_note.append("命中词：" + "、".join(matched))
            if source.get("confidence") is not None:
                source_note.append(f"匹配分 {source['confidence']}/100")
            if source_note:
                st.caption(" · ".join(source_note))
            evidence = source.get("evidence") or source["content"][:240] + "……"
            st.markdown(f"> {evidence}")

            preview_key = f"{record_id}_{source['id']}"
            if st.button(
                "定位到原文",
                key=f"source_{record_id}_{source['id']}",
            ):
                try:
                    opened = rag.open_source(
                        user,
                        source["group"],
                        source["source"],
                        source.get("page"),
                    )
                    snippet = source.get("highlight") or evidence
                    st.session_state.preview = {
                        "key": preview_key,
                        "title": f"{source['group_name']}/{source['source']}",
                        "location": opened["location"],
                        "html": highlight_document(opened["text"], snippet),
                    }
                except Exception as exc:
                    st.error(f"原文打开失败：{exc}")

            current = st.session_state.get("preview")
            if current and current.get("key") == preview_key:
                st.caption(f"原文位置：{current['location']}，黄色部分为关键句")
                components.html(
                    current["html"]
                    + "<script>const el=document.getElementById('active-highlight');"
                      "if(el){el.scrollIntoView({block:'center'});}</script>",
                    height=560,
                    scrolling=True,
                )
                if st.button("关闭原文", key=f"close_{record_id}_{source['id']}"):
                    st.session_state.pop("preview", None)
                    st.rerun()
            st.divider()


def show_answer_meta(record: dict) -> None:
    confidence = int(record.get("confidence", 0))
    mode_names = {"hybrid": "混合", "vector": "纯向量", "bm25": "纯 BM25"}
    mode = mode_names.get(record.get("retrieval_mode"), record.get("retrieval_mode") or "旧版")
    rerank = "＋Rerank" if record.get("rerank_used") else ""
    st.caption(
        f"{record['created_at']} · {record.get('model') or '旧版模型'} · {mode}{rerank}"
    )
    st.progress(confidence / 100, text=f"检索匹配置信度：{confidence}/100")
    if confidence < 60:
        st.warning("知识库匹配度较低，回答仅供参考，请仔细核对原文。")

    feedback = record.get("feedback", "")
    if feedback:
        st.caption("已反馈：" + ("👍 有用" if feedback == "useful" else "👎 无用"))
    col1, col2, _ = st.columns([1, 1, 5])
    with col1:
        if st.button("👍 有用", key=f"useful_{record['id']}"):
            rag.set_feedback(user, int(record["id"]), "useful")
            st.rerun()
    with col2:
        if st.button("👎 无用", key=f"useless_{record['id']}"):
            rag.set_feedback(user, int(record["id"]), "useless")
            st.rerun()


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


tab_names = ["教务问答", "我的知识库", "版本对比"]
if user["role"] == "admin":
    tab_names += ["文档管理", "审计日志"]
tabs = st.tabs(tab_names)

with tabs[0]:
    if st.session_state.get("rerank_notice"):
        st.warning(st.session_state.pop("rerank_notice"))
    records = list(reversed(rag.get_qa_records(user, limit=30)))
    if not records:
        st.info("请输入问题开始咨询。系统只会检索当前角色有权访问的知识库。")
    for record in records:
        with st.chat_message("user"):
            st.markdown(record["question"])
        with st.chat_message("assistant"):
            show_answer_meta(record)
            st.markdown(rag.clean_answer(record["answer"]))
            show_sources(record["sources"], int(record["id"]))

    question = st.chat_input("请输入要查询的教务制度、办理条件或流程")
    if question:
        try:
            with st.spinner("正在权限范围内检索并生成回答……"):
                result = rag.ask(
                    user,
                    question,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                    use_rerank=use_rerank,
                )
            if result.get("rerank_notice"):
                st.session_state.rerank_notice = result["rerank_notice"]
            st.session_state.pop("qa_csv", None)
            st.rerun()
        except Exception as exc:
            st.error(f"处理失败：{exc}")

with tabs[1]:
    st.subheader("当前角色可访问的文档")
    render_documents()

with tabs[2]:
    st.subheader("同一问题的回答版本对比")
    all_records = rag.get_qa_records(user, limit=200)
    questions = sorted({record["question"] for record in all_records})
    if not questions:
        st.info("暂无可对比的问答版本。")
    else:
        selected_question = st.selectbox("选择历史问题", questions)
        versions = list(reversed([
            record for record in all_records if record["question"] == selected_question
        ]))
        for version in versions:
            title = (
                f"#{version['id']} · {version['created_at']} · "
                f"{version.get('model') or '旧版'} · "
                f"{version.get('retrieval_mode', 'hybrid')} · "
                f"置信度 {version.get('confidence', 0)}"
            )
            with st.expander(title, expanded=len(versions) == 1):
                st.markdown(rag.clean_answer(version["answer"]))
                st.caption(
                    "Rerank：" + ("开启" if version.get("rerank_used") else "关闭")
                    + "；反馈：" + (version.get("feedback") or "未反馈")
                )

if user["role"] == "admin":
    with tabs[3]:
        st.subheader("批量导入")
        group_labels = {name: key for key, name in GROUPS.items()}
        target_group = group_labels[
            st.selectbox("上传到权限分组", list(group_labels.keys()))
        ]
        uploads = st.file_uploader(
            "可一次选择多个 PDF / TXT / DOCX",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )
        st.caption("支持 Word DOCX 的正文和表格；旧版 DOC 请先用 Word 另存为 DOCX。")
        preflight_fatal = False
        if uploads:
            st.markdown("**上传预检**")
            cache = st.session_state.setdefault("preflight_cache", {})
            for upload in uploads:
                content = upload.getvalue()
                digest = hashlib.sha256(
                    target_group.encode("utf-8") + upload.name.encode("utf-8") + content
                ).hexdigest()
                if digest not in cache:
                    cache[digest] = rag.preflight_document(
                        user, target_group, upload.name, content
                    )
                check = cache[digest]
                if not check["ok"]:
                    preflight_fatal = True
                    st.error(f"{upload.name}：{check['warning']}")
                elif check["warning"]:
                    st.warning(f"{upload.name}：{check['warning']}（仍可继续导入）")
                else:
                    st.success(f"{upload.name}：预检通过，提取到 {check['char_count']} 个字符")

        if st.button("保存并逐篇向量化", type="primary", disabled=preflight_fatal):
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

    with tabs[4]:
        st.subheader("角色访问审计日志")
        role_filter_label = st.selectbox(
            "按角色筛选", ["全部角色", "学生", "辅导员", "管理员"]
        )
        label_to_role = {None: None, "全部角色": None, **{v: k for k, v in ROLE_NAMES.items()}}
        if st.button("生成脱敏审计日志 CSV"):
            try:
                st.session_state.audit_csv = rag.export_audit_csv(user)
                st.success("已生成：用户与敏感信息脱敏，并嵌入当前管理员水印。")
            except Exception as exc:
                st.error(f"审计日志导出失败：{exc}")
        if st.session_state.get("audit_csv"):
            st.download_button(
                "下载审计日志 CSV",
                data=st.session_state.audit_csv,
                file_name="校务通_脱敏审计日志.csv",
                mime="text/csv",
            )

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