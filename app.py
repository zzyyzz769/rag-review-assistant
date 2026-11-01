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


def on_model_change() -> None:
    """切换生成模型时移除旧组件状态，避免历史消息与原文预览交叉更新。"""
    st.session_state.pop("preview", None)
    st.session_state.pop("qa_csv", None)


def plain_answer(text: str) -> str:
    """历史消息使用稳定纯文本渲染，保留结构但移除 Markdown 标记。"""
    return rag.clean_answer(text).replace("**", "").strip()


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
            on_change=on_model_change,
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
            "hybrid": "两种方式一起找", "vector": "按意思找（向量）", "bm25": "按关键词找（BM25）"
        }[value],
        horizontal=True,
    )
    use_rerank = st.checkbox("找到资料后再精细排序（BGE Rerank）", value=False)
    if use_rerank:
        st.caption("第一次使用需要下载排序模型，之后可在本地运行；不开启也能正常问答。")

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
            if source.get("academic_year"):
                source_note.append(source["academic_year"])
            source_note.append(source.get("status") or "现行有效")
            if source.get("confidence") is not None:
                source_note.append(f"匹配分 {source['confidence']}/100")
            if source_note:
                st.caption(" · ".join(source_note))
            if source.get("is_current") is False:
                st.warning("该来源不是现行有效版本，请优先核对最新制度。")
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
            "学年度": d.get("academic_year") or "未设置",
            "版本状态": d.get("status") or "现行有效",
            "生效日期": d.get("effective_date") or "未设置",
            "失效日期": d.get("expiry_date") or "未设置",
            "大小（KB）": d["size_kb"],
            "修改时间": d["modified"],
        }
        for d in docs
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


tab_names = ["教务问答", "我的知识库", "评测中心", "版本对比"]
if user["role"] == "admin":
    tab_names += ["文档管理", "自动诊断", "审计日志"]
tabs = st.tabs(tab_names)
tab_map = dict(zip(tab_names, tabs))

with tab_map["教务问答"]:
    st.caption(
        "使用说明：直接输入你想问的事情，再查看结论、要点和原文依据。接着问“那需要什么材料？”也可以，"
        "系统会结合前面的对话理解你的意思。如果问题太简单、看不出要问什么，系统会先请你补充。"
    )
    if st.session_state.get("rerank_notice"):
        st.warning(st.session_state.pop("rerank_notice"))
    if st.session_state.get("generation_notice"):
        st.warning(st.session_state.pop("generation_notice"))
    records = list(reversed(rag.get_qa_records(user, limit=30)))
    if not records:
        st.info("请输入问题开始咨询。系统只会检索当前角色有权访问的知识库。")
    for record in records:
        with st.chat_message("user"):
            st.text(record["question"])
            resolved = record.get("resolved_question") or record["question"]
            if resolved != record["question"]:
                st.caption(f"追问已理解为：{resolved}")
        with st.chat_message("assistant"): 
            show_answer_meta(record)
            st.text(plain_answer(record["answer"]))
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
                    history=[{"question": item["question"]} for item in records[-2:]],
                )
            if result.get("rerank_notice"):
                st.session_state.rerank_notice = result["rerank_notice"]
            if result.get("generation_notice"):
                st.session_state.generation_notice = result["generation_notice"]
            st.session_state.pop("qa_csv", None)
            st.rerun()
        except Exception as exc:
            st.error(f"处理失败：{exc}")

with tab_map["我的知识库"]:
    st.subheader("当前角色可访问的文档")
    render_documents()

with tab_map["评测中心"]:
    st.subheader("检索问答评测中心")
    st.caption(
        "这是给校务通自己做的小测验：你先告诉它正确答案在哪份文件、哪一页，"
        "再让它自己找一遍，看看找得准不准。这里只检查找资料的能力，不评价回答写得好不好。"
    )
    with st.expander("功能说明与填写方法"):
        st.markdown(
            "1. 填一个问题，再填正确答案所在的完整文件名和页码；不知道页码可以留空。\n"
            "2. 正常问题保持“应当找到资料”勾选；如果想测试没有资料时会不会乱找，就取消勾选。\n"
            "3. 点击“开始评测”。换一种找资料的方式再测一次，就能比较哪种方式更适合你的文档。\n\n"
            "**结果怎么看：**“第 1 条找对率”越高，说明正确答案越常排在第一位；"
            "“前 3 条找到率”越高，说明正确答案越常出现在前三条；“正确来源排名得分”越接近 1，"
            "说明正确答案整体排得越靠前；“没有资料时不乱答率”越高越好。"
        )
    default_cases = st.session_state.setdefault("eval_cases", [
        {"question": "", "expected_source": "", "expected_page": None, "should_answer": True},
        {"question": "", "expected_source": "", "expected_page": None, "should_answer": True},
        {"question": "", "expected_source": "", "expected_page": None, "should_answer": False},
    ])
    edited_cases = st.data_editor(
        default_cases,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "question": st.column_config.TextColumn("问题", required=True),
            "expected_source": st.column_config.TextColumn("期望文件名（可空）"),
            "expected_page": st.column_config.NumberColumn("期望页码（可空）", min_value=1),
            "should_answer": st.column_config.CheckboxColumn("应当找到资料", default=True),
        },
        key="evaluation_editor",
    )
    cases = (
        edited_cases.to_dict("records") if hasattr(edited_cases, "to_dict")
        else list(edited_cases)
    )
    current_mode_name = {
        "hybrid": "两种方式一起找", "vector": "按意思找", "bm25": "按关键词找"
    }[retrieval_mode]
    st.info(
        f"本次会使用左侧选择的“{current_mode_name}”，"
        f"精细排序{'已开启' if use_rerank else '未开启'}。"
        "如果这道题在知识库中没有答案，请取消“应当找到资料”。"
    )
    if st.button("开始评测", type="primary"):
        try:
            with st.spinner("正在逐条执行真实检索……"):
                st.session_state.last_evaluation = rag.evaluate_retrieval(
                    user, cases, retrieval_mode=retrieval_mode, use_rerank=use_rerank
                )
        except Exception as exc:
            st.error(f"评测失败：{exc}")

    evaluation = st.session_state.get("last_evaluation")
    if evaluation:
        st.markdown(f"**本次评测 #{evaluation['run_id']}**")
        cols = st.columns(5)
        cols[0].metric("第 1 条找对率", f"{evaluation['recall_1'] * 100:.1f}%")
        cols[1].metric("前 3 条找到率", f"{evaluation['recall_3'] * 100:.1f}%")
        cols[2].metric("正确来源排名得分", f"{evaluation['mrr']:.3f}")
        refusal_text = (
            f"{evaluation['refusal_accuracy'] * 100:.1f}%"
            if evaluation["refusal_case_count"] else "—"
        )
        cols[3].metric("没有资料时不乱答率", refusal_text)
        cols[4].metric("平均检索耗时", f"{evaluation['avg_latency_ms']:.0f} ms")
        detail_rows = [{
            "问题": item["question"],
            "期望来源": item["expected_source"] or "无资料问题",
            "命中排名": item["rank"] or "未命中",
            "返回来源": "；".join(
                f"{s['source']} 第{s['page']}页" if s.get("page") else s["source"]
                for s in item["sources"]
            ) or "无",
            "耗时（ms）": item["latency_ms"],
            "异常": item["error"],
        } for item in evaluation["details"]]
        st.dataframe(detail_rows, use_container_width=True, hide_index=True)

    previous_runs = rag.get_evaluation_runs(user, limit=20)
    if previous_runs:
        with st.expander("查看历史评测"):
            selected_run = st.selectbox(
                "选择评测批次", previous_runs,
                format_func=lambda run: (
                    f"#{run['id']} · {run['created_at']} · "
                    f"前 3 条找到率 {run['recall_3'] * 100:.1f}%"
                ),
            )
            st.write({
                "问题数": selected_run["case_count"],
                "第 1 条找对率": f"{selected_run['recall_1'] * 100:.1f}%",
                "前 3 条找到率": f"{selected_run['recall_3'] * 100:.1f}%",
                "正确来源排名得分": f"{selected_run['mrr']:.3f}",
                "平均耗时（ms）": round(selected_run["avg_latency_ms"], 1),
            })

with tab_map["版本对比"]:
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
                st.text(plain_answer(version["answer"]))
                st.caption(
                    "Rerank：" + ("开启" if version.get("rerank_used") else "关闭")
                    + "；反馈：" + (version.get("feedback") or "未反馈")
                )

if user["role"] == "admin":
    with tab_map["文档管理"]:
        st.subheader("批量导入")
        group_labels = {name: key for key, name in GROUPS.items()}
        target_group = group_labels[
            st.selectbox("上传到权限分组", list(group_labels.keys()))
        ]
        enable_ocr = st.checkbox(
            "识别扫描版 PDF 中的文字（OCR）",
            value=False,
            help="仅识别文字过少的页面；需安装 pymupdf、rapidocr_onnxruntime、numpy 和 pillow。",
        )
        st.caption(
            "使用说明：能正常选中文字的 PDF 不用开启。如果文件像照片一样、无法选中文字，"
            "勾选后再导入，系统会自动读出页面内容并保留页码。导入后请抽查数字、日期和表格是否识别正确。"
        )
        uploads = st.file_uploader(
            "可一次选择多个 PDF / TXT / DOCX",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )
        st.caption("支持 Word DOCX 的正文和表格；旧版 DOC 请先另存为 DOCX。单份文件最大 50 MB。")
        with st.expander("设置本批文档的版本信息"):
            st.caption(
                "用于区分不同学年的制度。填写学年度和日期，现行文件保持勾选；"
                "上传新版后，请将旧版设为历史版本。日期格式为 YYYY-MM-DD，也可留空。"
            )
            academic_year = st.text_input("学年度", placeholder="例如：2025—2026 学年")
            date_cols = st.columns(2)
            effective_date = date_cols[0].text_input("生效日期", placeholder="YYYY-MM-DD")
            expiry_date = date_cols[1].text_input("失效日期", placeholder="YYYY-MM-DD")
            is_current = st.checkbox("作为现行有效版本", value=True)
            replaces = st.text_input("替代的旧文档", placeholder="可填写旧文件名，也可留空")
        blocked_names: set[str] = set()
        if uploads:
            existing_names = {
                doc["filename"] for doc in rag.list_documents(user)
                if doc["group"] == target_group
            }
            duplicate_names = [upload.name for upload in uploads if upload.name in existing_names]
            if duplicate_names:
                st.warning(
                    "以下同名文件会被覆盖并重新整理：" + "、".join(duplicate_names)
                    + "。如需保留旧版，请先修改文件名。"
                )
            st.markdown("**上传预检**")
            cache = st.session_state.setdefault("preflight_cache", {})
            for upload in uploads:
                content = upload.getvalue()
                digest = hashlib.sha256(
                    target_group.encode("utf-8") + upload.name.encode("utf-8") + content
                    + str(enable_ocr).encode("ascii")
                ).hexdigest()
                if digest not in cache:
                    cache[digest] = rag.preflight_document(
                        user, target_group, upload.name, content, allow_ocr=enable_ocr
                    )
                check = cache[digest]
                if not check["ok"]:
                    blocked_names.add(upload.name)
                    st.error(f"{upload.name}：{check['warning']}（只跳过此文件）")
                elif check["warning"]:
                    st.warning(f"{upload.name}：{check['warning']}（仍可继续导入）")
                else:
                    st.success(f"{upload.name}：预检通过，提取到 {check['char_count']} 个字符")

        def run_import_tasks(items: list[dict]) -> None:
            task_rows = []
            failed_items = []
            for item in items:
                name = item["name"]
                try:
                    with st.status(f"{name}：正在入库", expanded=True) as task_status:
                        st.write("1. 保存原文件")
                        filename = rag.save_document(
                            user, target_group, name, item["content"]
                        )
                        st.write("2. 保存版本与有效期")
                        rag.set_document_metadata(
                            user, target_group, filename, academic_year,
                            effective_date, expiry_date, is_current, replaces,
                        )
                        st.write("3. 读取内容并整理成可搜索的资料")
                        count = rag.index_document(
                            user, target_group, filename, use_ocr=enable_ocr
                        )
                        task_status.update(
                            label=f"{filename}：完成，写入 {count} 个切片",
                            state="complete",
                        )
                    task_rows.append({"文件": filename, "状态": "完成", "详情": f"{count} 个切片"})
                except Exception as exc:
                    task_rows.append({"文件": name, "状态": "失败", "详情": str(exc)})
                    failed_items.append(item)
                    st.error(f"{name} 导入失败：{exc}")
            st.session_state.import_tasks = task_rows
            st.session_state.failed_imports = failed_items

        st.caption(
            "导入说明：点击后可以看到每份文件的处理进度，包括保存文件、记录版本、读取内容和整理成可搜索的资料。"
            "某一份失败不会影响其他文件；处理好失败原因后，点击“仅重试失败文件”即可。"
            "失败文件只在这次打开页面期间保留，重启程序后需要重新选择。"
        )
        if st.button("保存并逐篇向量化", type="primary"):
            ready_uploads = [
                upload for upload in (uploads or []) if upload.name not in blocked_names
            ]
            if not uploads:
                st.warning("请先选择文件。")
            elif not ready_uploads:
                st.warning("当前文件都未通过预检，请先处理文件问题。")
            else:
                run_import_tasks([
                    {"name": upload.name, "content": upload.getvalue()}
                    for upload in ready_uploads
                ])

        if st.session_state.get("import_tasks"):
            st.markdown("**最近一次入库任务**")
            st.dataframe(
                st.session_state.import_tasks, use_container_width=True, hide_index=True
            )
        failed_imports = st.session_state.get("failed_imports") or []
        if failed_imports:
            st.warning(f"有 {len(failed_imports)} 个文件失败，原文件内容仍保留在本次会话中。")
            if st.button("仅重试失败文件"):
                run_import_tasks(failed_imports)

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
            with st.expander("编辑所选文档的版本与有效期"):
                st.caption(
                    "修改后，回答会优先参考现在正在使用的文件，并在旧文件来源旁显示提醒。"
                    "只修改这些日期和状态即可，不需要重新处理文档内容。"
                )
                with st.form(f"metadata_{selected_id}"): 
                    edit_year = st.text_input(
                        "学年度", value=selected.get("academic_year", "")
                    )
                    edit_dates = st.columns(2)
                    edit_effective = edit_dates[0].text_input(
                        "生效日期", value=selected.get("effective_date", "")
                    )
                    edit_expiry = edit_dates[1].text_input(
                        "失效日期", value=selected.get("expiry_date", "")
                    )
                    edit_current = st.checkbox(
                        "现行有效", value=selected.get("is_current", True)
                    )
                    edit_replaces = st.text_input(
                        "替代的旧文档", value=selected.get("replaces", "")
                    )
                    if st.form_submit_button("保存版本信息"):
                        try:
                            rag.set_document_metadata(
                                user, selected["group"], selected["filename"],
                                edit_year, edit_effective, edit_expiry,
                                edit_current, edit_replaces,
                            )
                            st.success("版本信息已保存。")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"保存失败：{exc}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("仅重新向量化此文档", use_container_width=True):
                    try:
                        with st.spinner("正在更新当前文档的向量……"):
                            count = rag.index_document(
                                user, selected["group"], selected["filename"], use_ocr=enable_ocr
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

    with tab_map["自动诊断"]:
        st.subheader("权限与检索自动诊断")
        st.caption("检查不会删除你的文件或搜索内容；测试学生能否打开无权限文件时，会留下一条检查记录。")
        with st.expander("功能说明与结果解读"):
            st.markdown(
                "点击“运行全部诊断”，系统会自动检查：不同身份能看哪些文件、学生是否无法打开无权限文件、"
                "文件是否放错分组、文件删除后是否还能被搜到，以及两种找资料的方式是否正常。\n\n"
                "- **通过**：这一项没有发现异常。\n"
                "- **失败**：按照后面的说明检查文件、权限或模型服务。\n"
                "- **跳过**：目前没有足够文件做这项测试，例如没有管理员文件时，就不能测试学生是否会被拦截。\n\n"
                "这些检查用来帮助发现问题，不能保证系统完全没有其他问题。"
            )
        if st.button("运行全部诊断", type="primary"): 
            with st.spinner("正在执行权限、索引一致性和双路检索检查……"):
                st.session_state.diagnostic_checks = rag.run_diagnostics(user)
        checks = st.session_state.get("diagnostic_checks") or []
        if checks:
            passed = sum(item["结果"] == "通过" for item in checks)
            failed = sum(item["结果"] == "失败" for item in checks)
            skipped = sum(item["结果"] == "跳过" for item in checks)
            cols = st.columns(3)
            cols[0].metric("通过", passed)
            cols[1].metric("失败", failed)
            cols[2].metric("跳过", skipped)
            st.dataframe(checks, use_container_width=True, hide_index=True)
            if failed:
                st.error("存在失败项，请根据说明检查权限或索引。")
            else:
                st.success("未发现失败项。")

    with tab_map["审计日志"]:
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