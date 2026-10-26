"""校务通核心逻辑：本地 RAG、RBAC 权限隔离、审计和问答记录。"""

import csv
import hashlib
import hmac
import io
import json
import re
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document as WordDocument
import jieba
from rank_bm25 import BM25Okapi


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
GROUPS = {
    "student": "学生文档",
    "counselor": "辅导员文档",
    "admin": "管理员文档",
}
ROLE_NAMES = {
    "student": "学生",
    "counselor": "辅导员",
    "admin": "管理员",
}
# 角色可访问的独立知识库。所有检索和原文读取均在此再次校验。
ROLE_ACCESS = {
    "student": ("student",),
    "counselor": ("student", "counselor"),
    "admin": ("student", "counselor", "admin"),
}
COLLECTIONS = {
    "student": "campus_student",
    "counselor": "campus_counselor",
    "admin": "campus_admin",
}
QUERY_EXPANSIONS = {
    "缓考": "缓考 考试 申请 条件 材料 流程 审批",
    "补考": "补考 重修 考试 成绩 申请",
    "重修": "重修 补考 选课 学分 成绩",
    "休学": "休学 复学 学籍 申请 流程",
    "复学": "复学 休学 学籍 申请 流程",
    "退学": "退学 学籍 申请 审批",
    "转专业": "转专业 专业 申请 条件 考核",
    "奖学金": "奖学金 评定 申请 条件 金额",
    "助学金": "助学金 困难认定 申请 材料",
    "请假": "请假 销假 审批 辅导员",
    "选课": "选课 退课 补选 学分",
    "成绩": "成绩 查询 复核 绩点 学分",
}
STOPWORDS = {
    "什么", "怎么", "如何", "需要", "可以", "请问", "一下", "知道", "告诉我",
    "本科生", "学生", "学校", "相关", "规定", "办理", "申请", "的", "了", "吗",
}


class CampusRAG:
    """校务通本地核心。actor 必须包含 username 和 role。"""

    def __init__(
        self,
        data_dir: str = "data",
        db_dir: str = "chroma_db",
        system_db: str = "campus_system.db",
        llm_model: str = "qwen2.5:3b",
        embedding_model: str = "bge-m3",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.db_dir = Path(db_dir)
        self.system_db = Path(system_db)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        for group in GROUPS:
            (self.data_dir / GROUPS[group]).mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._vectorstores: dict[str, Chroma] = {}
        self._chunk_cache: dict[str, list[Document]] = {}
        self._bm25_cache: dict[str, tuple[BM25Okapi, list[list[str]]]] = {}
        self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
        self.llm = OllamaLLM(model=llm_model, base_url=ollama_base_url, temperature=0.1)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", " ", ""],
        )
        self._init_system_db()

    # ---------- 通用与权限 ----------

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _role(actor: dict[str, str]) -> str:
        role = str(actor.get("role", ""))
        if role not in ROLE_ACCESS or not actor.get("username"):
            raise PermissionError("登录状态无效，请重新登录。")
        return role

    def _allowed_groups(self, actor: dict[str, str]) -> tuple[str, ...]:
        return ROLE_ACCESS[self._role(actor)]

    def _require_admin(self, actor: dict[str, str]) -> None:
        if self._role(actor) != "admin":
            raise PermissionError("仅管理员可以执行文档管理操作。")

    @staticmethod
    def _safe_filename(filename: str) -> str:
        # 同时处理 Windows 与 POSIX 分隔符，防止路径穿越。
        name = filename.replace("\\", "/").split("/")[-1].strip()
        if not name or name in {".", ".."} or any(ord(c) < 32 for c in name):
            raise ValueError("文件名无效。")
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("仅支持 PDF、TXT 和 DOCX 文件；旧版 DOC 请先另存为 DOCX。")
        return name

    def _path(self, group: str, filename: str) -> Path:
        if group not in GROUPS:
            raise ValueError("未知文档分组。")
        return self.data_dir / GROUPS[group] / self._safe_filename(filename)

    def _vectorstore(self, group: str) -> Chroma:
        if group not in GROUPS:
            raise ValueError("未知知识库分组。")
        if group not in self._vectorstores:
            self._vectorstores[group] = Chroma(
                collection_name=COLLECTIONS[group],
                embedding_function=self.embeddings,
                persist_directory=str(self.db_dir),
            )
        return self._vectorstores[group]

    # ---------- 用户、密码与审计 ----------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.system_db, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_system_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    iterations INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    detail TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS qa_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_role
                    ON audit_logs(role, created_at);
                CREATE INDEX IF NOT EXISTS idx_qa_user
                    ON qa_records(username, created_at);
                """
            )

        # 仅首次运行创建演示账户；修改密码后不会被覆盖。
        defaults = (
            ("student", "student", "Student@123"),
            ("counselor", "counselor", "Counselor@123"),
            ("admin", "admin", "Admin@123"),
        )
        for username, role, password in defaults:
            with self._connect() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM users WHERE username = ?", (username,)
                ).fetchone()
            if not exists:
                salt, digest, iterations = self._hash_password(password)
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
                        (username, role, salt, digest, iterations, self._now()),
                    )

    @staticmethod
    def _hash_password(password: str, salt: str | None = None) -> tuple[str, str, int]:
        iterations = 260_000
        salt = salt or secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations
        ).hex()
        return salt, digest, iterations

    def _audit(
        self,
        username: str,
        role: str,
        action: str,
        resource: str = "",
        detail: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_logs (created_at, username, role, action, resource, detail)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (self._now(), username, role, action, resource, detail[:2000]),
            )

    def authenticate(self, username: str, password: str) -> dict[str, str] | None:
        username = username.strip()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        valid = False
        role = "unknown"
        if row:
            role = row["role"]
            # 使用账户保存的迭代数重新计算，兼容未来调整。
            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(row["salt"]),
                int(row["iterations"]),
            ).hex()
            valid = hmac.compare_digest(candidate, row["password_hash"])

        self._audit(
            username or "空用户名",
            role,
            "login_success" if valid else "login_failed",
            detail="本地账号密码登录",
        )
        return {"username": username, "role": role} if valid else None

    def change_password(
        self, actor: dict[str, str], old_password: str, new_password: str
    ) -> None:
        username = actor["username"]
        self._role(actor)
        if len(new_password) < 8:
            raise ValueError("新密码至少需要 8 个字符。")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        if not row:
            raise PermissionError("账户不存在。")
        old_hash = hashlib.pbkdf2_hmac(
            "sha256",
            old_password.encode("utf-8"),
            bytes.fromhex(row["salt"]),
            int(row["iterations"]),
        ).hex()
        if not hmac.compare_digest(old_hash, row["password_hash"]):
            raise ValueError("原密码不正确。")
        salt, digest, iterations = self._hash_password(new_password)
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET salt = ?, password_hash = ?, iterations = ?"
                " WHERE username = ?",
                (salt, digest, iterations, username),
            )
        self._audit(username, actor["role"], "change_password")

    # ---------- 文档加载与单文档索引 ----------

    @staticmethod
    def _read_txt(path: Path) -> str:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _read_docx(path: Path) -> str:
        """提取 DOCX 正文与表格；不依赖本机安装 Microsoft Word。"""
        word = WordDocument(str(path))
        lines = [p.text.strip() for p in word.paragraphs if p.text.strip()]
        for table in word.tables:
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    lines.append(" | ".join(cells))
        return "\n".join(lines).strip()

    def _load_file(self, group: str, filename: str) -> list[Document]:
        path = self._path(group, filename)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{GROUPS[group]}/{filename}")
        documents: list[Document] = []
        if path.suffix.lower() == ".pdf":
            for page in PyPDFLoader(str(path)).load():
                text = page.page_content.strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": int(page.metadata.get("page", 0)) + 1,
                                "file_type": "pdf",
                            },
                        )
                    )
        elif path.suffix.lower() == ".docx":
            text = self._read_docx(path)
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "file_type": "docx"},
                    )
                )
        else:
            text = self._read_txt(path).strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "file_type": "txt"},
                    )
                )
        if not documents:
            raise ValueError(
                f"{filename} 中没有可读取的文字；扫描版 PDF 或纯图片 Word 需先 OCR。"
            )
        return documents

    def save_document(
        self, actor: dict[str, str], group: str, filename: str, content: bytes
    ) -> str:
        self._require_admin(actor)
        if not content:
            raise ValueError(f"{filename} 是空文件。")
        path = self._path(group, filename)
        path.write_bytes(content)
        self._audit(
            actor["username"], actor["role"], "upload_document",
            f"{group}/{path.name}", f"{len(content)} bytes",
        )
        return path.name

    def index_document(self, actor: dict[str, str], group: str, filename: str) -> int:
        """只删除并重写当前文件的向量，不影响同库其他文档。"""
        self._require_admin(actor)
        filename = self._safe_filename(filename)
        documents = self._load_file(group, filename)
        chunks = [d for d in self.splitter.split_documents(documents) if d.page_content.strip()]
        for index, doc in enumerate(chunks, start=1):
            doc.metadata.update({
                "group": group,
                "source": filename,
                "doc_id": f"{group}/{filename}",
                "chunk": index,
            })
        store = self._vectorstore(group)
        store.delete(where={"source": filename})
        if chunks:
            for start in range(0, len(chunks), 32):
                store.add_documents(chunks[start:start + 32])
        self._chunk_cache.pop(group, None)
        self._bm25_cache.pop(group, None)
        self._audit(
            actor["username"], actor["role"], "index_document",
            f"{group}/{filename}", f"{len(chunks)} 个切片",
        )
        return len(chunks)

    def delete_document(self, actor: dict[str, str], group: str, filename: str) -> None:
        self._require_admin(actor)
        path = self._path(group, filename)
        self._vectorstore(group).delete(where={"source": path.name})
        path.unlink(missing_ok=True)
        self._chunk_cache.pop(group, None)
        self._bm25_cache.pop(group, None)
        self._audit(
            actor["username"], actor["role"], "delete_document", f"{group}/{path.name}"
        )

    def list_documents(self, actor: dict[str, str]) -> list[dict[str, Any]]:
        result = []
        for group in self._allowed_groups(actor):
            folder = self.data_dir / GROUPS[group]
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    result.append({
                        "doc_id": f"{group}/{path.name}",
                        "group": group,
                        "group_name": GROUPS[group],
                        "filename": path.name,
                        "size_kb": round(path.stat().st_size / 1024, 1),
                        "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    })
        return result

    def accessible_chunk_count(self, actor: dict[str, str]) -> int:
        return sum(
            int(self._vectorstore(group)._collection.count())
            for group in self._allowed_groups(actor)
        )

    # ---------- 检索、问答与原文 ----------

    @staticmethod
    def _tokens(text: str) -> list[str]:
        # 修复部分 PDF 在汉字之间插入空格导致的分词断裂。
        text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
        tokens = []
        for token in jieba.lcut(text.lower()):
            token = token.strip()
            if len(token) >= 2 and token not in STOPWORDS and re.search(r"\w", token):
                tokens.append(token)
        return tokens

    @staticmethod
    def _expanded_query(question: str) -> str:
        additions = [value for key, value in QUERY_EXPANSIONS.items() if key in question]
        return question + (" " + " ".join(additions) if additions else "")

    def _group_chunks(self, group: str) -> list[Document]:
        """读取一个权限组的切片用于 BM25；索引变更时会主动失效缓存。"""
        if group not in self._chunk_cache:
            result = self._vectorstore(group).get(include=["documents", "metadatas"])
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            self._chunk_cache[group] = [
                Document(page_content=text, metadata=metadata or {})
                for text, metadata in zip(documents, metadatas)
                if text and (metadata or {}).get("group") == group
            ]
        return self._chunk_cache[group]

    def _keyword_candidates(
        self, group: str, question: str, limit: int
    ) -> list[tuple[Document, float]]:
        docs = self._group_chunks(group)
        if not docs:
            return []
        if group not in self._bm25_cache:
            tokenized = [self._tokens(doc.page_content) for doc in docs]
            if not any(tokenized):
                return []
            self._bm25_cache[group] = (BM25Okapi(tokenized), tokenized)
        bm25, _ = self._bm25_cache[group]
        query_tokens = self._tokens(question)
        if not query_tokens:
            return []
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [
            (docs[index], float(score))
            for index, score in ranked[:limit]
            if score > 0
        ]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        metadata = doc.metadata
        return (
            f"{metadata.get('group')}/{metadata.get('source')}/"
            f"{metadata.get('page')}/{metadata.get('chunk')}"
        )

    @staticmethod
    def _evidence(doc: Document, question: str) -> tuple[str, str, list[str]]:
        """从切片中抽取最关键的 1～2 句，而不是把整段交给用户阅读。"""
        text = re.sub(r"[ \t]+", " ", doc.page_content.strip())
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[。！？；])|\n+", text)
            if part.strip()
        ]
        if not sentences:
            sentences = [text]
        query_tokens = set(self._tokens(question))
        core_terms = [key for key in QUERY_EXPANSIONS if key in question]

        def score(sentence: str) -> int:
            compact = re.sub(r"\s+", "", sentence)
            value = sum(3 if len(token) > 2 else 2 for token in query_tokens if token in compact)
            value += sum(8 for term in core_terms if term in compact)
            return value

        best_index = max(range(len(sentences)), key=lambda i: score(sentences[i]))
        chosen = {best_index}
        if len(sentences) > 1:
            second_index = max(
                (i for i in range(len(sentences)) if i != best_index),
                key=lambda i: score(sentences[i]),
            )
            if score(sentences[second_index]) > 0:
                chosen.add(second_index)
        selected = [sentences[i] for i in sorted(chosen)]
        evidence = " …… ".join(selected)
        if len(evidence) > 360:
            evidence = sentences[best_index][:357] + "……"
        compact_text = re.sub(r"\s+", "", text)
        matched = sorted(
            {token for token in query_tokens if token in compact_text} |
            {term for term in core_terms if term in compact_text}
        )
        return evidence, sentences[best_index], matched[:6]

    def ask(self, actor: dict[str, str], question: str, top_k: int = 4) -> dict[str, Any]:
        allowed = self._allowed_groups(actor)
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空。")
        top_k = max(1, min(int(top_k), 6))
        candidate_limit = max(12, top_k * 4)

        # RRF 融合中文 BM25 与向量检索；关键词略高权重，减少“相似但无关”。
        merged: dict[str, dict[str, Any]] = {}

        def add_candidate(
            doc: Document, rank: int, channel: str, score: float
        ) -> None:
            key = self._doc_key(doc)
            entry = merged.setdefault(key, {
                "doc": doc,
                "rrf": 0.0,
                "keyword_score": 0.0,
                "vector_distance": None,
            })
            weight = 1.35 if channel == "keyword" else 1.0
            entry["rrf"] += weight / (60 + rank + 1)
            if channel == "keyword":
                entry["keyword_score"] = max(entry["keyword_score"], score)
            else:
                entry["vector_distance"] = score

        expanded_query = self._expanded_query(question)
        for group in allowed:
            vector_results = self._vectorstore(group).similarity_search_with_score(
                expanded_query, k=candidate_limit
            )
            for rank, (doc, distance) in enumerate(vector_results):
                if doc.metadata.get("group") == group:
                    add_candidate(doc, rank, "vector", float(distance))
            for rank, (doc, score) in enumerate(
                self._keyword_candidates(group, question, candidate_limit)
            ):
                add_candidate(doc, rank, "keyword", score)

        candidates = list(merged.values())
        core_terms = [key for key in QUERY_EXPANSIONS if key in question]
        if core_terms:
            # 对“缓考、休学”等明确事项，只保留直接出现该事项的证据。
            candidates = [
                entry for entry in candidates
                if any(
                    term in re.sub(r"\s+", "", entry["doc"].page_content)
                    for term in core_terms
                )
            ]
            for entry in candidates:
                entry["rrf"] += 0.04
        else:
            # 有关键词命中时，不再拿纯向量结果凑满来源数量。
            keyword_hits = [entry for entry in candidates if entry["keyword_score"] > 0]
            if keyword_hits:
                candidates = keyword_hits

        candidates.sort(key=lambda entry: entry["rrf"], reverse=True)
        selected = []
        seen_pages = set()
        for entry in candidates:
            doc = entry["doc"]
            page_key = (doc.metadata.get("group"), doc.metadata.get("source"),
                        doc.metadata.get("page"))
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            selected.append(doc)
            if len(selected) >= top_k:
                break

        if not selected:
            topic = "、".join(core_terms) if core_terms else question
            answer = (
                f"未在当前权限范围内的文档中找到与“{topic}”直接相关的规定。"
                "为避免误导，本次不附无关来源。请补充包含该事项的教务制度，"
                "或换用文件中的原文表述再查询。"
            )
            sources: list[dict[str, Any]] = []
        else:
            blocks = []
            sources = []
            for index, doc in enumerate(selected, start=1):
                page = doc.metadata.get("page")
                location = f"，第 {page} 页" if page else ""
                group = str(doc.metadata["group"])
                evidence, highlight, matched = self._evidence(doc, question)
                label = f"{GROUPS[group]}/{doc.metadata['source']}{location}"
                blocks.append(f"[资料{index}] 来源：{label}\n{doc.page_content.strip()}")
                sources.append({
                    "id": index,
                    "group": group,
                    "group_name": GROUPS[group],
                    "source": doc.metadata["source"],
                    "page": page,
                    "chunk": doc.metadata.get("chunk"),
                    "content": doc.page_content.strip(),
                    "evidence": evidence,
                    "highlight": highlight,
                    "matched": matched,
                })

            prompt = f"""你是“校务通”校园教务问答助手。目标是直接解决用户问题，而不是堆砌资料。
1. 只使用与问题直接相关的资料，忽略其他资料，不得用相近制度替代原问题。
2. 第一行先给明确结论；随后用不超过4条短句列出条件、材料或步骤，并标注[资料1]等编号。
3. 如果资料只提到该事项但没有所需条件或流程，直接说明“资料仅提到该事项，未写明具体条件/材料/流程”，不要泛泛回答。
4. 不得编造条款、日期、部门、材料或流程，不复述这些指令。

用户问题：{question}

资料：
{chr(10).join(blocks)}
"""
            answer = str(self.llm.invoke(prompt)).strip()

        sources_json = json.dumps(sources, ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO qa_records (created_at, username, role, question, answer, sources_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (self._now(), actor["username"], actor["role"], question, answer, sources_json),
            )
            record_id = int(cursor.lastrowid)
        self._audit(
            actor["username"], actor["role"], "hybrid_query", ",".join(allowed),
            f"问题：{question[:500]}；命中："
            + "；".join(f"{s['group']}/{s['source']}" for s in sources),
        )
        return {"answer": answer, "sources": sources, "record_id": record_id}

    def open_source(
        self,
        actor: dict[str, str],
        group: str,
        filename: str,
        page: int | None = None,
    ) -> dict[str, str]:
        if group not in self._allowed_groups(actor):
            self._audit(actor.get("username", "?"), actor.get("role", "?"),
                        "access_denied", f"{group}/{filename}", "越权读取原文")
            raise PermissionError("当前角色无权查看该来源文档。")
        documents = self._load_file(group, filename)
        if page is not None:
            pages = [d for d in documents if d.metadata.get("page") == page]
            if not pages:
                raise ValueError("未找到对应 PDF 页面。")
            text = "\n\n".join(d.page_content for d in pages)
            location = f"第 {page} 页"
        else:
            text = "\n\n".join(d.page_content for d in documents)
            location = "DOCX 全文" if Path(filename).suffix.lower() == ".docx" else "文档全文"
        self._audit(
            actor["username"], actor["role"], "view_source",
            f"{group}/{filename}", location,
        )
        return {"text": text, "location": location}

    # ---------- 历史、CSV 与审计查询 ----------

    def get_qa_records(self, actor: dict[str, str], limit: int = 30) -> list[dict[str, Any]]:
        self._role(actor)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_records WHERE username = ? ORDER BY id DESC LIMIT ?",
                (actor["username"], max(1, min(limit, 200))),
            ).fetchall()
        records = []
        for row in rows:
            records.append({
                "id": row["id"],
                "created_at": row["created_at"],
                "question": row["question"],
                "answer": row["answer"],
                "sources": json.loads(row["sources_json"]),
            })
        return records

    @staticmethod
    def _csv_safe(value: Any) -> str:
        """避免用户文本在 Excel 中被当作公式执行。"""
        text = str(value)
        return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text

    def export_qa_csv(self, actor: dict[str, str]) -> bytes:
        self._role(actor)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_records WHERE username = ? ORDER BY id ASC",
                (actor["username"],),
            ).fetchall()
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["时间", "用户名", "角色", "问题", "回答", "原文来源"])
        for row in rows:
            sources = json.loads(row["sources_json"])
            source_text = "\n".join(
                f"[资料{s['id']}] {s['group_name']}/{s['source']}"
                + (f" 第{s['page']}页" if s.get("page") else "")
                + f" 切片{s.get('chunk', '')}\n{s.get('evidence', s['content'])}"
                for s in sources
            )
            writer.writerow([
                self._csv_safe(value) for value in (
                    row["created_at"], row["username"],
                    ROLE_NAMES.get(row["role"], row["role"]),
                    row["question"], row["answer"], source_text,
                )
            ])
        self._audit(actor["username"], actor["role"], "export_qa_csv", detail=f"{len(rows)} 条")
        # UTF-8 BOM，方便 Windows Excel 直接打开而不乱码。
        return output.getvalue().encode("utf-8-sig")

    def get_audit_logs(
        self, actor: dict[str, str], role: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        self._require_admin(actor)
        limit = max(1, min(int(limit), 1000))
        sql = "SELECT * FROM audit_logs"
        params: list[Any] = []
        if role in ROLE_NAMES:
            sql += " WHERE role = ?"
            params.append(role)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]