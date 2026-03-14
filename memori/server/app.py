"""
SAA Memori Service - FastAPI Application

提供记忆服务的 HTTP 接口：
- 记忆检索：根据用户查询返回相关记忆上下文
- 记忆存储：后台异步存储对话记忆
- 健康检查：服务状态检测
"""

import os
import logging
import re
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memori import Memori
from memori.search._types import FactSearchResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("saa-memori")

# 全局 Memori 实例
_memori_instance: Optional[Memori] = None
_augmentation_enabled: bool = False


def is_augmentation_enabled() -> bool:
    """
    增强开关策略：
    1) MEMORI_AUGMENTATION_ENABLED 显式配置优先（true/false）
    2) 未显式配置时：仅当 MEMORI_API_KEY 非空才启用远端增强
    """
    explicit = os.getenv("MEMORI_AUGMENTATION_ENABLED")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return bool((os.getenv("MEMORI_API_KEY") or "").strip())


def rollback_memori_adapter(memori: Optional[Memori]) -> None:
    """尝试回滚 Memori 底层 DB 会话，防止连接进入 invalid transaction 状态。"""
    if memori is None:
        return
    try:
        adapter = memori.config.storage.adapter
    except Exception:
        return

    rollback_targets = []
    conn = getattr(adapter, "conn", None)
    if conn is not None:
        rollback_targets.append(conn)
    rollback_targets.append(adapter)

    for target in rollback_targets:
        if target is None or not hasattr(target, "rollback"):
            continue
        try:
            target.rollback()
            return
        except Exception:
            continue


def persist_conversation_records(
    memori: Memori,
    *,
    entity_id: str,
    process_id: Optional[str],
    session_uuid: str,
    messages: list[dict],
) -> int:
    """先写入会话与对话消息，再触发增强，避免仅 enqueue 导致对话表为空。"""
    effective_process_id = process_id or "saa-agent-gateway"

    with memori.config.storage.conn as (_conn, adapter, driver):
        entity_db_id = driver.entity.create(entity_id)
        process_db_id = driver.process.create(effective_process_id)

        session_db_id = driver.session.read(session_uuid)
        if session_db_id is None:
            session_db_id = driver.session.create(
                session_uuid,
                entity_db_id,
                process_db_id,
            )

        conversation_db_id = driver.conversation.read_id_by_session_id(session_db_id)
        if conversation_db_id is None:
            conversation_db_id = driver.conversation.create(session_db_id, 60)

        for msg in messages:
            role = str(msg.get("role", "user") or "user")
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue
            if role not in ("system", "user", "assistant", "tool"):
                role = "user"
            driver.conversation.message.create(conversation_db_id, role, "text", content)

        adapter.commit()

    return conversation_db_id


def normalize_basic_fact_texts(messages: list[dict]) -> list[str]:
    """
    基础记忆模式：直接从对话消息抽取可检索文本（不依赖远端增强）。
    """
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _is_low_information_chunk(text: str, *, min_chars: int) -> bool:
        content = (text or "").strip()
        if not content:
            return True

        # 太短的文本通常是确认/寒暄，信息密度低。
        if len(content) < min_chars:
            return True

        # 没有有效文字/数字，通常是噪声（标点、符号、emoji 等）。
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", content):
            return True

        normalized = re.sub(r"[\s，,。.!！？?；;：:~～、…“”\"'`·\-_/\\|]+", "", content).lower()
        if not normalized:
            return True

        # 常见低信息确认语。
        low_info_set = {
            "好",
            "好的",
            "好的呢",
            "收到",
            "明白",
            "嗯",
            "嗯嗯",
            "哦",
            "ok",
            "okay",
            "行",
            "可以",
            "是的",
            "知道了",
            "没问题",
            "安排",
            "继续",
            "谢谢",
            "辛苦了",
        }
        if normalized in low_info_set:
            return True

        # 高频重复字符（如“哈哈哈哈”“嗯嗯嗯嗯”）判为低信息。
        if len(normalized) >= 8:
            uniq = set(normalized)
            max_count = max(normalized.count(ch) for ch in uniq)
            if len(uniq) <= 3 and (max_count / len(normalized)) >= 0.6:
                return True

        return False

    def _split_text_chunks(text: str, max_chars: int, overlap: int) -> list[str]:
        content = (text or "").strip()
        if not content:
            return []
        if len(content) <= max_chars:
            return [content]

        # 先按中英文句末标点切句，再按窗口拼接；避免简单按字节截断破坏语义。
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[。！？!?；;.\n])\s*", content)
            if s and s.strip()
        ]
        if not sentences:
            sentences = [content]

        chunks: list[str] = []
        current = ""
        for s in sentences:
            if not current:
                current = s
                continue
            if len(current) + 1 + len(s) <= max_chars:
                current = f"{current} {s}"
                continue
            chunks.append(current)
            if overlap > 0:
                tail = current[-overlap:]
                current = f"{tail} {s}" if tail else s
            else:
                current = s
        if current:
            chunks.append(current)

        return [c.strip() for c in chunks if c and c.strip()]

    max_chars = int(os.getenv("MEMORI_BASIC_FACT_CHUNK_SIZE", "400") or "400")
    overlap = int(os.getenv("MEMORI_BASIC_FACT_CHUNK_OVERLAP", "40") or "40")
    min_chars = int(os.getenv("MEMORI_BASIC_FACT_MIN_CHARS", "8") or "8")
    filter_low_info = _env_bool("MEMORI_BASIC_FACT_FILTER_LOW_INFO", True)
    store_assistant = _env_bool("MEMORI_BASIC_FACT_STORE_ASSISTANT", True)
    max_chars = max(120, min(2000, max_chars))
    overlap = max(0, min(max_chars // 2, overlap))
    min_chars = max(2, min(80, min_chars))

    texts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        if role == "assistant" and not store_assistant:
            continue
        content = str(msg.get("content", "") or "").strip()
        if not content:
            continue
        role_prefix = "用户" if role == "user" else "助手"
        chunks = _split_text_chunks(content, max_chars=max_chars, overlap=overlap)
        if not chunks:
            continue
        for chunk in chunks:
            if filter_low_info and _is_low_information_chunk(chunk, min_chars=min_chars):
                continue
            texts.append(f"{role_prefix}: {chunk}")
    # 去重但保持顺序
    seen = set()
    uniq_texts: list[str] = []
    for t in texts:
        if t in seen:
            continue
        seen.add(t)
        uniq_texts.append(t)
    return uniq_texts


def persist_basic_facts(
    memori: Memori,
    *,
    entity_id: str,
    messages: list[dict],
) -> int:
    """
    在增强关闭时，将消息文本直接向量化并写入 memori_entity_fact。
    返回成功写入的 fact 数量。
    """
    facts = normalize_basic_fact_texts(messages)
    if not facts:
        return 0

    embedder = get_embedder()
    embeddings = embedder.embed(facts)
    if not embeddings:
        return 0

    count = min(len(facts), len(embeddings))
    if count <= 0:
        return 0
    if len(embeddings) != len(facts):
        logger.warning(
            "基础记忆向量化数量不匹配 facts=%d embeddings=%d，将按最小数量写入",
            len(facts),
            len(embeddings),
        )

    with memori.config.storage.conn as (_conn, adapter, driver):
        entity_db_id = driver.entity.create(entity_id)
        driver.entity_fact.create(entity_db_id, facts[:count], embeddings[:count])
        adapter.commit()

    return count


class VolcanoEmbedder:
    """火山引擎 Embeddings API 客户端"""

    def __init__(self, api_key: str, base_url: str, model: str, api_path: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_path = api_path if api_path.startswith("/") else f"/{api_path}"
        self.client = httpx.Client(timeout=30.0)

    def _embed_request(self, texts: list[str]) -> list[list[float]]:
        """单次请求并解析 embeddings 结果。"""
        if not texts:
            return []

        url = f"{self.base_url}{self.api_path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        use_multimodal_input = "multimodal" in self.api_path.lower()
        if use_multimodal_input:
            model_input = [{"type": "text", "text": text} for text in texts]
        else:
            model_input = texts

        payload = {
            "model": self.model,
            "input": model_input,
            "encoding_type": "float",
        }

        response = self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        data_field = data.get("data")

        if isinstance(data_field, list):
            return [item["embedding"] for item in data_field if "embedding" in item]
        if isinstance(data_field, dict) and "embedding" in data_field:
            return [data_field["embedding"]]
        return []

    def embed(self, texts: list[str]) -> list[list[float]]:
        """生成文本的向量表示"""
        try:
            vectors = self._embed_request(texts)
            if len(vectors) == len(texts):
                return vectors

            # 兼容部分接口在批量输入时仅返回单向量：降级为逐条请求，保证 1:1 对齐。
            if len(texts) > 1:
                logger.warning(
                    "Embeddings 批量返回数量异常 texts=%d vectors=%d，降级逐条请求",
                    len(texts),
                    len(vectors),
                )
                per_item_vectors: list[list[float]] = []
                for text in texts:
                    single = self._embed_request([text])
                    if single:
                        per_item_vectors.append(single[0])
                return per_item_vectors

            return vectors
        except Exception as e:
            logger.error(f"火山引擎 Embeddings 调用失败: {e}")
            raise


class LocalSentenceTransformerEmbedder:
    """本地 Sentence Transformers embedder（备用，需安装 optional: embedding-local）"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "本地 embeddings 需要安装 sentence-transformers。"
                "请配置火山引擎 VOLCANO_EMBEDDING_API_KEY/VOLCANO_EMBEDDING_BASE_URL，"
                "或安装可选依赖: uv sync --extra embedding-local"
            ) from e
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


def get_embedder():
    """获取 embedder 实例"""
    # 优先使用火山引擎 Embeddings
    volcano_api_key = os.getenv("VOLCANO_EMBEDDING_API_KEY")
    volcano_base_url = os.getenv("VOLCANO_EMBEDDING_BASE_URL")
    volcano_model = os.getenv("VOLCANO_EMBEDDING_MODEL", "doubao-embedding-vision-251215")
    volcano_api_path = os.getenv("VOLCANO_EMBEDDING_API_PATH", "/embeddings/multimodal")

    if volcano_api_key and volcano_base_url:
        logger.info(f"使用火山引擎 Embeddings: {volcano_base_url}/{volcano_model}")
        return VolcanoEmbedder(
            api_key=volcano_api_key,
            base_url=volcano_base_url,
            model=volcano_model,
            api_path=volcano_api_path,
        )

    # 备用：使用本地模型（需安装 optional embedding-local）
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "未配置火山引擎 Embeddings 且未安装本地 embedder。"
            "请设置 VOLCANO_EMBEDDING_API_KEY 与 VOLCANO_EMBEDDING_BASE_URL，"
            "或安装可选依赖: uv sync --extra embedding-local"
        )
    logger.info("使用本地 Sentence Transformers 模型")
    return LocalSentenceTransformerEmbedder()


def get_memori() -> Memori:
    """获取 Memori 单例实例"""
    global _memori_instance
    if _memori_instance is None:
        raise RuntimeError("Memori 未初始化")
    return _memori_instance


def get_database_url() -> str:
    """获取数据库连接字符串"""
    # 优先使用 DATABASE_URL 环境变量
    url = os.getenv("DATABASE_URL")
    if url:
        # 将 postgresql:// 替换为 postgresql+psycopg:// 以使用 psycopg v3 驱动
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    # 否则从分离的环境变量构建
    host = os.getenv("POSTGRES_HOST", "host.docker.internal")
    port = os.getenv("POSTGRES_PORT", "15432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "your-super-secret-and-long-postgres-password")
    dbname = os.getenv("POSTGRES_DB", "saa_memori")

    # 使用 psycopg (v3) 驱动
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"


def create_db_connection():
    """创建数据库连接工厂函数"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    database_url = get_database_url()
    logger.info(f"连接数据库: {database_url.split('@')[1] if '@' in database_url else database_url}")

    engine = create_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    return sessionmaker(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _memori_instance, _augmentation_enabled

    logger.info("正在初始化 Memori 服务...")

    try:
        # 创建数据库连接工厂
        session_factory = create_db_connection()

        # 初始化 Memori
        _memori_instance = Memori(conn=session_factory)

        # 初始化数据库 Schema
        _memori_instance.config.storage.build()
        logger.info("Memori 数据库 Schema 初始化完成")

        # 配置 embeddings
        volcano_api_key = os.getenv("VOLCANO_EMBEDDING_API_KEY")
        volcano_base_url = os.getenv("VOLCANO_EMBEDDING_BASE_URL")

        if volcano_api_key and volcano_base_url:
            # 使用火山引擎 embeddings：monkey patch embed_texts
            from memori.embeddings import _api as embeddings_api
            volcano_model = os.getenv("VOLCANO_EMBEDDING_MODEL", "doubao-embedding-vision-251215")
            volcano_api_path = os.getenv("VOLCANO_EMBEDDING_API_PATH", "/embeddings/multimodal")
            embedder = VolcanoEmbedder(
                api_key=volcano_api_key,
                base_url=volcano_base_url,
                model=volcano_model,
                api_path=volcano_api_path,
            )

            def custom_embed_texts(texts, model, **kwargs):
                """自定义 embeddings 函数，使用火山引擎 API"""
                return embedder.embed(texts if isinstance(texts, list) else [texts])

            # Monkey patch
            embeddings_api._embed_texts = custom_embed_texts
            logger.info(f"已配置火山引擎 Embeddings: {volcano_base_url}{volcano_api_path} model={volcano_model}")
        else:
            logger.info("使用默认本地 embeddings 模型")

        _augmentation_enabled = is_augmentation_enabled()
        if _augmentation_enabled:
            _memori_instance.augmentation.start(session_factory)
            logger.info("Memori Augmentation 后台处理已启动")
        else:
            logger.info("Memori Augmentation 已禁用（MEMORI_API_KEY 为空或开关关闭）")

        logger.info("Memori 服务启动成功")

        yield

    except Exception as e:
        logger.error(f"Memori 服务初始化失败: {e}")
        raise
    finally:
        # 清理资源
        if _memori_instance:
            _memori_instance.close()
            logger.info("Memori 服务已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="SAA Memori Service",
    description="记忆服务 HTTP API，为 saa-agent-gateway 提供记忆检索和存储能力",
    version="1.0.0",
    lifespan=lifespan,
)


# ============ 请求/响应模型 ============

class RecallRequest(BaseModel):
    """记忆检索请求"""
    query: str = Field(..., description="检索查询文本", min_length=1, max_length=10000)
    entity_id: str = Field(..., description="用户/实体ID", min_length=1, max_length=100)
    process_id: Optional[str] = Field(None, description="Agent/进程ID", max_length=100)
    limit: Optional[int] = Field(10, description="返回结果数量限制", ge=1, le=100)


class MemoryItem(BaseModel):
    """单个记忆项"""
    content: str = Field(..., description="记忆内容")
    score: float = Field(..., description="相关性分数")
    memory_type: Optional[str] = Field(None, description="记忆类型")
    created_at: Optional[str] = Field(None, description="创建时间")


class RecallResponse(BaseModel):
    """记忆检索响应"""
    success: bool = Field(..., description="是否成功")
    memories: list[MemoryItem] = Field(default_factory=list, description="记忆列表")
    context: Optional[str] = Field(None, description="拼接后的上下文文本")
    message: Optional[str] = Field(None, description="附加消息")


class AttributionRequest(BaseModel):
    """设置归属标记请求"""
    entity_id: str = Field(..., description="用户/实体ID", min_length=1, max_length=100)
    process_id: Optional[str] = Field(None, description="Agent/进程ID", max_length=100)


class AttributionResponse(BaseModel):
    """设置归属标记响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="结果消息")


class NewSessionRequest(BaseModel):
    """新建会话请求"""
    entity_id: Optional[str] = Field(None, description="用户/实体ID")


class NewSessionResponse(BaseModel):
    """新建会话响应"""
    success: bool = Field(..., description="是否成功")
    session_id: str = Field(..., description="新会话ID")
    message: str = Field(..., description="结果消息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="服务版本")
    database: str = Field(..., description="数据库连接状态")


class StoreRequest(BaseModel):
    """记忆存储请求"""
    entity_id: str = Field(..., description="用户/实体ID", min_length=1, max_length=100)
    process_id: Optional[str] = Field(None, description="Agent/进程ID", max_length=100)
    session_id: Optional[str] = Field(None, description="会话ID")
    messages: Optional[list[dict]] = Field(None, description="对话消息列表 [{role, content}]")
    user_message: Optional[str] = Field(None, description="用户消息（用于快速存储）")
    assistant_message: Optional[str] = Field(None, description="助手回复（用于快速存储）")


class StoreResponse(BaseModel):
    """记忆存储响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="结果消息")
    session_id: Optional[str] = Field(None, description="会话ID")


def _normalize_recall_item(item: object) -> tuple[str, float, Optional[str], Optional[str]]:
    """兼容 FactSearchResult / dict / 字符串 三种 recall 返回结构。"""
    if isinstance(item, FactSearchResult):
        score = float(item.rank_score if item.rank_score is not None else item.similarity)
        return item.content, score, "fact", item.date_created

    if isinstance(item, dict):
        content = str(item.get("content", "") or "")
        raw_score = item.get("score", item.get("rank_score", item.get("similarity", 0.0)))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        memory_type = item.get("memory_type")
        created_at = item.get("created_at", item.get("date_created"))
        return content, score, str(memory_type) if memory_type is not None else None, str(created_at) if created_at is not None else None

    if isinstance(item, str):
        return item, 0.0, "fact", None

    return "", 0.0, None, None


_IDENTITY_FACT_PATTERNS = (
    re.compile(r"^self\s+is\s+(家长|parent|father|mother)\b", re.IGNORECASE),
    re.compile(r"^self\s+has\s+(id|name|role)\b", re.IGNORECASE),
)
_IDENTITY_QUERY_HINTS = (
    "你是谁",
    "我是谁",
    "身份",
    "角色",
    "叫什么",
    "家长",
    "parent",
    "father",
    "mother",
    "name",
    "role",
    "id",
)


def _is_identity_fact(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    return any(p.search(text) for p in _IDENTITY_FACT_PATTERNS)


def _is_identity_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(hint in q for hint in _IDENTITY_QUERY_HINTS)


_ROLE_PREFIX_RE = re.compile(r"^\s*(用户|助手)\s*:\s*", re.IGNORECASE)
_LEADING_ROLE_JSON_RE = re.compile(
    r'^\s*\{[^{}]*"角色"\s*:\s*"[^"]+"[^{}]*\}\s*',
    re.DOTALL,
)
_TASK_CONTEXT_BLOCK_RE = re.compile(
    r"^\s*<task_context>[\s\S]*?</task_context>\s*",
    re.IGNORECASE,
)


def _strip_recall_noise_prefix(content: str) -> str:
    """
    清理 recall 文本前缀中的模板噪声（角色 JSON / task_context）。
    仅做前缀清理，不改正文语义。
    """
    text = (content or "").strip()
    if not text:
        return ""

    role_prefix = ""
    role_match = _ROLE_PREFIX_RE.match(text)
    if role_match:
        role_prefix = f"{role_match.group(1)}: "
        text = text[role_match.end() :].lstrip()

    changed = True
    while changed and text:
        changed = False
        m_role = _LEADING_ROLE_JSON_RE.match(text)
        if m_role:
            text = text[m_role.end() :].lstrip()
            changed = True
        m_task = _TASK_CONTEXT_BLOCK_RE.match(text)
        if m_task:
            text = text[m_task.end() :].lstrip()
            changed = True

    if not text:
        return ""
    if role_prefix:
        return role_prefix + text
    return text


def _canonical_recall_key(content: str) -> str:
    """
    为 recall 去重生成稳定 key：去前缀模板后，再去空白/标点差异。
    """
    base = _strip_recall_noise_prefix(content)
    if not base:
        return ""
    base = _ROLE_PREFIX_RE.sub("", base).lower()
    # 只保留中文、英文、数字，消除格式差异导致的“伪不同”
    key = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", base)
    return key[:240]


def _dedupe_recall_items(
    items: list[tuple[str, float, Optional[str], Optional[str]]],
    limit: int,
) -> list[tuple[str, float, Optional[str], Optional[str]]]:
    """
    recall 结果去重：保持原排序，按清洗后文本 key 去重。
    """
    deduped: list[tuple[str, float, Optional[str], Optional[str]]] = []
    seen_keys: set[str] = set()

    for content, score, memory_type, created_at in items:
        cleaned = _strip_recall_noise_prefix(content)
        if not cleaned:
            continue
        key = _canonical_recall_key(cleaned)
        if not key:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append((cleaned, score, memory_type, created_at))
        if len(deduped) >= limit:
            break

    return deduped


def _get_recall_fetch_limit(limit: int) -> int:
    """
    召回候选扩充倍率：
    - 优先使用 Memori 内置排序，多取候选供后续去重/阈值过滤
    - 默认倍率 2，最大 5，最终上限 100
    """
    raw = os.getenv("MEMORI_RECALL_FETCH_MULTIPLIER", "2")
    try:
        mul = int(raw)
    except ValueError:
        mul = 2
    mul = max(1, min(5, mul))
    return min(100, max(limit, limit * mul))


def _get_recall_min_score() -> float:
    """
    recall 最低分阈值（基于 Memori rank_score / similarity）。
    默认 0.18（在噪声较多的语音场景更稳），可通过环境变量覆盖：
    MEMORI_RECALL_MIN_SCORE=0.18
    """
    raw = os.getenv("MEMORI_RECALL_MIN_SCORE", "0.18")
    try:
        score = float(raw)
    except ValueError:
        score = 0.0
    return max(0.0, min(1.0, score))


def _get_recall_min_score_ratio() -> float:
    """
    相对阈值：保留 score >= (top_score * ratio) 的候选。
    默认 0.45，值越大召回越“聚焦”。
    """
    raw = os.getenv("MEMORI_RECALL_MIN_SCORE_RATIO", "0.45")
    try:
        ratio = float(raw)
    except ValueError:
        ratio = 0.45
    return max(0.0, min(1.0, ratio))


# ============ API 端点 ============

@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    try:
        memori = get_memori()
        # 简单测试数据库连接 - 执行一个简单查询
        adapter = memori.config.storage.adapter
        if adapter and adapter.conn:
            # 尝试执行简单查询测试连接
            try:
                adapter.execute("SELECT 1")
                db_status = "connected"
            except Exception as db_err:
                logger.warning(f"数据库连接测试失败: {db_err}")
                rollback_memori_adapter(memori)
                db_status = f"error: {str(db_err)[:50]}"
        else:
            db_status = "no_connection"
    except Exception as e:
        logger.warning(f"健康检查失败: {e}")
        try:
            rollback_memori_adapter(get_memori())
        except Exception:
            pass
        db_status = f"error: {str(e)[:50]}"

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        service="saa-memori",
        version="1.0.0",
        database=db_status,
    )


@app.post("/api/v1/recall", response_model=RecallResponse, tags=["记忆操作"])
async def recall_memories(request: RecallRequest):
    """
    检索记忆

    根据查询文本检索与指定用户相关的记忆，返回相关性最高的记忆列表。
    返回的 context 字段是拼接后的文本，可直接注入到 LLM 提示词中。
    """
    try:
        memori = get_memori()

        # 设置归属标记
        memori.attribution(
            entity_id=request.entity_id,
            process_id=request.process_id,
        )

        effective_limit = request.limit or 10
        fetch_limit = _get_recall_fetch_limit(effective_limit)

        # 执行检索（优先依赖 Memori 内置混合排序，先扩候选）
        results = memori.recall(query=request.query, limit=fetch_limit)

        # 转换结果
        memories = []
        context_parts = []
        normalized_items = []

        for item in results:
            content, score, memory_type, created_at = _normalize_recall_item(item)
            if not content:
                continue
            normalized_items.append((content, score, memory_type, created_at))

        # 对非身份类问题，降低“self is 家长/parent”等身份事实的主导性，优先返回主题记忆
        if normalized_items and not _is_identity_query(request.query):
            non_identity = [it for it in normalized_items if not _is_identity_fact(it[0])]
            identity = [it for it in normalized_items if _is_identity_fact(it[0])]
            if non_identity:
                normalized_items = non_identity + identity[:1]

        min_score = _get_recall_min_score()
        min_score_ratio = _get_recall_min_score_ratio()
        if normalized_items:
            top_score = float(normalized_items[0][1])
            ratio_floor = top_score * min_score_ratio
            final_min_score = max(min_score, ratio_floor)

            before = len(normalized_items)
            normalized_items = [it for it in normalized_items if float(it[1]) >= final_min_score]
            logger.debug(
                "Memori recall 最低分过滤已应用 min_score=%.3f ratio=%.2f top=%.3f final=%.3f before=%d after=%d",
                min_score,
                min_score_ratio,
                top_score,
                final_min_score,
                before,
                len(normalized_items),
            )

        # 去模板前缀噪声并按语义 key 去重，减少重复上下文污染。
        normalized_items = _dedupe_recall_items(normalized_items, effective_limit)

        # 应用 limit，保持接口语义稳定
        for content, score, memory_type, created_at in normalized_items[:effective_limit]:
            memory = MemoryItem(
                content=content,
                score=score,
                memory_type=memory_type,
                created_at=created_at,
            )
            memories.append(memory)
            context_parts.append(memory.content)

        # 拼接上下文
        context = "\n".join(context_parts) if context_parts else None

        return RecallResponse(
            success=True,
            memories=memories,
            context=context,
            message=f"检索到 {len(memories)} 条记忆",
        )

    except Exception as e:
        logger.error(f"记忆检索失败: {e}")
        try:
            rollback_memori_adapter(memori if "memori" in locals() else None)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"记忆检索失败: {str(e)}")


@app.post("/api/v1/store", response_model=StoreResponse, tags=["记忆操作"])
async def store_conversation(request: StoreRequest):
    """
    存储对话记忆

    将对话内容存储到 Memori，后台异步提取记忆（facts、preferences 等）。
    支持两种方式：
    1. 完整消息列表：传入 messages 数组
    2. 快速存储：传入 user_message 和 assistant_message

    记忆提取是异步的，可通过 /api/v1/wait 等待完成。
    """
    try:
        from memori.memory.augmentation.input import AugmentationInput
        from memori.memory.augmentation._message import ConversationMessage
        import uuid

        memori = get_memori()

        # 设置归属标记
        memori.attribution(
            entity_id=request.entity_id,
            process_id=request.process_id,
        )

        # 恢复会话（如果提供了 session_id）
        if request.session_id:
            try:
                memori.set_session(uuid.UUID(request.session_id))
            except (ValueError, TypeError):
                pass  # 无效的 session_id，使用新会话

        # 构建消息列表
        messages = request.messages
        if not messages and request.user_message:
            # 快速存储模式
            messages = [{"role": "user", "content": request.user_message}]
            if request.assistant_message:
                messages.append({"role": "assistant", "content": request.assistant_message})

        if not messages:
            return StoreResponse(
                success=False,
                message="没有可存储的对话内容",
                session_id=str(memori.config.session_id),
            )

        # 先持久化基础会话/对话记录，确保 DB 有可追踪数据
        session_id = str(memori.config.session_id)
        conversation_db_id = persist_conversation_records(
            memori,
            entity_id=request.entity_id,
            process_id=request.process_id,
            session_uuid=session_id,
            messages=messages,
        )

        # 转换为 Memori 的 ConversationMessage 格式（用于增强管线）
        conversation_messages = [
            ConversationMessage(role=msg.get("role", "user"), content=msg.get("content", ""))
            for msg in messages
        ]

        # 触发增强任务：提取事实并更新总结（可配置关闭）
        if _augmentation_enabled:
            augmentation_input = AugmentationInput(
                conversation_id=str(conversation_db_id),
                entity_id=request.entity_id,
                process_id=request.process_id,
                conversation_messages=conversation_messages,
            )
            memori.augmentation.enqueue(augmentation_input)
        else:
            written = persist_basic_facts(
                memori,
                entity_id=request.entity_id,
                messages=messages,
            )
            logger.info("基础记忆模式已写入 facts=%d entity_id=%s", written, request.entity_id)

        logger.info(f"已存储对话 entity_id={request.entity_id} session_id={session_id} messages={len(messages)}")

        return StoreResponse(
            success=True,
            message=f"对话已存储，共 {len(messages)} 条消息",
            session_id=session_id,
        )

    except Exception as e:
        logger.error(f"存储对话失败: {e}")
        try:
            rollback_memori_adapter(memori if "memori" in locals() else None)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"存储对话失败: {str(e)}")


@app.post("/api/v1/attribution", response_model=AttributionResponse, tags=["记忆操作"])
async def set_attribution(request: AttributionRequest):
    """
    设置归属标记

    设置当前会话的归属标记（entity_id 和 process_id）。
    后续的记忆操作将关联到这个归属标记。
    """
    try:
        memori = get_memori()
        memori.attribution(
            entity_id=request.entity_id,
            process_id=request.process_id,
        )

        return AttributionResponse(
            success=True,
            message=f"归属标记已设置: entity_id={request.entity_id}, process_id={request.process_id}",
        )

    except Exception as e:
        logger.error(f"设置归属标记失败: {e}")
        try:
            rollback_memori_adapter(memori if "memori" in locals() else None)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"设置归属标记失败: {str(e)}")


@app.post("/api/v1/session/new", response_model=NewSessionResponse, tags=["记忆操作"])
async def new_session(request: NewSessionRequest = None):
    """
    新建会话

    创建新的记忆会话，返回新的 session_id。
    可选地重新设置 entity_id。
    """
    try:
        memori = get_memori()

        if request and request.entity_id:
            memori.attribution(entity_id=request.entity_id)

        memori.new_session()
        session_id = str(memori.config.session_id)

        return NewSessionResponse(
            success=True,
            session_id=session_id,
            message=f"新会话已创建: {session_id}",
        )

    except Exception as e:
        logger.error(f"新建会话失败: {e}")
        try:
            rollback_memori_adapter(memori if "memori" in locals() else None)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"新建会话失败: {str(e)}")


@app.post("/api/v1/wait", tags=["记忆操作"])
async def wait_for_augmentation():
    """
    等待后台增强完成

    等待 Memori 的后台增强（记忆提取）完成。
    通常在短生命周期的程序中使用，生产环境可以忽略。
    """
    try:
        if not _augmentation_enabled:
            return {"success": True, "message": "增强已禁用，无需等待"}
        memori = get_memori()
        memori.augmentation.wait()

        return {"success": True, "message": "后台增强已完成"}

    except Exception as e:
        logger.error(f"等待增强失败: {e}")
        try:
            rollback_memori_adapter(memori if "memori" in locals() else None)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"等待增强失败: {str(e)}")


def run_server(host: str = "0.0.0.0", port: int = 8887):
    """启动 HTTP 服务器"""
    logger.info(f"启动 SAA Memori 服务: http://{host}:{port}")
    uvicorn.run(
        "memori.server.app:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    run_server()
