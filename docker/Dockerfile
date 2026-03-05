# ============ 构建阶段 ============
FROM python:3.12-slim AS builder

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 加速依赖管理
RUN pip install uv

COPY sources.list /etc/apt/

# 复制依赖文件
COPY pyproject.toml uv.lock* ./

# 安装生产依赖（包含 PostgreSQL 和 Server 支持）
RUN uv sync --extra cockroachdb --extra server --no-dev

# ============ 运行阶段 ============
FROM python:3.12-slim

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

# 从构建阶段复制虚拟环境
COPY --from=builder /app/.venv /app/.venv

# 复制应用代码
COPY memori/ /app/memori/
COPY pyproject.toml /app/

# 添加 venv 到 PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8887

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8887/health || exit 1

# 启动服务
CMD ["python", "-m", "memori.server"]
