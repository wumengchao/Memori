"""
SAA Memori Service - HTTP API Server

提供记忆服务的 HTTP 接口，供 saa-agent-gateway 调用
"""

from memori.server.app import app, run_server

__all__ = ["app", "run_server"]
