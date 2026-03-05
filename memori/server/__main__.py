"""
SAA Memori Service - 命令行入口

支持通过 python -m memori.server 启动服务
"""

from memori.server.app import run_server

if __name__ == "__main__":
    run_server()
