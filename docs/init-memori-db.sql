-- ============================================================
-- SAA Memori 数据库初始化脚本
-- ============================================================
-- 在 saa-backend 的 PostgreSQL 实例中创建 saa_memori 数据库
--
-- 执行方式：
-- 方式1: psql -h localhost -p 15432 -U postgres -f init-memori-db.sql
-- 方式2: 在 DBeaver/pgAdmin 等工具中执行
--
-- 注意：
-- 1. 数据库只需创建一次
-- 2. Memori 的表结构会在服务首次启动时自动创建（通过 memori.config.storage.build()）
-- ============================================================

-- 检查数据库是否已存在，不存在则创建
SELECT 'CREATE DATABASE saa_memori'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'saa_memori')\gexec

\echo '========================================'
\echo 'saa_memori 数据库创建完成'
\echo '========================================'
\echo ''
\echo 'Memori 表结构将在服务首次启动时自动创建，包括：'
\echo '  - memori_conversation         (会话)'
\echo '  - memori_conversation_message (会话消息)'
\echo '  - memori_entity               (实体/用户)'
\echo '  - memori_entity_fact          (实体事实/记忆)'
\echo '  - memori_process              (进程/Agent)'
\echo '  - memori_process_attribute    (进程属性)'
\echo '  - memori_session              (会话)'
\echo '  - memori_knowledge_graph      (知识图谱)'
\echo '  - memori_schema_version       (Schema版本)'
\echo ''
