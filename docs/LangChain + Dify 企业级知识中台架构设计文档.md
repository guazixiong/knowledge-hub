# LangChain + Dify 企业级知识中台架构设计文档

## 🏗️ 系统架构概述

### 整体架构图

```
┌─────────────────────────────────────────────┐
│          数据中台前端（管理界面）              │
│  • 多知识库切换与管理                         │
│  • 文档批量上传                               │
│  • 向量数据监控                               │
│  • 检索效果测试                               │
└───────────────────┬─────────────────────────┘
                    │ HTTP API
┌───────────────────▼─────────────────────────┐
│          数据中台后端（FastAPI）              │
│  • 知识库 CRUD 接口                          │
│  • 文档上传与处理调度                         │
│  • 统一检索接口（供 Dify 调用）               │
│  • 任务队列与状态管理                         │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│          LangChain 处理层                    │
│  • Unstructured 多格式文档解析                │
│  • RecursiveCharacterTextSplitter 智能分块   │
│  • Embedding 向量化（OpenAI/本地模型）        │
│  • Metadata 自动提取与管理                    │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│       向量数据库（持久化存储）                 │
│  • 小规模：ChromaDB (PersistentClient)       │
│  • 大规模：Milvus / Qdrant                   │
│  • 按知识库隔离：./dbs/{kb_name}/            │
└─────────────────────────────────────────────┘

           ↑ 检索 API
           │
┌──────────┴────────┐
│       Dify        │
│  • 通过 API 工具   │
│    调用数据中台     │
│  • 整合检索结果     │
│  • 生成最终答案     │
└───────────────────┘
```

### 核心设计原则

1. **知识库隔离**：每个项目/系统独立知识库，互不干扰
2. **数据持久化**：向量数据存储在磁盘，支持备份与迁移
3. **解耦架构**：数据中台作为中间层，Dify 和 LangChain 通过标准 API 通信
4. **可扩展性**：支持从单机到分布式的平滑升级

------

## 🧩 支持的文档类型

LangChain + Unstructured 支持主流办公格式：

| 文档类型        | 支持程度             | 说明                         |
| --------------- | -------------------- | ---------------------------- |
| .txt            | ✅ 完全支持           | 最推荐，解析简单稳定         |
| .md（Markdown） | ✅ 完全支持           | 能保留标题层级结构           |
| .doc / .docx    | ✅ 完全支持           | Word 文档可以解析段落和标题  |
| .pdf            | ✅ 支持（建议文字版） | 图片型 PDF 需 OCR 才能识别   |
| .html / .htm    | ✅ 完全支持           | 适合存网页内容               |
| .pptx           | ✅ 支持               | 会提取幻灯片文字内容         |
| .csv / .xlsx    | ⚠️ 可选支持           | 内容是结构化表格，语义价值低 |
| .png / .jpg     | ⚠️ 需 OCR             | 需安装 Tesseract 等依赖      |

### 推荐的文档类型使用场景

| 文档场景 | 推荐格式    | 原因                       |
| -------- | ----------- | -------------------------- |
| 需求文档 | .docx, .md  | 章节结构清晰，易于分块     |
| 设计文档 | .docx, .pdf | 模块化明显，chunk 效果最好 |
| 运维文档 | .md, .txt   | 操作步骤明确，适合检索     |
| 技术手册 | .pdf, .html | 内容完整，适合长期归档     |

------

## 🧠 文档内容格式要求

知识中台的语义匹配依赖 embedding，embedding 只能理解文字内容。

### 文档编写最佳实践

| 建议项             | 原因                                       | 示例                                          |
| ------------------ | ------------------------------------------ | --------------------------------------------- |
| 使用清晰的章节层级 | chunk 分段时可以按章节切，提升语义独立性   | `1. 概述 / 1.1 模块功能 / 1.1.1 登录模块`     |
| 控制段落长度       | 每段 5~10 行，太长难以分块，太短语义不完整 | 一个功能点一段，不超过 200 字                 |
| 统一中英文符号     | 避免分句错误                               | 统一使用中文标点或英文标点                    |
| 表格配文字说明     | embedding 可能识别不到表格含义             | "下表展示了三种认证方式的对比：..."           |
| 图片配文字描述     | 图片信息不会被直接识别进向量               | "架构图如下（包含前端、网关、后端三层）：..." |
| 文档元信息完整     | 自动提取 metadata，方便版本管理            | 文档开头写明：系统名、版本号、作者、日期      |

### Chunk 分割策略

每个 chunk 最好控制在：**500 ~ 1500 字**（或 200 ~ 600 tokens）

**示例**：一份《CRM 系统设计文档》

```
1. 系统概述                    ← chunk 1 (800 字)
2. 功能模块设计  
   - 2.1 登录模块              ← chunk 2 (600 字)
   - 2.2 客户管理              ← chunk 3 (1200 字)
   - 2.3 报表分析              ← chunk 4 (900 字)
3. 接口说明                    ← chunk 5 (1100 字)
4. 部署方案                    ← chunk 6 (700 字)
```

------

## 🗂️ 多知识库设计

### 目录结构设计

```
knowledge_platform/
├── knowledge_bases/              # 知识库根目录
│   ├── crm_system/              # CRM 系统知识库
│   │   ├── docs/                # 原始文档存放
│   │   │   ├── requirements/
│   │   │   ├── design/
│   │   │   └── operations/
│   │   ├── db/                  # 向量数据库（持久化）
│   │   └── config.json          # 知识库配置
│   ├── erp_system/
│   │   ├── docs/
│   │   ├── db/
│   │   └── config.json
│   └── ops_manual/
│       ├── docs/
│       ├── db/
│       └── config.json
├── data_platform/               # 数据中台代码
│   ├── backend/                 # FastAPI 后端
│   ├── frontend/                # React/Vue 前端
│   └── config/                  # 全局配置
└── logs/                        # 日志目录
```

### 知识库配置文件示例

**文件路径**：`knowledge_bases/crm_system/config.json`

```json
{
  "kb_id": "crm_system",
  "kb_name": "CRM系统知识库",
  "description": "包含CRM系统的需求、设计、运维文档",
  "created_at": "2025-10-30",
  "embedding_model": "text-embedding-3-small",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "db_type": "chromadb",
  "db_path": "./knowledge_bases/crm_system/db",
  "allowed_doc_types": [".docx", ".pdf", ".md", ".txt"],
  "auto_extract_metadata": true,
  "metadata_fields": ["system", "doc_type", "version", "module"]
}
```

### 多知识库管理 API

```python
# 创建知识库
POST /api/kb/create
{
  "kb_id": "erp_system",
  "kb_name": "ERP系统知识库",
  "embedding_model": "text-embedding-3-small"
}

# 列出所有知识库
GET /api/kb/list
# 返回：
[
  {"kb_id": "crm_system", "kb_name": "CRM系统知识库", "doc_count": 15},
  {"kb_id": "erp_system", "kb_name": "ERP系统知识库", "doc_count": 8}
]

# 删除知识库（谨慎操作）
DELETE /api/kb/{kb_id}
```

------

## 💾 向量数据库持久化方案

### ChromaDB 持久化配置（推荐小规模使用）

```python
import chromadb
from chromadb.config import Settings

# ❌ 错误：纯内存模式（重启丢失）
client = chromadb.Client()

# ✅ 正确：持久化到磁盘
client = chromadb.PersistentClient(
    path="./knowledge_bases/crm_system/db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# 创建或获取 collection
collection = client.get_or_create_collection(
    name="crm_documents",
    metadata={"kb_id": "crm_system"}
)
```

### 数据库选型建议

| 数据规模      | 推荐方案 | 部署方式     | 特点             |
| ------------- | -------- | ------------ | ---------------- |
| < 10 万条     | ChromaDB | 单机文件存储 | 轻量级，零配置   |
| 10 ~ 100 万条 | Qdrant   | Docker 容器  | 高性能，支持过滤 |
| > 100 万条    | Milvus   | K8s 集群     | 分布式，企业级   |

### 数据库维护策略

#### 1. 定期备份

```bash
#!/bin/bash
# backup_dbs.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/chroma_$DATE"

# 备份所有知识库
mkdir -p $BACKUP_DIR
cp -r ./knowledge_bases/*/db $BACKUP_DIR/

# 压缩
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

#### 2. 版本清理脚本

```python
def clean_old_versions(kb_id: str, doc_name: str, keep_latest: int = 3):
    """
    清理同一文档的旧版本，保留最新的 N 个版本
    """
    collection = get_collection(kb_id)
    
    # 查询该文档的所有版本
    results = collection.get(
        where={"doc_name": doc_name},
        include=["metadatas"]
    )
    
    # 按版本号排序
    versions = sorted(results['metadatas'], 
                     key=lambda x: x.get('version', '0.0'), 
                     reverse=True)
    
    # 删除旧版本
    old_versions = versions[keep_latest:]
    for v in old_versions:
        collection.delete(where={"version": v['version'], "doc_name": doc_name})
    
    print(f"Cleaned {len(old_versions)} old versions of {doc_name}")
```

#### 3. 数据迁移

```python
# 迁移到新环境
def migrate_kb(source_path: str, target_path: str, kb_id: str):
    """
    迁移知识库到新环境
    """
    import shutil
    
    source_db = f"{source_path}/{kb_id}/db"
    target_db = f"{target_path}/{kb_id}/db"
    
    # 复制数据库文件
    shutil.copytree(source_db, target_db)
    
    # 验证数据完整性
    client = chromadb.PersistentClient(path=target_db)
    collection = client.get_or_create_collection(f"{kb_id}_documents")
    count = collection.count()
    
    print(f"Migration completed. Total documents: {count}")
```

#### 4. 监控与告警

```python
import os

def monitor_db_size(kb_id: str, threshold_gb: float = 10.0):
    """
    监控数据库磁盘占用
    """
    db_path = f"./knowledge_bases/{kb_id}/db"
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(db_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    size_gb = total_size / (1024**3)
    
    if size_gb > threshold_gb:
        print(f"⚠️ Warning: {kb_id} database size {size_gb:.2f}GB exceeds threshold!")
        # 发送告警通知...
    
    return size_gb
```

------

## 🏛️ 数据中台架构设计

### 后端 API 设计（FastAPI）

#### 完整 API 列表

```python
# ==================== 知识库管理 ====================
POST   /api/kb/create                    # 创建知识库
GET    /api/kb/list                      # 获取知识库列表
GET    /api/kb/{kb_id}/info              # 获取知识库详情
PUT    /api/kb/{kb_id}/update            # 更新知识库配置
DELETE /api/kb/{kb_id}                   # 删除知识库

# ==================== 文档管理 ====================
POST   /api/kb/{kb_id}/documents/upload  # 上传文档（支持批量）
GET    /api/kb/{kb_id}/documents         # 获取文档列表
GET    /api/kb/{kb_id}/documents/{doc_id} # 获取文档详情
DELETE /api/kb/{kb_id}/documents/{doc_id} # 删除文档
PUT    /api/kb/{kb_id}/documents/{doc_id}/reprocess # 重新处理文档

# ==================== 检索接口（供 Dify 调用） ====================
POST   /api/retrieve                     # 语义检索
POST   /api/retrieve/multi               # 多知识库联合检索

# ==================== 监控与维护 ====================
GET    /api/system/stats                 # 系统统计信息
GET    /api/kb/{kb_id}/stats             # 知识库统计
POST   /api/maintenance/backup           # 触发备份
POST   /api/maintenance/cleanup          # 清理旧版本
```

#### 核心接口实现示例

**1. 文档上传接口**

```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from typing import List
import uuid

app = FastAPI()

@app.post("/api/kb/{kb_id}/documents/upload")
async def upload_documents(
    kb_id: str,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    上传文档到指定知识库
    支持批量上传，异步处理
    """
    results = []
    
    for file in files:
        # 生成文档 ID
        doc_id = str(uuid.uuid4())
        
        # 保存文件
        file_path = f"./knowledge_bases/{kb_id}/docs/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 添加到后台任务队列
        background_tasks.add_task(
            process_document,
            kb_id=kb_id,
            doc_id=doc_id,
            file_path=file_path
        )
        
        results.append({
            "doc_id": doc_id,
            "filename": file.filename,
            "status": "processing"
        })
    
    return {"success": True, "documents": results}
```

**2. 检索接口（供 Dify 调用）**

```python
from pydantic import BaseModel
from typing import Optional, Dict, List

class RetrieveRequest(BaseModel):
    query: str
    kb_id: str
    top_k: int = 5
    filters: Optional[Dict] = None
    min_score: float = 0.6

@app.post("/api/retrieve")
async def retrieve(request: RetrieveRequest):
    """
    语义检索接口
    Dify 通过此接口获取相关知识
    """
    # 加载知识库
    collection = get_collection(request.kb_id)
    
    # 向量化查询
    query_embedding = get_embedding(request.query)
    
    # 检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.top_k,
        where=request.filters  # 按 metadata 过滤
    )
    
    # 格式化返回
    chunks = []
    for i, doc in enumerate(results['documents'][0]):
        score = results['distances'][0][i]
        
        if score >= request.min_score:
            chunks.append({
                "content": doc,
                "metadata": results['metadatas'][0][i],
                "score": score
            })
    
    return {
        "success": True,
        "query": request.query,
        "kb_id": request.kb_id,
        "chunks": chunks,
        "total": len(chunks)
    }
```

**3. 文档处理逻辑（LangChain 层）**

```python
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import json

def process_document(kb_id: str, doc_id: str, file_path: str):
    """
    处理上传的文档
    1. 解析文档
    2. 分块
    3. 向量化
    4. 存入数据库
    """
    try:
        # 1. 读取配置
        with open(f"./knowledge_bases/{kb_id}/config.json") as f:
            config = json.load(f)
        
        # 2. 加载文档
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        
        # 3. 提取元信息（从文件名或内容）
        metadata = extract_metadata(file_path, documents[0].page_content)
        
        # 4. 分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            separators=["\n\n", "\n", "。", "!", "?", "；", "……", "…", " "]
        )
        chunks = text_splitter.split_documents(documents)
        
        # 5. 向量化
        embeddings = OpenAIEmbeddings(model=config['embedding_model'])
        
        # 6. 存入数据库
        collection = get_collection(kb_id)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "doc_id": doc_id,
                "chunk_index": i,
                "source": file_path
            }
            
            embedding = embeddings.embed_query(chunk.page_content)
            
            collection.add(
                embeddings=[embedding],
                documents=[chunk.page_content],
                metadatas=[chunk_metadata],
                ids=[f"{doc_id}_chunk_{i}"]
            )
        
        # 7. 更新文档状态
        update_document_status(kb_id, doc_id, "completed", len(chunks))
        
        print(f"✅ Document {doc_id} processed successfully: {len(chunks)} chunks")
        
    except Exception as e:
        update_document_status(kb_id, doc_id, "failed", 0, str(e))
        print(f"❌ Error processing document {doc_id}: {e}")

def extract_metadata(file_path: str, content: str) -> dict:
    """
    从文件名和内容中提取元信息
    """
    import re
    from pathlib import Path
    
    filename = Path(file_path).stem
    
    metadata = {
        "filename": filename,
        "doc_type": "unknown",
        "version": "1.0",
        "system": "unknown"
    }
    
    # 从文件名提取（例如：CRM_设计文档_v3.0.docx）
    match = re.search(r'([A-Z]+)_(.+?)_v([\d.]+)', filename)
    if match:
        metadata['system'] = match.group(1)
        metadata['doc_type'] = match.group(2)
        metadata['version'] = match.group(3)
    
    # 从内容提取（查找"系统："、"版本："等关键词）
    if '需求文档' in content[:500]:
        metadata['doc_type'] = '需求文档'
    elif '设计文档' in content[:500]:
        metadata['doc_type'] = '设计文档'
    elif '运维' in content[:500]:
        metadata['doc_type'] = '运维文档'
    
    return metadata
```

### 前端管理界面设计

#### 技术栈推荐

```
前端框架：React 18 + TypeScript
UI 组件库：Ant Design 5.x
状态管理：Zustand / Redux Toolkit
HTTP 客户端：Axios
图表展示：ECharts / Recharts
```

#### 核心功能模块

**1. 知识库管理页面**

```
功能：
✅ 创建/删除知识库
✅ 切换当前知识库
✅ 查看知识库统计（文档数、chunk 数、总大小）
✅ 配置知识库参数（chunk_size、embedding_model）
```

**2. 文档管理页面**

```
功能：
✅ 拖拽上传文档（支持批量）
✅ 文档列表展示（文件名、大小、状态、上传时间）
✅ 文档状态：processing（处理中）、completed（已完成）、failed（失败）
✅ 查看文档详情（chunk 列表、metadata）
✅ 重新处理失败的文档
✅ 删除文档
```

**3. 检索测试页面**

```
功能：
✅ 输入测试问题
✅ 选择知识库
✅ 设置检索参数（top_k、过滤条件）
✅ 查看召回的 chunks
✅ 显示相似度分数
✅ 高亮匹配关键词
```

**4. 系统监控页面**

```
功能：
✅ 数据库磁盘占用
✅ API 调用统计
✅ 文档处理速度
✅ 错误日志查看
✅ 触发手动备份
```

------

## 🚀 完整实施步骤

### 第一阶段：环境准备

```bash
# 1. 创建项目目录
mkdir knowledge_platform && cd knowledge_platform

# 2. 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install fastapi uvicorn langchain langchain-community \
            openai chromadb unstructured watchdog \
            python-multipart pydantic pillow

# 4. 创建目录结构
mkdir -p knowledge_bases data_platform/{backend,frontend} logs
```

### 第二阶段：搭建数据中台后端

**项目结构**：

```
data_platform/backend/
├── main.py                    # FastAPI 主入口
├── api/
│   ├── kb_management.py       # 知识库管理 API
│   ├── document_management.py # 文档管理 API
│   ├── retrieve.py            # 检索 API
│   └── monitoring.py          # 监控 API
├── services/
│   ├── langchain_processor.py # LangChain 处理逻辑
│   ├── db_manager.py          # 数据库管理
│   └── metadata_extractor.py # 元信息提取
├── models/
│   └── schemas.py             # Pydantic 数据模型
└── config/
    └── settings.py            # 配置文件
```

**启动后端**：

```bash
cd data_platform/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 第三阶段：对接 Dify

**在 Dify 中创建自定义工具**：

```yaml
工具名称: knowledge_platform_retrieve
描述: 从知识中台检索相关内容
API 配置:
  - Method: POST
  - URL: http://your-server:8000/api/retrieve
  - Headers:
      Content-Type: application/json
  - Body:
      {
        "query": "{{query}}",
        "kb_id": "{{kb_id}}",
        "top_k": 5
      }
输出变量:
  - chunks (array): 检索到的知识片段
  - total (number): 结果数量
```

**在 Dify Workflow 中使用**：

```
1. 接收用户问题
2. 调用 knowledge_platform_retrieve 工具
   - query: {{用户输入}}
   - kb_id: "crm_system"
3. 提取 chunks[].content 合并为上下文
4. 调用 LLM 节点
   - System Prompt: "你是一个技术助手，请根据以下知识回答问题"
   - Context: {{检索结果}}
   - User Query: {{用户输入}}
5. 返回答案
```

### 第四阶段：开发前端管理界面（可选）

```bash
# 使用 Vite + React 创建项目
cd data_platform/frontend
npm create vite@latest . -- --template react-ts
npm install antd axios zustand recharts

# 启动开发服务器
npm run dev
```

### 第五阶段：部署上线

**Docker 部署（推荐）**：

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "data_platform.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
# docker-compose.yml
version: '3.8'

services:
  knowledge-platform:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./knowledge_bases:/app/knowledge_bases
      - ./logs:/app/logs
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

------

## 🎯 进阶功能扩展

### 1. 自动化文档监听

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

class DocumentWatcher(FileSystemEventHandler):
    """
    监听文档文件夹，自动触发处理
    """
    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        self.watch_path = f"./knowledge_bases/{kb_id}/docs"
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        # 检查文件类型
        allowed_extensions = ['.txt', '.md', '.docx', '.pdf', '.html']
        if not any(event.src_path.endswith(ext) for ext in allowed_extensions):
            return
        
        print(f"📄 New document detected: {event.src_path}")
        
        # 等待文件写入完成
        time.sleep(2)
        
        # 触发处理
        doc_id = str(uuid.uuid4())
        process_document(self.kb_id, doc_id, event.src_path)
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        print(f"📝 Document modified: {event.src_path}")
        # 可以选择重新处理文档

def start_watcher(kb_id: str):
    """
    启动文件监听器
    """
    event_handler = DocumentWatcher(kb_id)
    observer = Observer()
    observer.schedule(
        event_handler, 
        f"./knowledge_bases/{kb_id}/docs", 
        recursive=True
    )
    observer.start()
    
    print(f"👀 Watching knowledge base: {kb_id}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 使用示例
if __name__ == "__main__":
    start_watcher("crm_system")
```

### 2. 版本管理与对比

```python
def compare_versions(kb_id: str, doc_name: str, version1: str, version2: str):
    """
    对比同一文档的两个版本差异
    """
    collection = get_collection(kb_id)
    
    # 获取两个版本的内容
    v1_chunks = collection.get(
        where={"doc_name": doc_name, "version": version1}
    )
    v2_chunks = collection.get(
        where={"doc_name": doc_name, "version": version2}
    )
    
    # 使用 difflib 对比
    import difflib
    
    v1_text = "\n".join(v1_chunks['documents'])
    v2_text = "\n".join(v2_chunks['documents'])
    
    diff = difflib.unified_diff(
        v1_text.splitlines(),
        v2_text.splitlines(),
        lineterm='',
        fromfile=f'{doc_name} v{version1}',
        tofile=f'{doc_name} v{version2}'
    )
    
    return list(diff)

def get_version_history(kb_id: str, doc_name: str):
    """
    获取文档的版本历史
    """
    collection = get_collection(kb_id)
    
    results = collection.get(
        where={"doc_name": doc_name},
        include=["metadatas"]
    )
    
    versions = {}
    for metadata in results['metadatas']:
        version = metadata.get('version', '1.0')
        if version not in versions:
            versions[version] = {
                "version": version,
                "created_at": metadata.get('created_at'),
                "author": metadata.get('author'),
                "chunk_count": 0
            }
        versions[version]['chunk_count'] += 1
    
    return sorted(versions.values(), key=lambda x: x['version'], reverse=True)
```

### 3. 智能元信息提取

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

def extract_metadata_with_llm(content: str) -> dict:
    """
    使用 LLM 智能提取文档元信息
    """
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""
        请从以下文档内容中提取关键元信息，以 JSON 格式返回：
        
        文档内容：
        {content}
        
        需要提取的信息：
        - system: 系统名称（如 CRM、ERP）
        - doc_type: 文档类型（需求文档/设计文档/运维文档/API文档）
        - version: 版本号
        - module: 主要涉及的模块
        - keywords: 5个关键词
        
        只返回 JSON，不要其他内容：
        """
    )
    
    llm = OpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    result = chain.run(content=content[:3000])  # 只取前3000字
    
    try:
        metadata = json.loads(result)
        return metadata
    except:
        return {
            "system": "unknown",
            "doc_type": "unknown",
            "version": "1.0",
            "module": "general",
            "keywords": []
        }
```

### 4. 多知识库联合检索

```python
@app.post("/api/retrieve/multi")
async def multi_kb_retrieve(
    query: str,
    kb_ids: List[str],
    top_k: int = 5,
    merge_strategy: str = "score"  # score / round_robin / kb_priority
):
    """
    跨多个知识库检索
    适用于：需要综合多个系统的知识回答问题
    """
    all_results = []
    
    # 从每个知识库检索
    for kb_id in kb_ids:
        collection = get_collection(kb_id)
        query_embedding = get_embedding(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # 添加来源标识
        for i, doc in enumerate(results['documents'][0]):
            all_results.append({
                "content": doc,
                "metadata": {
                    **results['metadatas'][0][i],
                    "source_kb": kb_id
                },
                "score": results['distances'][0][i]
            })
    
    # 合并策略
    if merge_strategy == "score":
        # 按相似度排序
        all_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = all_results[:top_k]
    
    elif merge_strategy == "round_robin":
        # 轮流取每个知识库的结果
        final_results = []
        kb_indices = {kb_id: 0 for kb_id in kb_ids}
        
        while len(final_results) < top_k:
            for kb_id in kb_ids:
                kb_results = [r for r in all_results if r['metadata']['source_kb'] == kb_id]
                idx = kb_indices[kb_id]
                if idx < len(kb_results):
                    final_results.append(kb_results[idx])
                    kb_indices[kb_id] += 1
    
    elif merge_strategy == "kb_priority":
        # 按知识库优先级顺序
        final_results = []
        for kb_id in kb_ids:
            kb_results = [r for r in all_results if r['metadata']['source_kb'] == kb_id]
            final_results.extend(kb_results[:top_k])
            if len(final_results) >= top_k:
                break
        final_results = final_results[:top_k]
    
    return {
        "success": True,
        "query": query,
        "kb_ids": kb_ids,
        "chunks": final_results,
        "total": len(final_results)
    }
```

### 5. 检索结果重排序（Rerank）

```python
from sentence_transformers import CrossEncoder

class RerankerService:
    """
    使用交叉编码器对检索结果重排序
    提升检索准确性
    """
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
        """
        对初步检索结果重排序
        """
        # 准备输入对
        pairs = [[query, chunk['content']] for chunk in chunks]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 添加重排序分数
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(scores[i])
        
        # 按重排序分数排序
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]

# 在检索接口中使用
reranker = RerankerService()

@app.post("/api/retrieve/rerank")
async def retrieve_with_rerank(request: RetrieveRequest):
    """
    带重排序的检索
    """
    # 初步检索（取 top_k * 2）
    initial_results = await retrieve(
        RetrieveRequest(
            query=request.query,
            kb_id=request.kb_id,
            top_k=request.top_k * 2
        )
    )
    
    # 重排序
    reranked = reranker.rerank(
        request.query,
        initial_results['chunks'],
        request.top_k
    )
    
    return {
        "success": True,
        "query": request.query,
        "chunks": reranked,
        "total": len(reranked)
    }
```

### 6. 增量更新与差异检测

```python
def incremental_update(kb_id: str, doc_id: str, new_file_path: str):
    """
    增量更新文档
    只处理变化的部分
    """
    collection = get_collection(kb_id)
    
    # 获取旧文档的 chunks
    old_chunks = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"]
    )
    
    # 解析新文档
    loader = UnstructuredFileLoader(new_file_path)
    new_documents = loader.load()
    
    # 分块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_chunks = text_splitter.split_documents(new_documents)
    
    # 对比差异
    old_texts = set(old_chunks['documents'])
    new_texts = [chunk.page_content for chunk in new_chunks]
    
    # 找出需要添加的新 chunks
    to_add = [text for text in new_texts if text not in old_texts]
    
    # 找出需要删除的旧 chunks
    to_delete = [text for text in old_texts if text not in new_texts]
    
    # 执行删除
    for text in to_delete:
        collection.delete(where={"doc_id": doc_id, "content": text})
    
    # 执行添加
    embeddings = OpenAIEmbeddings()
    for i, text in enumerate(to_add):
        embedding = embeddings.embed_query(text)
        collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "doc_id": doc_id,
                "chunk_index": len(old_chunks['documents']) + i,
                "updated_at": datetime.now().isoformat()
            }],
            ids=[f"{doc_id}_chunk_{len(old_chunks['documents']) + i}"]
        )
    
    print(f"✅ Incremental update: +{len(to_add)} chunks, -{len(to_delete)} chunks")
```

### 7. 知识图谱构建（高级）

```python
from langchain.chains import GraphQAChain
import networkx as nx

class KnowledgeGraphBuilder:
    """
    从文档中提取实体和关系，构建知识图谱
    """
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def extract_entities_and_relations(self, text: str):
        """
        使用 NER 和关系抽取提取知识三元组
        """
        # 使用 spaCy 或 LLM 提取实体
        prompt = f"""
        从以下文本中提取知识三元组（主体-关系-客体）：
        
        文本：{text}
        
        返回格式：
        主体|关系|客体
        主体|关系|客体
        ...
        """
        
        # 这里可以接入 GPT-4 或本地 NER 模型
        # 示例返回
        return [
            ("CRM系统", "包含", "登录模块"),
            ("登录模块", "支持", "账号密码认证"),
            ("登录模块", "支持", "手机验证码"),
        ]
    
    def build_graph(self, kb_id: str):
        """
        为整个知识库构建图谱
        """
        collection = get_collection(kb_id)
        all_docs = collection.get()
        
        for doc in all_docs['documents']:
            triples = self.extract_entities_and_relations(doc)
            
            for subject, relation, obj in triples:
                self.graph.add_edge(subject, obj, relation=relation)
        
        return self.graph
    
    def query_graph(self, query: str):
        """
        基于图谱的查询
        """
        # 使用 Cypher 或 GraphQL 查询
        # 示例：找到 CRM 系统的所有模块
        # MATCH (s:System {name: "CRM系统"})-[:包含]->(m:Module)
        # RETURN m.name
        pass
```

### 8. 自动化测试与质量评估

```python
class KnowledgeQualityEvaluator:
    """
    评估知识库质量
    """
    def evaluate_coverage(self, kb_id: str, test_questions: List[str]) -> dict:
        """
        测试知识库覆盖率
        """
        results = {
            "total_questions": len(test_questions),
            "answered": 0,
            "no_answer": 0,
            "low_confidence": 0
        }
        
        for question in test_questions:
            response = retrieve(RetrieveRequest(
                query=question,
                kb_id=kb_id,
                top_k=3
            ))
            
            if len(response['chunks']) == 0:
                results['no_answer'] += 1
            elif response['chunks'][0]['score'] < 0.7:
                results['low_confidence'] += 1
            else:
                results['answered'] += 1
        
        results['coverage_rate'] = results['answered'] / results['total_questions']
        
        return results
    
    def evaluate_chunk_quality(self, kb_id: str) -> dict:
        """
        评估 chunk 质量
        """
        collection = get_collection(kb_id)
        all_chunks = collection.get()
        
        stats = {
            "total_chunks": len(all_chunks['documents']),
            "avg_length": 0,
            "too_short": 0,  # < 100 字
            "too_long": 0,   # > 2000 字
            "optimal": 0     # 500-1500 字
        }
        
        lengths = []
        for doc in all_chunks['documents']:
            length = len(doc)
            lengths.append(length)
            
            if length < 100:
                stats['too_short'] += 1
            elif length > 2000:
                stats['too_long'] += 1
            else:
                stats['optimal'] += 1
        
        stats['avg_length'] = sum(lengths) / len(lengths)
        stats['optimal_rate'] = stats['optimal'] / stats['total_chunks']
        
        return stats
```

------

## 📊 系统监控与告警

### 监控指标设计

```python
from datetime import datetime, timedelta
import psutil

class SystemMonitor:
    """
    系统监控服务
    """
    def get_system_stats(self):
        """
        获取系统资源使用情况
        """
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_kb_stats(self, kb_id: str):
        """
        获取知识库统计信息
        """
        collection = get_collection(kb_id)
        
        # 数据库大小
        db_path = f"./knowledge_bases/{kb_id}/db"
        db_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(db_path)
            for filename in filenames
        ) / (1024**3)  # GB
        
        # chunk 统计
        total_chunks = collection.count()
        
        # 文档统计
        all_metadata = collection.get(include=["metadatas"])
        unique_docs = len(set(m['doc_id'] for m in all_metadata['metadatas']))
        
        return {
            "kb_id": kb_id,
            "total_chunks": total_chunks,
            "total_documents": unique_docs,
            "db_size_gb": round(db_size, 2),
            "avg_chunks_per_doc": round(total_chunks / unique_docs, 1) if unique_docs > 0 else 0
        }
    
    def get_api_stats(self, time_range: int = 24):
        """
        获取 API 调用统计（最近 N 小时）
        """
        # 这里需要从日志或数据库读取
        # 示例数据
        return {
            "total_calls": 1523,
            "retrieve_calls": 1200,
            "upload_calls": 150,
            "avg_response_time_ms": 245,
            "error_rate": 0.02
        }

# 在 FastAPI 中使用
monitor = SystemMonitor()

@app.get("/api/system/stats")
async def get_system_stats():
    return monitor.get_system_stats()

@app.get("/api/kb/{kb_id}/stats")
async def get_kb_stats(kb_id: str):
    return monitor.get_kb_stats(kb_id)
```

### 告警配置

```python
class AlertService:
    """
    告警服务
    """
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def check_and_alert(self):
        """
        检查各项指标并发送告警
        """
        alerts = []
        
        # 检查磁盘空间
        disk_usage = psutil.disk_usage('/').percent
        if disk_usage > 85:
            alerts.append({
                "level": "critical",
                "message": f"磁盘使用率过高: {disk_usage}%"
            })
        
        # 检查数据库大小
        for kb_id in list_all_kb_ids():
            size_gb = monitor_db_size(kb_id)
            if size_gb > 50:
                alerts.append({
                    "level": "warning",
                    "message": f"知识库 {kb_id} 大小超过 50GB: {size_gb}GB"
                })
        
        # 发送告警
        if alerts:
            self.send_alerts(alerts)
    
    def send_alerts(self, alerts: List[dict]):
        """
        发送告警到企业微信/钉钉/邮件
        """
        import requests
        
        for alert in alerts:
            requests.post(self.webhook_url, json={
                "msgtype": "text",
                "text": {
                    "content": f"[{alert['level'].upper()}] {alert['message']}"
                }
            })
```

------

## 🔒 安全与权限控制

### API 认证

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    验证 JWT Token
    """
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# 在 API 中使用
@app.post("/api/kb/{kb_id}/documents/upload")
async def upload_documents(
    kb_id: str,
    files: List[UploadFile],
    user: dict = Depends(verify_token)
):
    # 检查用户权限
    if not has_permission(user['user_id'], kb_id, 'write'):
        raise HTTPException(status_code=403, detail="No permission")
    
    # 处理上传...
```

### 知识库权限管理

```python
class PermissionManager:
    """
    权限管理
    """
    def __init__(self):
        self.permissions = {}  # {user_id: {kb_id: [permissions]}}
    
    def grant_permission(self, user_id: str, kb_id: str, permission: str):
        """
        授予权限: read / write / admin
        """
        if user_id not in self.permissions:
            self.permissions[user_id] = {}
        
        if kb_id not in self.permissions[user_id]:
            self.permissions[user_id][kb_id] = []
        
        if permission not in self.permissions[user_id][kb_id]:
            self.permissions[user_id][kb_id].append(permission)
    
    def has_permission(self, user_id: str, kb_id: str, permission: str) -> bool:
        """
        检查权限
        """
        if user_id not in self.permissions:
            return False
        
        if kb_id not in self.permissions[user_id]:
            return False
        
        return permission in self.permissions[user_id][kb_id] or \
               'admin' in self.permissions[user_id][kb_id]
```

------

## 🎓 最佳实践建议

### 文档准备建议

1. **统一命名规范**

   ```
   格式：{系统名}_{文档类型}_{模块名}_v{版本号}.{扩展名}
   示例：CRM_设计文档_登录模块_v3.0.docx
   ```

2. **章节结构模板**

   ```markdown
   # {系统名} - {文档类型}
   
   **版本**: v3.0
   **作者**: 张三
   **日期**: 2025-10-30
   
   ## 1. 概述
   （系统背景、目标）
   
   ## 2. 功能模块
   ### 2.1 模块A
   #### 2.1.1 功能说明
   #### 2.1.2 技术实现
   
   ### 2.2 模块B
   ...
   ```

3. **表格与图片处理**

   - 表格前后加文字说明
   - 图片配文字描述
   - 架构图用 Mermaid 或文字补充

### 检索优化策略

1. **合理设置 top_k**

   - 一般问题：top_k = 3-5
   - 复杂问题：top_k = 8-10
   - 需要全面了解：top_k = 15-20

2. **使用 metadata 过滤**

   ```python
   # 只检索设计文档
   filters = {"doc_type": "设计文档"}
   
   # 只检索最新版本
   filters = {"version": "v3.0"}
   
   # 组合条件
   filters = {
       "doc_type": "设计文档",
       "module": "登录",
       "version": "v3.0"
   }
   ```

3. **启用重排序**

   - 对于重要查询，使用 Rerank 提升准确性
   - 初检索取 top_k * 2，重排序后取 top_k

### 运维建议

1. **定期备份**

   ```bash
   # 每天凌晨 2 点执行
   0 2 * * * /path/to/backup_dbs.sh
   ```

2. **监控关键指标**

   - 磁盘空间 > 80% 告警
   - API 响应时间 > 1s 告警
   - 错误率 > 5% 告警

3. **版本清理策略**

   - 保留最近 3 个版本
   - 每月清理一次旧版本
   - 重要版本手动标记保留

------

## 📚 附录

### 常见问题FAQ

**Q1: 如何切换 embedding 模型？**

A: 修改知识库的 `config.json`：

```json
{
  "embedding_model": "text-embedding-3-large"  // 或本地模型
}
```

**Q2: 如何处理超大文档（> 100MB）？**

A:

- 方案1：手动拆分文档
- 方案2：增加 chunk_size，减少 chunk 数量
- 方案3：使用流式处理

**Q3: 检索效果不好怎么办？**

A:

1. 检查文档质量（是否有结构）
2. 调整 chunk_size 和 overlap
3. 启用 Rerank
4. 使用更好的 embedding 模型

**Q4: 如何支持多语言？**

A: 使用多语言 embedding 模型，如：

- `text-embedding-3-small` (OpenAI，支持多语言)
- `multilingual-e5-large` (开源)

------

## ✅ 总结

通过以上设计，你将拥有一个：

✅ **多知识库隔离** - 不同项目互不干扰 ✅ **数据持久化** - 向量数据安全存储 ✅ **解耦架构** - 数据中台作为中间层 ✅ **自动化处理** - 文档上传即入库 ✅ **灵活检索** - 支持过滤、重排序、多库联合 ✅ **可视化管理** - 前端界面监控一切 ✅ **企业级功能** - 版本管理、权限控制、监控告警

这是一个完整的企业级知识中台解决方案！🎉1