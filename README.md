# 🧠 企业级知识中台 - LangChain + Dify 解决方案

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于 LangChain 和 Dify 的企业级知识中台解决方案，支持多知识库管理、智能文档处理和语义检索。

## 🌟 核心特性

### 🏗️ 企业级架构
- **解耦设计**：数据中台作为中间层，Dify 和 LangChain 通过标准 API 通信
- **知识库隔离**：每个项目/系统独立知识库，互不干扰
- **数据持久化**：向量数据存储在磁盘，支持备份与迁移
- **可扩展性**：支持从单机到分布式的平滑升级

### 📄 智能文档处理
- **多格式支持**：支持 .txt、.md、.docx、.pdf、.html、.pptx 等主流格式
- **智能分块**：基于 RecursiveCharacterTextSplitter 的语义分块
- **元数据提取**：自动提取文档版本、系统、类型等元信息
- **批量处理**：支持文档批量上传和异步处理

### 🔍 高效检索系统
- **语义检索**：基于 OpenAI Embeddings 的向量相似度匹配
- **多知识库检索**：支持跨知识库联合检索
- **过滤查询**：基于元数据的精确过滤
- **相似度评分**：提供检索结果的相关性评分

### 💾 灵活存储方案
- **小规模**：ChromaDB (< 10万条) - 轻量级，零配置
- **中等规模**：Qdrant (10-100万条) - 高性能，支持过滤
- **大规模**：Milvus (> 100万条) - 分布式，企业级

## 🏛️ 系统架构

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

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 16+ (前端开发)
- OpenAI API Key (或本地 Embedding 模型)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-org/knowledge-hub.git
cd knowledge-hub
```

2. **创建虚拟环境**

3. **安装后端依赖**

4. **安装前端依赖**

5. **配置环境变量**

6. **部署 MinIO 对象存储**

**使用 Docker 部署 MinIO (推荐)：**
```bash
# 创建 MinIO 数据目录
mkdir -p ./data/minio/{data,config}

# 启动 MinIO 服务
docker run -d \
  --name minio-server \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin123" \
  -v ./data/minio/data:/data \
  -v ./data/minio/config:/root/.minio \
  minio/minio server /data --console-address ":9001"

# 验证 MinIO 服务
curl http://localhost:9000/minio/health/live
```

**使用 Docker Compose 部署：**
```bash
# 创建 docker-compose.minio.yml 文件（参考实施方案文档）
docker-compose -f docker-compose.minio.yml up -d
```

**访问 MinIO：**
- API 端点：http://localhost:9000
- Web 控制台：http://localhost:9001
- 默认账号：minioadmin / minioadmin123

7. **启动服务**

后端服务：

前端服务：

8. **访问系统**
- 管理界面：http://localhost:3000
- API 文档：http://localhost:8000/docs
- MinIO 控制台：http://localhost:9001

## 📁 项目结构

```

```

## 📚 相关文档

本项目提供了完整的技术文档和实施指南，帮助您深入了解系统架构和实施细节：

### 📖 核心文档

| 文档名称 | 描述 | 链接 |
|---------|------|------|
| 🏗️ **架构设计文档** | 详细的系统架构设计、技术选型和核心设计原则 | [LangChain + Dify 企业级知识中台架构设计文档](./docs/LangChain%20+%20Dify%20企业级知识中台架构设计文档.md) |
| 🚀 **实施方案与验收标准** | 完整的项目实施计划、任务分解和验收标准 | [企业级知识中台实施方案与验收标准](./docs/企业级知识中台实施方案与验收标准.md) |
| 📋 **完整实施指南** | 从零开始的详细实施步骤和代码示例 | [企业级知识中台完整实施指南](./docs/企业级知识中台完整实施指南.md) |

### 📝 文档说明

- **架构设计文档**：包含系统整体架构图、技术栈选择、数据流设计、安全策略等核心设计内容
- **实施方案与验收标准**：提供8个阶段的详细实施计划，包含任务清单、代码示例和验收标准
- **完整实施指南**：提供从环境搭建到部署上线的完整操作步骤，适合开发团队参考

### 🎯 使用建议

1. **项目启动前**：先阅读架构设计文档，了解整体技术方案
2. **开发阶段**：参考实施方案与验收标准，按阶段推进项目
3. **具体实施**：使用完整实施指南中的代码示例和操作步骤

这些文档将帮助您：
- 🎯 快速理解项目架构和技术选型
- 📋 制定详细的项目实施计划
- 💻 获得完整的代码示例和最佳实践
- ✅ 建立明确的验收标准和质量控制

## 🔧 核心 API

### 知识库管理
```http
POST   /api/kb/create                    # 创建知识库
GET    /api/kb/list                      # 获取知识库列表
GET    /api/kb/{kb_id}/info              # 获取知识库详情
PUT    /api/kb/{kb_id}/update            # 更新知识库配置
DELETE /api/kb/{kb_id}                   # 删除知识库
```

### 文档管理
```http
POST   /api/kb/{kb_id}/documents/upload  # 上传文档（支持批量）
GET    /api/kb/{kb_id}/documents         # 获取文档列表
GET    /api/kb/{kb_id}/documents/{doc_id} # 获取文档详情
DELETE /api/kb/{kb_id}/documents/{doc_id} # 删除文档
```

### 文件存储管理（MinIO）
```http
POST   /api/files/upload                 # 上传文件到 MinIO
GET    /api/files/{file_id}/download      # 下载文件
GET    /api/files/{file_id}/url           # 获取文件预签名 URL
DELETE /api/files/{file_id}               # 删除文件
GET    /api/files/list                    # 获取文件列表
```

### 检索接口（供 Dify 调用）
```http
POST   /api/retrieve                     # 语义检索
POST   /api/retrieve/multi               # 多知识库联合检索
```

### 使用示例

**创建知识库：**

```bash
curl -X POST "http://localhost:8000/api/kb/create" \
     -H "Content-Type: application/json" \
     -d '{
       "kb_id": "crm_system",
       "kb_name": "CRM系统知识库",
       "embedding_model": "text-embedding-3-small"
     }'
```

**语义检索：**
```bash
curl -X POST "http://localhost:8000/api/retrieve" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "如何配置用户权限",
       "kb_id": "crm_system",
       "top_k": 5,
       "min_score": 0.6
     }'
```

**文件上传到 MinIO：**
```bash
curl -X POST "http://localhost:8000/api/files/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/document.pdf" \
     -F "kb_id=crm_system"
```

**获取文件预签名 URL：**
```bash
curl -X GET "http://localhost:8000/api/files/file123/url?expires=3600"
```

## 📋 支持的文档类型

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

## 🔧 配置说明

### 知识库配置示例

```json
{
  "kb_id": "crm_system",
  "kb_name": "CRM系统知识库",
  "description": "包含CRM系统的需求、设计、运维文档",
  "created_at": "2025-01-30",
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

### 环境变量配置

```bash
# OpenAI 配置
OPENAI_API_KEY=your_openai_api_key
EMBEDDING_MODEL=text-embedding-3-small

# 数据库配置
DB_TYPE=chromadb
DB_PATH=./knowledge_bases

# MinIO 对象存储配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET=knowledge-hub-docs
MINIO_SECURE=false

# 服务配置
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_URL=http://localhost:3000

# 文档处理配置
MAX_FILE_SIZE=50MB
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🤝 与 Dify 集成

### 在 Dify 中配置 API 工具

1. **添加 API 工具**
   - 工具名称：Knowledge Retrieval
   - API 端点：http://your-domain:8000/api/retrieve
   - 方法：POST

2. **配置参数**
```json
{
  "query": "{{query}}",
  "kb_id": "crm_system",
  "top_k": 5,
  "min_score": 0.6
}
```

3. **在工作流中使用**
   - 用户输入 → Knowledge Retrieval → LLM 生成答案

## 🧪 测试

### 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-asyncio httpx

# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_api.py -v
```

### API 测试示例

```python
import httpx

async def test_create_kb():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/kb/create",
            json={
                "kb_id": "test_kb",
                "kb_name": "测试知识库"
            }
        )
        assert response.status_code == 200
```

## 🆘 故障排除

### 常见问题

**Q: 检索结果不准确？**
A: 调整 chunk_size 和 embedding 模型，优化文档内容质量。

**Q: 数据库连接失败？**
A: 确认数据库路径权限，检查配置文件是否正确。

### 日志查看

```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
tail -f logs/error.log
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证

## 📞 联系我们

- 项目主页：https://github.com/guazixiong/knowledge-hub
- 问题反馈：https://github.com/guazixiong/knowledge-hub/issues
- 邮箱：pbad0606@163.com

## 🙏 致谢

感谢以下开源项目：
- [LangChain](https://langchain.com/) - 强大的 LLM 应用框架
- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的 Python Web 框架
- [ChromaDB](https://www.trychroma.com/) - 开源向量数据库
- [Unstructured](https://unstructured.io/) - 文档解析工具

---

⭐ 如果这个项目对您有帮助，请给我们一个 Star！
