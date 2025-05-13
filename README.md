# Codebase Indexing and Q&A Tool (OpenAI Powered)

此项目提供了一个Python工具，用于对源代码目录进行文本向量化索引，并基于索引内容通过大型语言模型（LLM）回答相关问题。

## 功能

1.  **代码库索引 (`index`)**: 
    *   扫描指定源代码目录下的文件。
    *   使用OpenAI的文本嵌入模型（如 `text-embedding-ada-002`）为文件内容创建向量嵌入。
    *   将文本内容、元数据（如文件路径）及其向量存储到Chroma向量数据库中。
    *   支持增量更新：如果文件内容未改变（通过MD5哈希比较），则跳过重新索引。
2.  **问答 (`ask`)**: 
    *   接收用户关于已索引代码库的问题。
    *   使用相同的嵌入模型将问题向量化。
    *   从Chroma数据库中检索与问题向量最相关的代码片段作为上下文。
    *   将问题和检索到的上下文信息一起发送给OpenAI的聊天模型（如 `gpt-3.5-turbo`）以生成答案。

## 目录结构

```
OpenAICodeVectorizer/
├── src/                  # 源代码目录
│   ├── __init__.py
│   ├── embedding_service.py # 处理文本嵌入和ChromaDB交互
│   ├── rag_service.py       # 处理RAG检索和LLM问答逻辑
│   └── utils.py             # 辅助函数 (如配置加载)
├── main.py               # 程序主入口，命令行接口
├── config.yaml           # 配置文件 (API密钥, 模型名称, DB路径等)
├── requirements.txt      # Python依赖包
└── README.md             # 本说明文件
```

## 安装与配置

1.  **克隆项目 (如果您是从git仓库获取)**
    ```bash
    # git clone <repository_url>
    # cd OpenAICodeVectorizer
    ```

2.  **创建并激活Python虚拟环境 (推荐)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    # source venv/bin/activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置API密钥和模型**
    打开 `config.yaml` 文件，进行如下配置：
    *   `openai_api_key`: 填入您从服务提供商（如OpenAI、火山引擎等）获取的API密钥。
    *   `embedding_api_type`: 指定使用的嵌入API类型，例如 `'openai'` 或 `'volcano'`。这将决定使用哪个API基准URL（`openai_api_base` 或 `volcano_engine_base_url`）。
    *   `volcano_engine_base_url`: (当 `embedding_api_type` 为 `'volcano'` 时使用) 火山引擎或兼容服务的API基础URL (例如: `https://ark.cn-beijing.volces.com/api/v3`)。
    *   `openai_api_base`: (可选, 当 `embedding_api_type` 为 `'openai'` 时使用) 标准OpenAI API的备用基础URL，例如用于代理或Azure OpenAI。
    *   `embedding_model`: 更改为您想使用的文本嵌入模型名称。
    *   `llm_model`: 更改为您想使用的聊天模型名称。
    *   `embedding_strategy`: (可选) 处理长文本的策略。默认为 `'chunking'`（将长文本分割成块进行嵌入）。可选值为 `'direct'`（直接嵌入整个文本，如果超过模型限制可能会失败）。后续可能支持更多策略。
    *   `max_tokens_per_chunk`: (可选, 仅当 `embedding_strategy` 为 `'chunking'` 时有效) 每个文本块的最大token数量。应设置为小于或等于嵌入模型支持的最大序列长度。默认 `4000`。
    *   `chunk_overlap_ratio`: (可选, 仅当 `embedding_strategy` 为 `'chunking'` 时有效) 相邻文本块之间的重叠比例（token数）。例如 `0.1` 表示10%的重叠。默认 `0.1`。
    *   `chroma_db_path`: (可选) Chroma向量数据库的持久化存储路径（相对路径或绝对路径）。默认为 `./chroma_db`。
    *   `collection_name`: (可选) ChromaDB中的集合名称。
    *   `source_file_extensions`: (可选) 指定在索引时应包含的文件扩展名列表。

## 使用方法

确保您的虚拟环境已激活，并且您位于 `OpenAICodeVectorizer` 项目的根目录下。

### 1. 索引代码库

使用 `index` 命令指定要索引的源代码目录的绝对路径。

```bash
python main.py index "/path/to/your/source/code/project"
```

例如，在Windows上:
```bash
python main.py index "D:\my_projects\cool_project_src"
```

索引过程会将向量数据存储在 `config.yaml` 中 `chroma_db_path` 指定的位置。

### 2. 就代码库提问

使用 `ask` 命令提出与已索引代码库相关的问题。

```bash
python main.py ask "你的问题是什么?"
```

例如:
```bash
python main.py ask "这个项目是如何处理用户认证的?"
```

您可以指定检索上下文时返回的相关文档数量 (默认为3):
```bash
python main.py ask "解释一下utils模块中的主要功能" --n_results 5
```

### 3. 可视化向量数据库

使用 `visualize` 命令将已索引的 Chroma 向量数据库以散点图方式可视化：

```bash
python main.py visualize
```

可选参数：
- `--chroma_db_path` 指定数据库路径（默认读取 config.yaml）
- `--collection_name` 指定集合名（默认读取 config.yaml）
- `--max_points` 控制最大可视化点数（默认2000）

例如：
```bash
python main.py visualize --chroma_db_path "./chroma_db" --collection_name "code_vectors" --max_points 1000
```

运行后会弹出matplotlib窗口，展示降维后的散点图，并标注部分文件名。

## 注意事项

*   首次索引大型代码库可能需要一些时间，具体取决于文件数量、大小以及网络连接。
*   确保您的API密钥有足够的配额用于嵌入和聊天模型调用，并且正确配置了第三方服务的`base_url`。
*   生成的向量数据库会持久化存储，下次对同一代码库提问时无需重新索引 (除非代码已更改并希望更新索引)。

# Version: 1.0
# Updated: 2025-05-12 14:51:59
# Version: 1.1
# Updated: 2025-05-12 16:50:46
# Version: 1.2
# Updated: 2025-05-13 11:00:00 