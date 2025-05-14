import os
import chromadb
from chromadb.utils import embedding_functions
from .utils import load_config
import hashlib
import pathspec
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 更新时间：2025-05-12 14:51:59
# Version: 1.0
# Updated: 2025-05-12 16:50:46
# Version: 1.1
# Updated: 2025-05-13 10:30:00

class EmbeddingService:
    def __init__(self):
        """初始化服务，加载配置，设置嵌入函数、ChromaDB客户端和文本分割器"""
        self.config = load_config()
        # --- Embedding Function Setup ---
        # 根据配置选择嵌入模型API类型
        api_type = self.config.get('embedding_api_type', 'volcano') # 默认为 'volcano'
        model_name = self.config['embedding_model']
        api_key = self.config['openai_api_key'] # 通用API Key字段

        if api_type == 'volcano':
            # 火山引擎特定配置
            api_base = self.config.get('volcano_engine_base_url')
            if not api_base:
                raise ValueError("使用火山引擎时必须配置 'volcano_engine_base_url'")
            # 注意: chromadb 的 OpenAIEmbeddingFunction 可能不直接兼容所有火山引擎的API细节
            # 可能需要自定义或使用火山引擎推荐的SDK/库进行嵌入
            # 此处暂时保留 OpenAIEmbeddingFunction，但需确认其兼容性或替换
            print(f"使用火山引擎嵌入: model={model_name}, base_url={api_base}")
            self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model_name,
                api_base=api_base
            )
        elif api_type == 'openai':
             # 标准 OpenAI API 配置
            api_base = self.config.get('openai_api_base') # 可选，用于代理或Azure OpenAI
            print(f"使用 OpenAI 嵌入: model={model_name}" + (f", base_url={api_base}" if api_base else ""))
            self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model_name,
                api_base=api_base # 如果提供了 api_base 则使用
            )
        else:
            raise ValueError(f"不支持的 embedding_api_type: {api_type}")
        # --- End Embedding Function Setup ---

        # --- ChromaDB Setup ---
        chroma_db_dir = os.path.abspath(self.config['chroma_db_path'])
        if not os.path.exists(chroma_db_dir):
            os.makedirs(chroma_db_dir, exist_ok=True)
            print(f"ChromaDB 目录已创建: {chroma_db_dir}")

        self.client = chromadb.PersistentClient(path=chroma_db_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.config['collection_name'],
            embedding_function=self.openai_ef # 将配置好的嵌入函数传递给集合
        )
        # --- End ChromaDB Setup ---

        # --- Text Splitter Setup ---
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.config['embedding_model'])
            print(f"已加载 tokenizer for model: {self.config['embedding_model']}")
        except KeyError:
             # 如果模型名称不在tiktoken预设中，使用通用分词器（可能不准确）
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            print(f"警告: 模型 {self.config['embedding_model']} 未在 tiktoken 中找到，使用 cl100k_base tokenizer。Token计数可能不完全准确。")
        
        # 从配置中获取分块大小和重叠大小，提供默认值
        # 注意：这里的 chunk_size 指的是 token 数量
        self.max_tokens_per_chunk = self.config.get('max_tokens_per_chunk', 4000) # 默认4000 token
        self.chunk_overlap_ratio = self.config.get('chunk_overlap_ratio', 0.1) # 默认10%重叠比例
        chunk_overlap = int(self.max_tokens_per_chunk * self.chunk_overlap_ratio)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens_per_chunk,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(self.tokenizer.encode(text, disallowed_special=())) # 使用tiktoken计算长度
        )
        print(f"文本分割器配置: chunk_size={self.max_tokens_per_chunk} tokens, chunk_overlap={chunk_overlap} tokens")
        # --- End Text Splitter Setup ---


    def _get_file_hash(self, file_path):
        """计算文件内容的MD5哈希值"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except IOError:
            return None # 文件读取错误则返回None

    def _get_embedding_strategy(self, strategy_name=None):
        """根据名称获取文本处理和嵌入策略函数"""
        strategy_name = strategy_name or self.config.get('embedding_strategy', 'chunking') # 默认'chunking'
        
        if strategy_name == "chunking":
            print("使用分块嵌入策略 (chunking strategy)")
            return self._embed_strategy_chunking
        elif strategy_name == "direct":
            # 保留一个直接处理的选项，用于短文件或测试
            print("使用直接嵌入策略 (direct strategy)")
            return self._embed_strategy_direct
        # 在这里可以添加其他策略的 elif 分支
        # elif strategy_name == "truncation":
        #     return self._embed_strategy_truncation
        else:
            print(f"警告: 未知的嵌入策略 '{strategy_name}'，将默认使用分块策略。")
            return self._embed_strategy_chunking # 默认回退到分块

    def _embed_strategy_direct(self, content, base_metadata, file_id_prefix):
        """
        直接嵌入策略：将整个内容视为单个文档。
        注意：如果内容超过模型限制，此策略会导致嵌入函数报错。
        """
        token_count = len(self.tokenizer.encode(content, disallowed_special=()))
        # 实际的模型限制可能需要从模型提供方获取，这里用配置的chunk size作为一个参考
        # 但即使小于chunk_size，实际模型的限制也可能更小
        # 嵌入函数本身会进行最终检查并报错
        print(f"直接嵌入: 文本长度 {token_count} tokens。")
        # if token_count > self.max_tokens_per_chunk: # 可以选择在此处添加预检查警告
        #     print(f"警告: 文件内容 ({token_count} tokens) 可能超过模型单次处理限制。")

        # 即使是直接策略，也返回列表形式，以保持接口一致
        return [content], [base_metadata], [f"{file_id_prefix}_doc"] # 使用 "_doc" 后缀

    def _embed_strategy_chunking(self, content, base_metadata, file_id_prefix):
        """分块嵌入策略：使用配置的文本分割器分割内容，并为每个块生成元数据和ID"""
        chunks = self.text_splitter.split_text(content)
        chunk_documents = []
        chunk_metadatas = []
        chunk_ids = []

        print(f"分块策略: 将内容分割成 {len(chunks)} 个块。")

        for i, chunk in enumerate(chunks):
            if not chunk.strip(): # 跳过完全是空白的块
                print(f"  跳过空块 chunk {i+1}/{len(chunks)}")
                continue
                
            chunk_id = f"{file_id_prefix}_chunk_{i}" # 为每个块生成唯一ID
            chunk_metadata = base_metadata.copy() # 复制基础元数据
            chunk_metadata.update({
                "chunk_index": i,            # 当前块的索引 (0-based)
                "total_chunks": len(chunks), # 文件总块数
                "chunk_length": len(chunk),  # 块的字符长度 (可选)
                "chunk_tokens": len(self.tokenizer.encode(chunk, disallowed_special=())) # 块的token数 (可选)
            })
            chunk_documents.append(chunk)
            chunk_metadatas.append(chunk_metadata)
            chunk_ids.append(chunk_id)
            print(f"  - Chunk {i}: ID={chunk_id}, Tokens={chunk_metadata['chunk_tokens']}") # Debugging

        return chunk_documents, chunk_metadatas, chunk_ids

    def index_codebase(self, source_dir_abs_path):
        """
        对指定目录下的源代码文件进行索引。
        会自动处理 .gitignore 文件。
        会根据配置的嵌入策略（如分块）处理长文件。
        会检查文件内容哈希，跳过未更改的文件，并更新已更改的文件（删除旧块，添加新块）。
        """
        print(f"开始索引源代码目录: {source_dir_abs_path}")
        if not os.path.isdir(source_dir_abs_path):
            print(f"错误:提供的路径不是一个有效的目录: {source_dir_abs_path}")
            return

        # --- .gitignore handling ---
        gitignore_path = os.path.join(source_dir_abs_path, '.gitignore')
        spec = None
        if os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    gitignore_content = f.read()
                # Add common VCS directories to ignore patterns implicitly
                # Gitignore patterns are relative to the .gitignore file location (source_dir_abs_path)
                spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_content.splitlines())
                print(f"已加载 .gitignore 文件: {gitignore_path}")
            except Exception as e:
                print(f"读取或解析 .gitignore 文件时出错: {e}")
        else:
            print("未找到 .gitignore 文件，将索引所有支持的文件。")
        # --- End .gitignore handling ---

        indexed_files_count = 0 # 记录处理的文件数
        indexed_chunks_count = 0 # 记录索引的块/文档总数
        skipped_files_count = 0
        updated_files_count = 0 # 记录更新的文件数
        error_files_count = 0   # 记录出错的文件数
        supported_extensions = tuple(self.config.get('source_file_extensions', []))

        # --- 选择嵌入策略 ---
        embedding_processor = self._get_embedding_strategy()
        # ---

        # Modify os.walk to exclude directories specified in .gitignore
        all_files_to_process = []
        for root, dirs, files in os.walk(source_dir_abs_path, topdown=True):
            # Filter directories in-place using list slicing
            # Get relative paths for matching against gitignore
            dirs[:] = [
                d for d in dirs 
                if not spec or not spec.match_file(os.path.relpath(os.path.join(root, d), source_dir_abs_path))
            ]
            
            for file_name in files:
                file_path_abs = os.path.join(root, file_name)
                relative_file_path = os.path.relpath(file_path_abs, source_dir_abs_path)

                # Check if the file path matches .gitignore rules
                if spec and spec.match_file(relative_file_path):
                    # print(f"根据 .gitignore 跳过文件: {relative_file_path}") # Optional: Add for debugging
                    continue # Skip ignored files
                
                # Check file extension after gitignore check
                if not file_name.endswith(supported_extensions):
                    continue # Skip non-supported extensions
                    
                all_files_to_process.append((file_path_abs, relative_file_path))

        # Process the filtered list of files
        for file_path_abs, relative_file_path in all_files_to_process:
            file_id_prefix = f"file_{hashlib.md5(file_path_abs.encode()).hexdigest()}" # 文件唯一标识符前缀

            try:
                # --- 文件变更检查 ---
                current_file_hash = self._get_file_hash(file_path_abs)
                if not current_file_hash: # 如果无法计算哈希（例如读取错误）
                     print(f"警告: 无法计算文件哈希，跳过文件 {relative_file_path}")
                     error_files_count += 1
                     continue

                # 查询数据库中是否已存在该文件的块 (通过文件绝对路径元数据)
                # 注意：这里假设所有属于同一个文件的块都共享相同的 'absolute_path' 和 'content_hash' 元数据
                existing_chunks = self.collection.get(
                    where={"absolute_path": file_path_abs},
                    include=["metadatas"] # 只需要元数据来检查哈希
                )

                existing_file_hash = None
                if existing_chunks and existing_chunks['ids']:
                    # 从第一个块的元数据中获取已存储的文件哈希
                    # 假设同一文件的所有块具有相同的 content_hash
                    existing_file_hash = existing_chunks['metadatas'][0].get('content_hash')

                if existing_file_hash == current_file_hash:
                    # 文件未更改，跳过
                    print(f"文件 {relative_file_path} 未更改，跳过索引。")
                    skipped_files_count += 1
                    continue # 处理下一个文件
                else:
                    # 文件是新的或已更改
                    if existing_file_hash is not None:
                        # 文件已更改，需要删除旧的块
                        print(f"文件 {relative_file_path} 内容已更改，删除旧条目...")
                        self.collection.delete(where={"absolute_path": file_path_abs})
                        print(f"旧条目已删除，准备重新索引。")
                        updated_files_count += 1 # 标记为更新
                    else:
                        # 新文件
                        print(f"索引新文件: {relative_file_path}")
                        indexed_files_count += 1 # 标记为新增索引
                # --- 文件变更检查结束 ---

                # --- 读取和处理文件内容 ---
                try:
                    with open(file_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except Exception as read_error:
                    print(f"读取文件 {relative_file_path} 时发生错误: {read_error}")
                    error_files_count += 1
                    continue # 跳过这个文件

                if not content.strip(): # 跳过内容为空或只有空白的文件
                    print(f"文件 {relative_file_path} 为空或只包含空白，跳过索引。")
                    # 如果之前有这个文件的块，也应该删除
                    if existing_file_hash is not None:
                         self.collection.delete(where={"absolute_path": file_path_abs})
                         print(f"已删除之前存在的空文件 {relative_file_path} 的条目。")
                    skipped_files_count += 1 # 计入跳过
                    continue
                
                # --- 使用选定的策略处理和嵌入文档 ---
                base_metadata = {
                    "source": relative_file_path,
                    "absolute_path": file_path_abs,
                    "content_hash": current_file_hash # 存储当前文件哈希，用于下次比较
                }
                
                # 调用策略函数处理内容
                documents, metadatas, ids = embedding_processor(content, base_metadata, file_id_prefix)

                if not documents: # 如果策略处理后没有返回任何文档/块
                    print(f"处理文件 {relative_file_path} 后无有效内容可索引，跳过。")
                    skipped_files_count += 1
                    continue
                
                # --- 添加到 ChromaDB ---
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                current_indexed_chunks = len(documents)
                indexed_chunks_count += current_indexed_chunks # 累加索引的块数
                print(f"已成功索引 {current_indexed_chunks} 个文档块来自文件: {relative_file_path}")
                # --- 添加完成 ---

            except Exception as e:
                # 捕获处理单个文件时的其他异常
                print(f"处理或索引文件 {relative_file_path} 时发生未知错误: {e}")
                error_files_count += 1
                # 考虑是否需要清理：如果错误发生在add之前，可能不需要；如果发生在add过程中，ChromaDB可能部分添加
                # 为了简化，这里仅记录错误并继续处理下一个文件
        
        print("--- 索引任务总结 ---")
        print(f"处理目录: {source_dir_abs_path}")
        print(f"总计扫描到符合条件的文件: {len(all_files_to_process)}")
        print(f"  - 新增索引文件数: {indexed_files_count}")
        print(f"  - 更新文件数 (重新索引): {updated_files_count}")
        print(f"  - 跳过未更改文件数: {skipped_files_count}")
        print(f"  - 跳过空文件或处理后无内容文件数: (包含在 skipped_files_count 或 error_files_count 中，取决于具体情况)")
        print(f"  - 处理失败文件数: {error_files_count}")
        print(f"本次任务总共索引/更新了 {indexed_chunks_count} 个文档块。")
        try:
            total_vectors = self.collection.count()
            print(f"当前向量数据库 '{self.config['collection_name']}' 中的向量总数: {total_vectors}")
        except Exception as count_error:
            print(f"无法获取向量总数: {count_error}")
        print("--- 索引完成 ---")


    def query_vector_db(self, query_text, n_results=5, filters: dict = None):
        """查询向量数据库以获取相关代码片段，支持基于元数据的过滤。"""
        query_params = {
            "query_texts": [query_text],
            "n_results": n_results,
            "include": ["documents", "metadatas"]
        }
        if filters:
            query_params["where"] = filters
            print(f"使用元数据过滤器进行查询: {filters}") # 添加日志
        
        results = self.collection.query(**query_params)
        return results

# Version: 1.1
# Updated: 2025-05-13 10:30:00 