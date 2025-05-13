import os
import chromadb
from chromadb.utils import embedding_functions
from .utils import load_config
import hashlib
import pathspec

# 更新时间：2025-05-12 14:51:59
# Version: 1.0
# Updated: 2025-05-12 16:50:46

class EmbeddingService:
    def __init__(self):
        self.config = load_config()
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.config['openai_api_key'],
            model_name=self.config['embedding_model'],
            api_base=self.config.get('volcano_engine_base_url')
        )
        # 使用绝对路径初始化ChromaDB客户端
        chroma_db_dir = os.path.abspath(self.config['chroma_db_path'])
        if not os.path.exists(chroma_db_dir):
            os.makedirs(chroma_db_dir, exist_ok=True)
            print(f"ChromaDB 目录已创建: {chroma_db_dir}")

        self.client = chromadb.PersistentClient(path=chroma_db_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.config['collection_name'],
            embedding_function=self.openai_ef
        )

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

    def index_codebase(self, source_dir_abs_path):
        """对指定目录下的源代码文件进行索引"""
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

        indexed_files_count = 0
        skipped_files_count = 0
        supported_extensions = tuple(self.config.get('source_file_extensions', []))

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
            file_id = f"file_{hashlib.md5(file_path_abs.encode()).hexdigest()}" # 使用路径哈希作为稳定ID

            try:
                # 检查文件是否已存在且未更改
                existing_doc = self.collection.get(ids=[file_id], include=["metadatas"])
                current_file_hash = self._get_file_hash(file_path_abs)

                if existing_doc and existing_doc['ids']:
                    if existing_doc['metadatas'][0].get('content_hash') == current_file_hash:
                        print(f"文件 {relative_file_path} 未更改，跳过索引。")
                        skipped_files_count += 1
                        continue 
                    else:
                        print(f"文件 {relative_file_path} 内容已更改，重新索引。")
                        # 如果内容有变，先删除旧的
                        self.collection.delete(ids=[file_id])
                
                with open(file_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip(): # 跳过空文件
                    print(f"文件 {relative_file_path} 为空，跳过索引。")
                    skipped_files_count += 1
                    continue

                self.collection.add(
                    documents=[content],
                    metadatas=[{
                        "source": relative_file_path, 
                        "absolute_path": file_path_abs,
                        "content_hash": current_file_hash
                    }],
                    ids=[file_id]
                )
                indexed_files_count += 1
                print(f"已索引文件: {relative_file_path}")

            except Exception as e:
                print(f"索引文件 {relative_file_path} 时发生错误: {e}")
                skipped_files_count += 1
        
        print(f"索引完成。成功索引 {indexed_files_count} 个文件，跳过 {skipped_files_count} 个文件。")
        print(f"向量总数: {self.collection.count()}")

    def query_vector_db(self, query_text, n_results=5):
        """查询向量数据库以获取相关代码片段"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        return results

# Version: 1.0
# Updated: 2025-05-12 14:51:59
# Version: 1.1
# Updated: 2025-05-12 16:50:46 