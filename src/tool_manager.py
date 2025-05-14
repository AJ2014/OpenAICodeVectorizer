import os
import abc
from .embedding_service import EmbeddingService

# --- Custom Tool Exceptions ---
class ToolError(Exception):
    """所有工具相关错误的基类。"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ToolPathNotFoundError(ToolError, FileNotFoundError):
    """当工具期望的路径不存在时抛出。"""
    def __init__(self, path: str, message: str = None):
        self.path = path
        super().__init__(message or f"路径 '{path}' 未找到。")

class ToolPathIsNotDirectoryError(ToolError):
    """当工具期望一个目录路径，但提供的路径不是目录时抛出。"""
    def __init__(self, path: str, message: str = None):
        self.path = path
        super().__init__(message or f"路径 '{path}' 不是一个有效的目录。")

class ToolPathIsNotFileError(ToolError):
    """当工具期望一个文件路径，但提供的路径不是文件时抛出。"""
    def __init__(self, path: str, message: str = None):
        self.path = path
        super().__init__(message or f"路径 '{path}' 不是一个有效的文件。")

class ToolInvalidArgumentError(ToolError, ValueError):
    """当提供给工具的参数无效时抛出。"""
    def __init__(self, argument_name: str, value: any, reason: str, message: str = None):
        self.argument_name = argument_name
        self.value = value
        self.reason = reason
        super().__init__(message or f"参数 '{argument_name}' 的值 '{value}' 无效: {reason}")

class ToolExecutionError(ToolError):
    """当工具在执行过程中遇到未预期的错误时抛出。"""
    def __init__(self, tool_name: str, original_exception: Exception, message: str = None):
        self.tool_name = tool_name
        self.original_exception = original_exception
        super().__init__(message or f"工具 '{tool_name}' 执行失败: {str(original_exception)}")

# --- Tool Base Class ---
class Tool(abc.ABC):
    """工具的抽象基类"""
    name: str = "BaseTool"
    description: str = "这是一个基础工具模板。"

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> any: # Return type 'any' as it varies per tool
        """执行工具的具体逻辑"""
        raise NotImplementedError("每个工具都必须实现 'run' 方法。")

    def get_description(self) -> str:
        """返回工具的名称和描述，用于提供给LLM。"""
        return f"{self.name}: {self.description}"

# --- Concrete Tool Implementations ---
class ListDirectoryTool(Tool):
    """列出指定目录中的文件和子目录"""
    name: str = "list_directory"
    description: str = "列出指定项目路径下的文件和子目录。参数: path (str)。"

    def run(self, path: str) -> list[str]:
        """
        执行列出目录内容的逻辑。
        参数:
            path (str): 需要列出内容的目录路径。
        返回:
            list[str]: 目录下的文件名和子目录名列表。
        抛出:
            ToolPathNotFoundError: 如果路径不存在。
            ToolPathIsNotDirectoryError: 如果路径不是一个目录。
            ToolExecutionError: 如果在列出目录时发生其他OS错误。
        """
        if not os.path.exists(path):
            raise ToolPathNotFoundError(path=path)
        if not os.path.isdir(path):
            raise ToolPathIsNotDirectoryError(path=path)
        try:
            return os.listdir(path)
        except OSError as e: # Catching more specific OS-related errors
            raise ToolExecutionError(tool_name=self.name, original_exception=e, message=f"列出目录 '{path}' 内容时发生OS错误。")
        except Exception as e: # Catch-all for truly unexpected issues during listdir
            raise ToolExecutionError(tool_name=self.name, original_exception=e)

class ReadFileContentTool(Tool):
    """读取文件的全部或部分内容"""
    name: str = "read_file_content"
    description: str = (
        "读取指定文件的全部内容，或指定行号范围内的内容。"
        "参数: file_path (str), start_line (int, 可选, 1-based), end_line (int, 可选, 1-based)。"
    )

    def run(self, file_path: str, start_line: int = None, end_line: int = None) -> str:
        """
        执行读取文件内容的逻辑。
        参数:
            file_path (str): 要读取的文件路径。
            start_line (int, 可选): 开始读取的行号 (1-based)。如果为None，则从文件开头读取。
            end_line (int, 可选): 结束读取的行号 (1-based, 包含该行)。如果为None，则读取到文件末尾。
        返回:
            str: 文件内容。
        抛出:
            ToolPathNotFoundError: 如果文件路径不存在。
            ToolPathIsNotFileError: 如果路径不是一个文件。
            ToolInvalidArgumentError: 如果行号参数无效 (例如 start_line > end_line)。
            ToolExecutionError: 如果在读取文件时发生其他IO错误或编码错误。
        """
        if not os.path.exists(file_path):
            raise ToolPathNotFoundError(path=file_path)
        if not os.path.isfile(file_path):
            raise ToolPathIsNotFileError(path=file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if start_line is None and end_line is None:
                    return f.read()
                else:
                    lines = f.readlines()
                    actual_start_line = 0
                    if start_line is not None:
                        if not isinstance(start_line, int):
                            raise ToolInvalidArgumentError("start_line", start_line, "必须是整数。")
                        if start_line <= 0: # 1-based, so 0 or negative is invalid for direct use
                             actual_start_line = 0 # Treat as from beginning
                        else:
                            actual_start_line = start_line - 1
                    
                    actual_end_line = len(lines)
                    if end_line is not None:
                        if not isinstance(end_line, int):
                            raise ToolInvalidArgumentError("end_line", end_line, "必须是整数。")
                        if end_line < 0 : # Negative end_line is not intuitive
                            raise ToolInvalidArgumentError("end_line", end_line, "不能为负数。")
                        actual_end_line = end_line # Slicing handles end_line > len(lines) gracefully

                    if actual_start_line > actual_end_line : # Check after actual_end_line is determined if start_line was given
                         # This condition is tricky if only end_line is given and start_line is default 0
                         # Re-evaluating the condition based on provided values
                         if start_line is not None and end_line is not None and start_line > end_line:
                            raise ToolInvalidArgumentError("line_range", f"({start_line}-{end_line})", "start_line 不能大于 end_line。")

                    # Ensure start_line is not beyond the file if specified
                    if actual_start_line >= len(lines) and start_line is not None:
                        return "" # Return empty if start is beyond content, consistent with slicing

                    return "".join(lines[actual_start_line:actual_end_line])
        except ToolInvalidArgumentError:
            raise
        except (IOError, OSError, UnicodeDecodeError) as e: # More specific exceptions for file operations
            raise ToolExecutionError(tool_name=self.name, original_exception=e, message=f"读取文件 '{file_path}' 时发生错误。")
        except Exception as e: # Catch-all for truly unexpected issues during file read
            raise ToolExecutionError(tool_name=self.name, original_exception=e)

class QueryVectorDBTool(Tool):
    """查询向量数据库以检索与查询文本相关的代码片段，支持元数据过滤。"""
    name: str = "query_vector_db"
    description: str = (
        "根据查询文本从向量数据库中检索最相关的代码块。"
        "参数: query_text (str, 必需), n_results (int, 可选, 默认5), "
        "filters (dict, 可选, 例如 {\'source\': \'path/to/file.py\'}). " # Escaped for literal_eval
        "filters 中的键应对应索引时存储的元数据字段。"
    )

    def __init__(self, embedding_service: EmbeddingService):
        if not isinstance(embedding_service, EmbeddingService):
            # 这更像是一个编程错误，而不是用户通过LLM调用工具时会遇到的错误
            # 所以使用标准的TypeError可能更合适
            raise TypeError("QueryVectorDBTool 需要一个 EmbeddingService 的实例。")
        self.embedding_service = embedding_service

    def run(self, query_text: str, n_results: int = 5, filters: dict = None) -> dict:
        """
        执行向量数据库查询。
        参数:
            query_text (str): 用于查询的文本。
            n_results (int): 要返回的结果数量，默认为5。
            filters (dict, 可选): 用于过滤结果的元数据字典。
                                 例如: {"source": "src/utils.py"}
        返回:
            dict: ChromaDB 查询结果，通常包含 'documents', 'metadatas', 'ids'。
        抛出:
            ToolInvalidArgumentError: 如果参数无效。
            ToolExecutionError: 如果在查询过程中发生错误。
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ToolInvalidArgumentError("query_text", query_text, "必须是一个非空字符串。")
        
        if not isinstance(n_results, int) or n_results <= 0:
            raise ToolInvalidArgumentError("n_results", n_results, "必须是一个正整数。")

        if filters is not None and not isinstance(filters, dict):
            # 进一步检查 filters 的内容是否符合预期 (例如，键是字符串)
            # 但由于ChromaDB的where子句比较灵活，这里仅做基本类型检查
            raise ToolInvalidArgumentError("filters", filters, "必须是一个字典或None。")

        try:
            results = self.embedding_service.query_vector_db(
                query_text=query_text,
                n_results=n_results,
                filters=filters
            )
            # 当前返回原始ChromaDB结果。如果需要，后续可以转换为更简洁的格式。
            # 例如，将文档和元数据配对：
            # simplified_results = []
            # if results and results.get('documents') and results.get('metadatas'):
            #     # Assuming query_texts was a single string, so documents[0] and metadatas[0]
            #     docs = results['documents'][0]
            #     metas = results['metadatas'][0]
            #     for doc, meta in zip(docs, metas):
            #         simplified_results.append({"document": doc, "metadata": meta})
            # return simplified_results
            return results 
        except Exception as e:
            raise ToolExecutionError(tool_name=self.name, original_exception=e, message=f"查询向量数据库时出错: {str(e)}")

# Example usage (for testing, Orchestrator would use this differently)
# if __name__ == '__main__':
#     list_tool = ListDirectoryTool()
#     read_tool = ReadFileContentTool()

#     print("--- Tool Descriptions ---")
#     print(list_tool.get_description())
#     print(read_tool.get_description())
    
#     print("\n--- ListDirectoryTool Examples ---")
#     try:
#         print(f"Contents of '.': {list_tool.run('.')}")
#         # print(f"Contents of 'src': {list_tool.run('src')}") # Assuming 'src' exists
#         # list_tool.run('non_existent_dir') 
#     except ToolError as e:
#         print(f"ListTool Error: {e}")

#     print("\n--- ReadFileContentTool Examples ---")
#     # Create a dummy file for testing read
#     dummy_file = "dummy_test_file.txt"
#     with open(dummy_file, "w", encoding="utf-8") as f:
#         f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
    
#     try:
#         print(f"All content of '{dummy_file}':\n{read_tool.run(dummy_file)}")
#         print(f"Lines 2-3 of '{dummy_file}':\n{read_tool.run(dummy_file, start_line=2, end_line=3)}")
#         print(f"From line 4 of '{dummy_file}':\n{read_tool.run(dummy_file, start_line=4)}")
#         print(f"Up to line 2 of '{dummy_file}':\n{read_tool.run(dummy_file, end_line=2)}")
#         # read_tool.run("non_existent_file.txt")
#         # read_tool.run(dummy_file, start_line=4, end_line=2)
#     except ToolError as e:
#         print(f"ReadTool Error: {e}")
#     finally:
#         if os.path.exists(dummy_file):
#             os.remove(dummy_file) 