import argparse
import os
from src.embedding_service import EmbeddingService
from src.rag_service import RAGService
from src.utils import load_config
from src.visualize import visualize_chroma_vectors
from src.orchestrator import Orchestrator
from src.prompt_engine import PromptEngine
from src.tool_manager import ListDirectoryTool, ReadFileContentTool, QueryVectorDBTool

# 更新时间：2025-05-12 14:51:59
# Version: 1.0
# Updated: 2025-05-12 16:50:46

def main():
    parser = argparse.ArgumentParser(description="代码库索引和问答工具 (基于OpenAI API)")
    subparsers = parser.add_subparsers(dest="command", help="可执行的命令", required=True)

    # 索引命令
    index_parser = subparsers.add_parser("index", help="索引源代码目录")
    index_parser.add_argument("source_dir", type=str, help="要索引的源代码工程的绝对路径")

    # 问答命令
    ask_parser = subparsers.add_parser("ask", help="就已索引的代码库提问")
    ask_parser.add_argument("question", type=str, help="您的问题")
    ask_parser.add_argument("--n_results", type=int, default=3, help="检索时返回的相关上下文片段数量 (RAGService直接调用时生效)")
    ask_parser.add_argument("--use-orchestrator", action="store_true", help="使用 Orchestrator 而不是直接的 RAGService 进行问答")

    # 可视化命令
    vis_parser = subparsers.add_parser("visualize", help="可视化chroma向量数据库为散点图")
    vis_parser.add_argument("--chroma_db_path", type=str, default=None, help="chroma数据库路径（可选，默认读取config.yaml）")
    vis_parser.add_argument("--collection_name", type=str, default=None, help="集合名（可选，默认读取config.yaml）")
    vis_parser.add_argument("--max_points", type=int, default=2000, help="最大可视化点数（默认2000）")

    args = parser.parse_args()

    try:
        config = load_config()
        if not config.get('openai_api_key') or config['openai_api_key'] == "YOUR_OPENAI_API_KEY_HERE":
            print("错误：请在 config.yaml 文件中配置您的 API 密钥 (openai_api_key)。")
            return
        if not config.get('volcano_engine_base_url') or config['volcano_engine_base_url'] == "YOUR_VOLCANO_ENGINE_BASE_URL_HERE":
            print("错误：请在 config.yaml 文件中配置火山引擎或第三方服务的基础 URL (volcano_engine_base_url)。")
            return
    except FileNotFoundError:
        print("错误：找不到 config.yaml 配置文件。请确保它在项目根目录下。")
        return
    except Exception as e:
        print(f"加载配置时出错: {e}")
        return

    if args.command == "index":
        source_directory_abs = os.path.abspath(args.source_dir)
        if not os.path.isdir(source_directory_abs):
            print(f"错误：提供的源代码目录 '{source_directory_abs}' 不是一个有效的目录或不存在。")
            return
        print(f"使用绝对路径进行索引: {source_directory_abs}")
        try:
            embed_service = EmbeddingService()
            embed_service.index_codebase(source_directory_abs)
        except Exception as e:
            print(f"执行索引操作时发生严重错误: {e}")

    elif args.command == "ask":
        if args.use_orchestrator:
            try:
                print(f"正在使用 Orchestrator 回答问题: \"{args.question}\"")
                # 1. 初始化 RAGService (Orchestrator 可能需要它)
                rag_service = RAGService()

                # 2. 初始化 EmbeddingService (QueryVectorDBTool 需要它)
                # 注意: EmbeddingService 会尝试从配置文件加载其设置
                embedding_service = EmbeddingService()
                
                # 3. 初始化工具
                list_tool = ListDirectoryTool()
                read_tool = ReadFileContentTool()
                query_db_tool = QueryVectorDBTool(embedding_service=embedding_service)
                available_tools = [list_tool, read_tool, query_db_tool]

                # 4. 初始化 PromptEngine
                prompt_engine = PromptEngine(tools=available_tools)

                # 5. 初始化 Orchestrator
                # Orchestrator 会使用内部配置或默认值来创建 OpenAI 客户端，
                # 并管理 LLM 模型、最大迭代次数等。
                orchestrator_instance = Orchestrator(
                    tools=available_tools,
                    prompt_engine=prompt_engine,
                    rag_service=rag_service
                )
                
                answer = orchestrator_instance.process_query(args.question)
                print("\nOrchestrator 回答:")
                print(answer)
            except Exception as e:
                print(f"使用 Orchestrator 执行问答操作时发生严重错误: {e}")
        else: # 默认使用 RAGService
            try:
                print(f"正在使用 RAGService 直接回答问题: \"{args.question}\"")
                rag_service = RAGService()
                answer = rag_service.answer_question(args.question, n_results_for_context=args.n_results)
                print("\nRAGService 回答:")
                print(answer)
            except Exception as e:
                print(f"使用 RAGService 执行问答操作时发生严重错误: {e}")

    elif args.command == "visualize":
        try:
            visualize_chroma_vectors(
                chroma_db_path=args.chroma_db_path,
                collection_name=args.collection_name,
                max_points=args.max_points
            )
        except Exception as e:
            print(f"可视化时发生严重错误: {e}")

if __name__ == "__main__":
    main()

# Version: 1.0
# Updated: 2025-05-12 14:51:59
# Version: 1.1
# Updated: 2025-05-12 16:50:46
# Version: 1.2
# Updated: 2025-05-12 17:10:00 