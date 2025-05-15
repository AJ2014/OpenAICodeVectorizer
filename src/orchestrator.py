import re
import ast # For safely evaluating string representations of arguments
from openai import OpenAI
from typing import List, Dict, Any, Optional
import os # Added for __main__ example

from .utils import load_config
from .prompt_engine import PromptEngine
from .tool_manager import Tool, ToolError, ToolPathNotFoundError, ToolInvalidArgumentError, ToolExecutionError
from .rag_service import RAGService

class Orchestrator:
    """负责协调LLM、工具和提示引擎以完成复杂任务。"""

    def __init__(self, 
                 tools: List[Tool], 
                 prompt_engine: PromptEngine, 
                 client: Optional[OpenAI] = None, # Added client parameter for DI
                 rag_service: Optional[RAGService] = None, # <--- ADDED RAGSERVICE PARAM
                 max_iterations: int = 5):
        """
        初始化 Orchestrator。
        参数:
            tools (List[Tool]): 可用工具的列表。
            prompt_engine (PromptEngine): 用于构建提示的 PromptEngine 实例。
            client (Optional[OpenAI]): 一个可选的 OpenAI 客户端实例。如果提供，则使用此实例；否则，内部创建。
            rag_service (Optional[RAGService]): 一个可选的 RAGService 实例，用于获取初步答案。
            max_iterations (int): 在单个查询中允许的最大LLM调用和工具执行迭代次数。
        """
        self.config = load_config()
        
        if client:
            self.client = client
        else:
            # Default behavior: create its own client if not provided
            self.client = OpenAI(
                api_key=self.config['openai_api_key'],
                base_url=self.config.get('volcano_engine_base_url') # 或者其他LLM提供商的base_url
            )
            
        self.tools = {tool.name: tool for tool in tools} # Store tools in a dict for easy lookup
        self.prompt_engine = prompt_engine
        self.rag_service = rag_service  # <--- STORE RAGSERVICE
        self.max_iterations = max_iterations
        self.llm_model = self.config.get('llm_model', 'gpt-3.5-turbo') # Default LLM model
        # Allow model override from config for flexibility e.g. gpt-4
        if 'llm_model' in self.config and self.config['llm_model']:
            self.llm_model = self.config['llm_model']

    def _parse_llm_action(self, llm_output: str) -> Optional[Dict[str, Any]]:
        """
        解析LLM的输出，查找工具调用指令。
        期望格式: ACTION: tool_name(param1=value1, param2="string value", ...)
        返回:
            一个包含 'tool_name' 和 'arguments' (dict) 的字典，如果找到action。
            否则返回 None。
        """
        match = re.search(r"ACTION:\s*(\w+)\s*\(([^)]*)\)", llm_output, re.IGNORECASE)
        if not match:
            return None

        tool_name = match.group(1).strip()
        args_str = match.group(2).strip()
        
        arguments = {}
        if args_str:
            try:
                raw_args = args_str.split(',')
                for arg_pair in raw_args:
                    arg_pair = arg_pair.strip() # Ensure no leading/trailing whitespace for the pair
                    if not arg_pair: continue # Skip if the pair is empty after splitting

                    if '=' not in arg_pair:
                        print(f"Warning: Argument '{arg_pair}' in '{args_str}' for tool '{tool_name}' is not a key-value pair. Skipping.")
                        continue
                    
                    key, value_str = arg_pair.split('=', 1)
                    key = key.strip()
                    value_str = value_str.strip()
                    
                    try:
                        value = ast.literal_eval(value_str)
                    except (ValueError, SyntaxError):
                        value = value_str.strip('\"\'') 
                    arguments[key] = value
            except Exception as e:
                print(f"Error parsing arguments '{args_str}' for tool '{tool_name}': {e}")
                return {"tool_name": tool_name, "error_parsing_args": str(e), "raw_args_str": args_str}

        return {"tool_name": tool_name, "arguments": arguments}

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        执行指定的工具并返回结果或错误信息的字符串。
        """
        if tool_name not in self.tools:
            return f"工具执行错误: 未找到名为 '{tool_name}' 的工具。请从可用工具列表中选择。"

        tool_to_execute = self.tools[tool_name]
        try:
            result = tool_to_execute.run(**arguments)
            if not isinstance(result, str):
                return str(result) # Ensure result is a string
            return result
        except ToolError as e:
            return f"工具 '{tool_name}' 执行失败: {e.message}" 
        except TypeError as e: 
             return f"工具 '{tool_name}' 参数错误: {str(e)}. 请检查工具描述中的参数和您的调用。"
        except Exception as e:
            print(f"Unexpected error executing tool '{tool_name}' with args '{arguments}': {type(e).__name__} - {e}")
            return f"工具 '{tool_name}' 执行时发生意外内部错误: {type(e).__name__} - {str(e)}"

    def process_query(self, user_question: str, initial_context: Optional[str] = None) -> str:
        """
        处理用户查询，可能涉及多轮LLM调用和工具执行。
        """
        chat_history: List[Dict[str, str]] = []
        current_context = initial_context if initial_context else ""
        
        # The initial user question doesn't go into chat_history directly for the *API call* yet,
        # as prompt_engine.build_prompt incorporates it into the first complex prompt.
        # However, for tracking the conversation flow, we might consider adding it to an internal log.

        # --- 调用 RAGService 获取初步答案 (如果可用) ---
        if self.rag_service:
            print(f"Orchestrator: 调用 RAGService 为问题提供初步上下文: '{user_question}'")
            rag_n_results = self.config.get('rag_initial_context_results', 3)
            try:
                rag_answer = self.rag_service.answer_question(user_question, n_results_for_context=rag_n_results)
                print(f"Orchestrator: RAGService 返回的初步答案 (前300字符): {rag_answer[:300]}...")
                # 将RAG的答案整合到初始上下文中，供PromptEngine使用
                # 如果已有initial_context，则附加；否则，以此为基础。
                rag_context_prefix = (
                    "一个初步的RAG（检索增强生成）系统尝试回答了这个问题。 "
                    "请回顾以下初步答案。如果它看起来准确且完整，您可以直接使用或优化它作为您的最终回复。"
                    "如果初步答案不充分或不正确，请利用您的工具来收集更多信息并形成一个全面准确的最终答案。\n\n"
                    "初步RAG答案：\n---\n"
                )
                rag_context_suffix = "\n---\n"
                formatted_rag_answer = f"{rag_context_prefix}{rag_answer}{rag_context_suffix}"
                
                if current_context: # 如果用户也提供了 initial_context
                    current_context = f"{formatted_rag_answer}\n\n明确提供的初始上下文:\n{current_context}"
                else:
                    current_context = formatted_rag_answer
            except Exception as e:
                print(f"Orchestrator: 调用 RAGService 时出错: {e}")
                # 可以选择将错误信息也加入 current_context，或者仅记录并继续
                error_message = f"尝试从RAG服务获取初步答案时发生错误: {e}。将不使用RAG的输出继续处理。"
                if current_context:
                    current_context += f"\n\n注意: {error_message}"
                else:
                    current_context = f"注意: {error_message}"
        # --- RAGService 调用结束 ---
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Orchestrator: Iteration {iteration + 1} ---")

            current_prompt = self.prompt_engine.build_prompt(
                user_question=user_question, 
                context_str=current_context, # current_context 现在可能包含RAG的输出
                chat_history=chat_history 
            )
            # 在第一次迭代后，清除 current_context，因为后续上下文将通过 chat_history 传递
            if iteration == 0:
                 current_context = "" 

            print(f"Orchestrator: Prompt to LLM (first 500 chars):\n{current_prompt[:500]}..." + ("" if len(current_prompt) <= 500 else " [TRUNCATED]"))

            try:
                messages_for_api = []
                for entry in chat_history: # This history has assistant msgs (LLM thoughts/actions) and user msgs (tool results)
                    messages_for_api.append(entry)
                messages_for_api.append({"role": "user", "content": current_prompt})

                print(f"Orchestrator: Sending to LLM model: {self.llm_model}")
                chat_completion = self.client.chat.completions.create(
                    messages=messages_for_api,
                    model=self.llm_model,
                    temperature=0.2, # Lower temperature for more predictable tool usage
                )
                llm_response_content = chat_completion.choices[0].message.content.strip()
                print(f"Orchestrator: LLM Response:\n{llm_response_content}")
                
            except Exception as e:
                print(f"Orchestrator: LLM API call failed: {e}")
                return f"抱歉，与语言模型交互时发生错误: {type(e).__name__} - {e}"

            action = self._parse_llm_action(llm_response_content)

            if action:
                # LLM decided to use a tool
                chat_history.append({"role": "assistant", "content": llm_response_content}) # LLM's thought/action request
                
                tool_feedback = ""
                if "error_parsing_args" in action:
                    tool_name_for_feedback = action.get("tool_name", "unknown_tool")
                    tool_feedback = f"参数解析错误: {action['error_parsing_args']}. 您提供的参数字符串是: '{action['raw_args_str']}' for tool '{tool_name_for_feedback}'"
                    print(f"Orchestrator: {tool_feedback}")
                else:
                    tool_name = action["tool_name"]
                    arguments = action["arguments"]
                    print(f"Orchestrator: LLM requests ACTION: {tool_name} with arguments: {arguments}")
                    
                    tool_feedback = self._execute_tool(tool_name, arguments)
                    print(f"Orchestrator: Tool '{tool_name}' execution feedback:\n{tool_feedback}")

                # Feed tool result back to LLM via chat history
                # This message simulates the environment/user providing the tool's output.
                chat_history.append({"role": "user", "content": f"Tool '{action.get("tool_name", "unknown")}' execution result: {tool_feedback}"})
                current_context = "" # Context is now in history
            else:
                # LLM provided a final answer (no action detected)
                print("Orchestrator: LLM provided a final answer.")
                # We don't add this final answer to chat_history here because the history is for the *next* turn.
                # The calling function will receive this answer.
                return llm_response_content
        
        print("Orchestrator: Reached max iterations.")
        return "抱歉，我已经尝试了多次（达到最大迭代次数），但仍无法处理您的请求。请尝试简化问题或提供更多信息。"

if __name__ == '__main__':
    from .tool_manager import ListDirectoryTool, ReadFileContentTool, QueryVectorDBTool
    from .embedding_service import EmbeddingService
    from .rag_service import RAGService

    print("Orchestrator __main__ block starting...")

    # --- 配置和初始化 --- 
    # 通常, EmbeddingService 和工具的实例化会在应用的更高层级进行

    # 1. Setup EmbeddingService (如果需要 QueryVectorDBTool)
    try:
        print("Initializing EmbeddingService...")
        embedding_service_instance = EmbeddingService()
        print("EmbeddingService initialized.")
        # 可以选择在这里执行一次索引，如果数据库是空的或需要更新
        # print("Attempting to index codebase for main test...")
        # test_codebase_path = os.path.abspath("../../tests/test_data/sample_project") # 调整路径
        # if not os.path.exists(test_codebase_path):
        #     os.makedirs(test_codebase_path, exist_ok=True)
        #     with open(os.path.join(test_codebase_path, "main.py"), "w") as f:
        #         f.write("print('Hello from sample project')")
        # embedding_service_instance.index_codebase(test_codebase_path)
        # print("Codebase indexing complete for main test.")

    except Exception as e:
        print(f"Error initializing EmbeddingService or indexing: {e}")
        print("QueryVectorDBTool 将不可用。继续执行，但仅使用文件系统工具。")
        embedding_service_instance = None # 确保后续不会因未初始化而出错

    # 2. Setup Tools
    print("Initializing tools...")
    list_tool = ListDirectoryTool()
    read_tool = ReadFileContentTool()
    available_tools = [list_tool, read_tool]

    if embedding_service_instance:
        query_db_tool = QueryVectorDBTool(embedding_service=embedding_service_instance)
        available_tools.append(query_db_tool)
        print("QueryVectorDBTool initialized and added.")
    else:
        print("Skipping QueryVectorDBTool initialization as EmbeddingService failed.")

    print(f"Initialized tools: {[tool.name for tool in available_tools]}")

    # 3. Setup RAGService (if needed by Orchestrator)
    rag_service_instance = None
    try:
        print("Initializing RAGService...")
        rag_service_instance = RAGService() # RAGService creates its own EmbeddingService internally for now
        print("RAGService initialized.")
    except Exception as e:
        print(f"Error initializing RAGService: {e}. Orchestrator will run without RAG pre-processing.")
    # --- End RAGService Setup ---

    # 4. Setup PromptEngine
    print("Initializing PromptEngine...")
    prompt_engine_instance = PromptEngine(tools=available_tools)
    print("Initialized PromptEngine.")

    # 5. Setup Orchestrator
    print("Initializing Orchestrator...")
    orchestrator_instance = Orchestrator(
        tools=available_tools, 
        prompt_engine=prompt_engine_instance,
        rag_service=rag_service_instance
    )
    print("Orchestrator initialized.")

    # --- 测试交互 ---
    # 创建一个临时目录和文件用于测试 ListDirectoryTool 和 ReadFileContentTool
    test_dir = "./orchestrator_test_dir"
    test_file = os.path.join(test_dir, "test_file.txt")
    os.makedirs(test_dir, exist_ok=True)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Hello from Orchestrator test file!\nLine 2.")
    print(f"Created temporary test directory and file for demo: {test_dir}")

    # 构造一个需要多次交互的示例问题
    # Example 1: List, then read
    # user_query = f"请列出目录 '{test_dir}' 下的文件，然后读取文件 '{test_file}' 的内容。"
    
    # Example 2: Query Vector DB (if available and indexed)
    if rag_service_instance:
        # Example query that might benefit from RAG + Tools
        user_query = "请根据我的代码库解释一下'main.py'文件是做什么的，并列出当前项目根目录下的所有文件。"
        # To make RAG effective for this, ensure 'main.py' or relevant content is indexed.
        # You might need to uncomment and run embedding_service_instance.index_codebase in the EmbeddingService setup part above.
        print("--- RAG Service is active. Using a query that might leverage it. ---")
    elif embedding_service_instance and 'query_vector_db' in orchestrator_instance.tools:
        user_query = "Search the vector database for documents related to 'test_file.txt' or 'orchestrator'. Then list the contents of the current directory."
        # 注意: 为了让这个查询有效, 你需要确保运行了 index_codebase, 并且 test_file.txt 或类似内容被索引了。
        # 上面的 embedding_service_instance.index_codebase(test_codebase_path) 需要被取消注释并正确配置路径。
        # 默认情况下，如果没有索引，这个查询可能会返回空结果或错误（取决于ChromaDB的具体行为）
    else:
        user_query = f"请列出目录 '{test_dir}' 下的文件，然后读取文件 '{test_file}' 的内容。"
        print("Skipping vector DB query example as it might not be set up.")

    print(f"\n--- Processing User Query ---")
    print(f"User Query: {user_query}")
    
    final_answer = orchestrator_instance.process_query(user_query)
    
    print(f"\n--- Final Answer from Orchestrator ---")
    print(final_answer)

    # 清理临时文件和目录
    print(f"Cleaning up temporary test directory: {test_dir}")
    os.remove(test_file)
    os.rmdir(test_dir)
    print("Cleanup complete.")
    print("Orchestrator __main__ block finished.") 