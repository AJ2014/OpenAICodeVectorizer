from typing import List, Dict, Any
from .tool_manager import Tool # Assuming Tool class has get_description method

class PromptEngine:
    """负责动态构建和管理LLM的提示模板。"""

    def __init__(self, tools: List[Tool] = None):
        """
        初始化 PromptEngine。
        参数:
            tools (List[Tool], 可选): 一个包含可用工具对象的列表。
        """
        self.tools = tools if tools else []

    def _get_tool_descriptions(self) -> str:
        """获取所有已注册工具的描述字符串。"""
        if not self.tools:
            return "当前没有可用的工具。"
        descriptions = ["可用工具如下:"]
        for tool in self.tools:
            descriptions.append(f"- {tool.get_description()}") # Tool应有get_description方法
        return "\n".join(descriptions)

    def build_prompt(self, user_question: str, context_str: str = None, chat_history: List[Dict[str, str]] = None) -> str:
        """
        构建发送给LLM的完整提示。
        在第一阶段，我们先实现一个基础版本，后续会更加复杂化以支持不同的任务和上下文管理。

        参数:
            user_question (str): 用户的当前问题。
            context_str (str, 可选): 从向量数据库或其他来源检索到的上下文信息。
            chat_history (List[Dict[str, str]], 可选): 对话历史, e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        返回:
            str: 构建好的完整提示字符串。
        """
        prompt_parts = []

        # 1. System Persona / Role
        system_persona = "您是一个AI编程助手。请根据提供的上下文信息和可用工具，尽力回答用户的问题。"
        prompt_parts.append(system_persona)

        # 2. Tool Descriptions
        tool_descriptions = self._get_tool_descriptions()
        prompt_parts.append(tool_descriptions)
        
        # Note: Action instruction moved after user question for clarity in final prompt structure.

        # 3. Chat History
        if chat_history:
            history_section_parts = ["对话历史:"]
            for entry in chat_history:
                # Ensure content is a string, as tool results might be other types initially
                content = str(entry['content']) if not isinstance(entry['content'], str) else entry['content']
                history_section_parts.append(f"{entry['role']}: {content}")
            prompt_parts.append("\n".join(history_section_parts))

        # 4. Context (if any)
        # Orchestrator currently clears context_str after the first tool use, 
        # as tool results are then fed back via chat_history.
        if context_str:
            context_section_str = f"相关上下文信息:\n{context_str}"
            prompt_parts.append(context_section_str)
        
        # 5. User Question
        prompt_parts.append(f"用户当前问题: {user_question}")
        
        # 6. Action Instruction (Placed after the question to guide LLM's response structure)
        action_instruction = (
            "请分析以上信息。如果需要使用工具来获取额外信息或执行操作，"
            "请明确指出工具名称和所需参数，格式为：\n"
            "ACTION: tool_name(param1=value1, param2=value2)\n\n"
            "如果可以直接回答，请给出您的答案。"
        )
        prompt_parts.append(action_instruction)

        # 7. Answer Placeholder
        prompt_parts.append("回答:")
        
        return "\n\n".join(filter(None, prompt_parts)).strip() # filter(None, ...) to remove empty sections if any

# 示例 (后续会由Orchestrator使用)
# if __name__ == '__main__':
#     from tool_manager import ListDirectoryTool, ReadFileContentTool
#     list_tool = ListDirectoryTool()
#     read_tool = ReadFileContentTool()
#     engine = PromptEngine(tools=[list_tool, read_tool])
#     
#     # 简单问答
#     prompt1 = engine.build_prompt(user_question="你好吗？")
#     print("--- Prompt 1 (无上下文) ---")
#     print(prompt1)

#     # 带上下文的问答
#     context = "代码片段A显示了如何初始化一个类。\n代码片段B讨论了错误处理。"
#     prompt2 = engine.build_prompt(user_question="如何处理初始化时的错误？", context_str=context)
#     print("\n--- Prompt 2 (有上下文) ---")
#     print(prompt2)

#     # 带历史和上下文
#     history = [
#         {"role": "user", "content": "什么是Python中的类？"},
#         {"role": "assistant", "content": "类是创建对象的蓝图。"},
#         {"role": "user", "content": "Tool list_directory execution result: ['file1.py', 'file2.txt']"} # Example tool result in history
#     ]
#     prompt3 = engine.build_prompt(user_question="读取 file1.py 的内容", context_str=None, chat_history=history)
#     print("\n--- Prompt 3 (有历史和上下文, 无初始上下文) ---")
#     print(prompt3) 