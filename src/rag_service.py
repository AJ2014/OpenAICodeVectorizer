from openai import OpenAI
from .utils import load_config
from .embedding_service import EmbeddingService

# 更新时间：2025-05-12 14:51:59
# Version: 1.0
# Updated: 2025-05-12 16:50:46

class RAGService:
    def __init__(self):
        self.config = load_config()
        self.client = OpenAI(
            api_key=self.config['openai_api_key'],
            base_url=self.config.get('volcano_engine_base_url')
        )
        self.embedding_service = EmbeddingService() # 用于查询向量数据库

    def answer_question(self, question, n_results_for_context=3):
        """根据问题和向量数据库中的上下文生成答案"""
        print(f"收到问题: {question}")
        
        # 1. 从向量数据库检索相关代码片段
        print(f"正在从向量数据库检索上下文 (top {n_results_for_context})...")
        retrieved_results = self.embedding_service.query_vector_db(
            query_text=question,
            n_results=n_results_for_context
        )

        if not retrieved_results or not retrieved_results.get('documents') or not retrieved_results['documents'][0]:
            print("未能从向量数据库中检索到相关上下文。请确保已成功索引代码库。")
            return "抱歉，我无法在已索引的知识库中找到与您问题相关的信息。请尝试更具体的问题或检查索引是否完整。"

        context_parts = []
        for i, docs in enumerate(retrieved_results['documents']):
            for j, doc_content in enumerate(docs):
                metadata = retrieved_results['metadatas'][i][j]
                file_path = metadata.get('source', '未知文件')
                print(f"相关代码片段来自 '{file_path}'")
                context_parts.append(f"相关代码片段来自 '{file_path}':\n```\n{doc_content}\n```")
        
        context_str = "\n\n".join(context_parts)
        # print(f"检索到的上下文摘要:\n{context_str[:1000]}...") # 打印部分上下文用于调试

        # 2. 构建提示 (Prompt)
        prompt = f"""
        您是一个AI编程助手。请根据以下从代码库中检索到的上下文信息，回答用户的问题。
        如果上下文信息不足以回答问题，请明确指出。
        不要编造信息。

        上下文信息:
        {context_str}

        用户问题: {question}

        回答:
        """

        # 3. 调用LLM生成答案
        print(f"正在使用 LLM ({self.config['llm_model']}) 生成答案...")
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "您是一个AI编程助手。请根据提供的上下文信息，回答用户的问题。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.config['llm_model'],
            )
            answer = chat_completion.choices[0].message.content
            print("LLM 生成的回答已收到。")
            return answer
        except Exception as e:
            print(f"调用 OpenAI LLM 时发生错误: {e}")
            return f"抱歉，在尝试生成答案时遇到错误: {e}"

# Version: 1.0
# Updated: 2025-05-12 14:51:59
# Version: 1.1
# Updated: 2025-05-12 16:50:46 