import unittest
from unittest.mock import MagicMock
import os
import shutil

from openai import OpenAI
from src.orchestrator import Orchestrator
from src.prompt_engine import PromptEngine
from src.tool_manager import ListDirectoryTool, ReadFileContentTool, Tool, QueryVectorDBTool
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion import Choice
from src.embedding_service import EmbeddingService

class TestOrchestratorIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup a temporary directory for list_directory tool to use."""
        cls.test_dir = "tests/temp_orchestrator_test_dir"
        cls.file_in_test_dir = os.path.join(cls.test_dir, "sample_file.txt")
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.makedirs(cls.test_dir, exist_ok=True)
        with open(cls.file_in_test_dir, "w") as f:
            f.write("Test content for orchestrator integration.")
        print(f"Created test file: {cls.file_in_test_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        print(f"Deleted test directory: {cls.test_dir}")

    def setUp(self):
        """Setup tools and prompt engine for each test."""
        self.list_tool = ListDirectoryTool()
        self.read_tool = ReadFileContentTool()
        
        self.mock_embedding_service = MagicMock(spec=EmbeddingService)
        self.query_db_tool = QueryVectorDBTool(embedding_service=self.mock_embedding_service)
        
        self.available_tools: list[Tool] = [self.list_tool, self.read_tool, self.query_db_tool]
        
        self.prompt_engine_instance = PromptEngine(tools=self.available_tools)
        # Orchestrator instance will be created in each test method with its own mock client
        # So, no self.orchestrator initialization here anymore.
        # The print statement for orchestrator init will also be removed from here.

    def test_process_query_with_one_tool_call_and_final_answer(self):
        """
        Test a scenario where LLM requests a tool, gets a result, and then provides a final answer.
        The OpenAI client is mocked and passed directly to the Orchestrator.
        """
        print("Starting test_process_query_with_one_tool_call_and_final_answer")
        
        # --- Create and Configure Mock OpenAI Client ---
        mock_openai_client = MagicMock(autospec=OpenAI)

        # --- Setup Mock LLM Responses ---
        # Response 1: LLM requests to list files in the test_dir
        mock_llm_response_1_content = f"Okay, I need to see the files. ACTION: list_directory(path=\"{self.test_dir}\")"
        mock_chat_completion_1 = ChatCompletion(
            id="chatcmpl-mock1",
            choices=[
                Choice(
                    finish_reason="stop", 
                    index=0, 
                    message=ChatCompletionMessage(role="assistant", content=mock_llm_response_1_content, tool_calls=None), # Ensure tool_calls is None if not used
                    logprobs=None
                )
            ],
            created=1677652288,
            model="gpt-3.5-turbo-0125",
            object="chat.completion",
            system_fingerprint=None,
            usage=None
        )

        # Response 2: LLM gives a final answer after seeing the list_directory result
        # (The orchestrator will feed the actual list_directory result back to the LLM)
        # Let's assume the LLM then says something based on seeing "sample_file.txt"
        mock_llm_response_2_content = "I see 'sample_file.txt' in the directory. That is the file."
        mock_chat_completion_2 = ChatCompletion(
            id="chatcmpl-mock2",
            choices=[
                Choice(
                    finish_reason="stop", 
                    index=0, 
                    message=ChatCompletionMessage(role="assistant", content=mock_llm_response_2_content, tool_calls=None),
                    logprobs=None
                )
            ],
            created=1677652289,
            model="gpt-3.5-turbo-0125",
            object="chat.completion",
            system_fingerprint=None,
            usage=None
        )

        # Configure the mock OpenAI client instance
        mock_openai_client.chat.completions.create.side_effect = [
            mock_chat_completion_1, 
            mock_chat_completion_2
        ]
        
        # --- Instantiate Orchestrator with the mock client ---
        orchestrator_instance = Orchestrator(
            tools=self.available_tools,
            prompt_engine=self.prompt_engine_instance,
            max_iterations=3,
            client=mock_openai_client # Pass the mock client directly
        )
        print(f"Initialized Orchestrator with max_iterations={orchestrator_instance.max_iterations} and model='{orchestrator_instance.llm_model}'. Client is mocked.")

        # --- Execute the Orchestrator ---
        user_query = f"What files are in the directory '{self.test_dir}'?"
        final_answer = orchestrator_instance.process_query(user_question=user_query)

        # --- Assertions ---
        # 1. Check if the LLM was called twice
        self.assertEqual(mock_openai_client.chat.completions.create.call_count, 2)

        # 2. Check the arguments of the first LLM call
        _first_call_pos_args, first_call_kwargs = mock_openai_client.chat.completions.create.call_args_list[0]
        # The last message sent to LLM in the first call should contain the user_query and tool info
        self.assertIn(user_query, first_call_kwargs['messages'][-1]['content'])
        self.assertIn("list_directory", first_call_kwargs['messages'][-1]['content'])

        # 3. Check the arguments of the second LLM call
        _second_call_pos_args, second_call_kwargs = mock_openai_client.chat.completions.create.call_args_list[1]
        # The last message sent to LLM in the second call should contain the result of list_directory
        # The actual result of list_directory(self.test_dir) will be ["sample_file.txt"]
        self.assertIn(f"Tool 'list_directory' execution result: {['sample_file.txt']}", second_call_kwargs['messages'][-1]['content'])

        # 4. Check if the final answer is what the mock LLM provided as the final answer
        self.assertEqual(final_answer, mock_llm_response_2_content)

        # 5. Check tool execution (optional, by spying on the tool if needed, or by checking side effects)
        # In this case, list_directory was called, so self.test_dir was accessed.

    def test_process_query_with_query_vector_db_tool(self):
        """
        Test a scenario where LLM requests query_vector_db, gets a result, and then provides a final answer.
        The EmbeddingService is mocked, and the OpenAI client is mocked and passed directly.
        """
        print("Starting test_process_query_with_query_vector_db_tool")

        # --- Create and Configure Mock OpenAI Client ---
        mock_openai_client = MagicMock(autospec=OpenAI)

        # --- Setup Mock LLM and Tool Responses ---
        # 1. Mock EmbeddingService response (via the QueryVectorDBTool's instance)
        mock_db_query_text = "find similar code"
        mock_db_n_results = 3
        mock_db_filters = {"file_type": ".py"}
        mock_db_response = {
            "documents": [["def foo():\n  pass", "class Bar:\n  pass"]],
            "metadatas": [[{"source": "a.py", "file_type": ".py"}, {"source": "b.py", "file_type": ".py"}]],
            "ids": [["id_a", "id_b"]]
        }
        self.mock_embedding_service.query_vector_db.return_value = mock_db_response

        # 2. Mock LLM Responses
        # Response 1: LLM requests to query the vector database
        # Note the string escaping for the filters argument for ast.literal_eval
        # ACTION: query_vector_db(query_text="find similar code", n_results=3, filters={'file_type': '.py'})
        escaped_filters_str = str(mock_db_filters)

        mock_llm_response_1_content = f"ACTION: query_vector_db(query_text=\"{mock_db_query_text}\", n_results={mock_db_n_results}, filters={escaped_filters_str})"
        mock_chat_completion_1 = ChatCompletion(
            id="chatcmpl-mock-qdb1", choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content=mock_llm_response_1_content, tool_calls=None),logprobs=None)],
            created=1677652288, model="gpt-3.5-turbo-0125", object="chat.completion" # Using a default model here for the mock object
        )

        # Response 2: LLM gives a final answer based on the DB query result
        mock_llm_response_2_content = f"Found 2 code snippets: a.py and b.py. The first is 'def foo()...'"
        mock_chat_completion_2 = ChatCompletion(
            id="chatcmpl-mock-qdb2", choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content=mock_llm_response_2_content, tool_calls=None), logprobs=None)],
            created=1677652289, model="gpt-3.5-turbo-0125", object="chat.completion"
        )

        # Configure the mock OpenAI client instance
        mock_openai_client.chat.completions.create.side_effect = [
            mock_chat_completion_1,
            mock_chat_completion_2
        ]
        
        # --- Instantiate Orchestrator with the mock client ---
        orchestrator_instance = Orchestrator(
            tools=self.available_tools,
            prompt_engine=self.prompt_engine_instance,
            max_iterations=3,
            client=mock_openai_client # Pass the mock client directly
        )
        # We can use orchestrator_instance.llm_model for the mock ChatCompletion model if needed, or hardcode as above.
        # For consistency, let's ensure the mock ChatCompletion objects use the same model as the orchestrator instance would try to use.
        mock_chat_completion_1.model = orchestrator_instance.llm_model
        mock_chat_completion_2.model = orchestrator_instance.llm_model
        print(f"Initialized Orchestrator with max_iterations={orchestrator_instance.max_iterations} and model='{orchestrator_instance.llm_model}'. Client is mocked.")

        # --- Execute the Orchestrator ---
        user_query = "Search for python code snippets about 'similar code'."
        final_answer = orchestrator_instance.process_query(user_question=user_query)

        # --- Assertions ---
        # 1. LLM called twice
        self.assertEqual(mock_openai_client.chat.completions.create.call_count, 2)

        # 2. EmbeddingService.query_vector_db called correctly by the tool
        self.mock_embedding_service.query_vector_db.assert_called_once_with(
            query_text=mock_db_query_text,
            n_results=mock_db_n_results,
            filters=mock_db_filters
        )

        # 3. Second LLM call contains the result from query_vector_db
        _second_call_pos_args, second_call_kwargs = mock_openai_client.chat.completions.create.call_args_list[1]
        # The content sent to LLM should contain the stringified version of mock_db_response
        # Orchestrator wraps tool output like: f"Tool '{tool_name}' execution result: {tool_feedback}"
        expected_tool_feedback_in_prompt = f"Tool 'query_vector_db' execution result: {str(mock_db_response)}"
        self.assertIn(expected_tool_feedback_in_prompt, second_call_kwargs['messages'][-1]['content'])

        # 4. Final answer is correct
        self.assertEqual(final_answer, mock_llm_response_2_content)

if __name__ == '__main__':
    unittest.main() 