from unittest.mock import MagicMock, patch


class TestBuildLlamaSystemMessage:
    def test_build_llama_system_message_with_existing_sm(self, mock_azure_provider):
        mock_azure_provider.build_tool_instructions = MagicMock(
            return_value="Tool Instructions"
        )
        mock_azure_provider.build_function_instructions = MagicMock(
            return_value="\nFunction Instructions"
        )

        openai_message = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Custom system message"},
        ]
        llama_message = "Initial message"
        tools = ["Tool1", "Tool2"]
        functions = ["Function1"]

        result = mock_azure_provider.build_llama_system_message(
            openai_message, llama_message, tools, functions
        )

        expected = (
            "Initial message\n"
            "        <|start_header_id|>system<|end_header_id|>\n"
            "        Custom system message\n"
            "        Tool Instructions\nFunction Instructions\n<|eot_id|>"  # identation here exists because in Python when adding a newline to a triple quote string it keeps identation
        )
        assert result == expected
        mock_azure_provider.build_tool_instructions.assert_called_once_with(tools)
        mock_azure_provider.build_function_instructions.assert_called_once_with(
            functions
        )

    def test_build_llama_system_message_with_default_sm(self, mock_azure_provider):
        mock_azure_provider.build_tool_instructions = MagicMock(
            return_value="Tool Instructions"
        )
        mock_azure_provider.build_function_instructions = MagicMock(
            return_value="\nFunction Instructions"
        )

        openai_message = [{"role": "user", "content": "Hello"}]
        llama_message = "Initial message"
        tools = ["Tool1"]
        functions = []

        result = mock_azure_provider.build_llama_system_message(
            openai_message, llama_message, tools, functions
        )

        expected = (
            "Initial message\n"
            "      <|start_header_id|>system<|end_header_id|>\n"
            "      You are a helpful AI assistant.\n"
            "      Tool Instructions\n<|eot_id|>"
        )
        assert result == expected
        mock_azure_provider.build_tool_instructions.assert_called_once_with(tools)
        mock_azure_provider.build_function_instructions.assert_not_called()

    def test_build_llama_system_message_without_tools_or_functions(
        self, mock_azure_provider
    ):
        mock_azure_provider.build_tool_instructions = MagicMock()
        mock_azure_provider.build_function_instructions = MagicMock()

        openai_message = [{"role": "system", "content": "Minimal system message"}]
        llama_message = "Initial message"
        tools = []
        functions = []

        result = mock_azure_provider.build_llama_system_message(
            openai_message, llama_message, tools, functions
        )

        expected = (
            "Initial message\n"
            "        <|start_header_id|>system<|end_header_id|>\n"
            "        Minimal system message\n        \n<|eot_id|>"
        )
        assert result == expected
        mock_azure_provider.build_tool_instructions.assert_not_called()
        mock_azure_provider.build_function_instructions.assert_not_called()


class TestBuildInstructions:
    def test_build_tool_instructions(self, mock_azure_provider):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "python_repl_ast",
                    "description": "execute Python code",
                    "parameters": {"query": "string"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "data_lookup",
                    "description": "retrieve data from a database",
                    "parameters": {"database": "string", "query": "string"},
                },
            },
        ]

        result = mock_azure_provider.build_tool_instructions(tools)

        expected = """
    You have access to the following tools:
    Use the function 'python_repl_ast' to 'execute Python code':
Parameters format:
{
    "query": "string"
}

Use the function 'data_lookup' to 'retrieve data from a database':
Parameters format:
{
    "database": "string",
    "query": "string"
}


If you choose to use a function to produce this response, ONLY reply in the following format with no prefix or suffix:
§{"type": "function", "name": "FUNCTION_NAME", "parameters": {"PARAMETER_NAME": PARAMETER_VALUE}}
IMPORTANT: IT IS VITAL THAT YOU NEVER ADD A PREFIX OR A SUFFIX TO THE FUNCTION CALL.

Here is an example of the output I desiere when performing function call:
§{"type": "function", "name": "python_repl_ast", "parameters": {"query": "print(df.shape)"}}
NOTE: There is no prefix before the symbol '§' and nothing comes after the call is done.

    Reminder:
    - Function calls MUST follow the specified format.
    - Only call one function at a time.
    - Required parameters MUST be specified.
    - Put the entire function call reply on one line.
    - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
    - If you have already called a tool and got the response for the users question please reply with the response.
    """
        assert result.strip() == expected.strip()

    def test_build_function_instructions(self, mock_azure_provider):
        functions = [
            {
                "name": "python_repl_ast",
                "description": "execute Python code",
                "parameters": {"query": "string"},
            },
            {
                "name": "data_lookup",
                "description": "retrieve data from a database",
                "parameters": {"database": "string", "query": "string"},
            },
        ]

        result = mock_azure_provider.build_function_instructions(functions)

        expected = """
You have access to the following functions:
Use the function 'python_repl_ast' to: 'execute Python code'
{
    "query": "string"
}

Use the function 'data_lookup' to: 'retrieve data from a database'
{
    "database": "string",
    "query": "string"
}


If you choose to use a function to produce this response, ONLY reply in the following format with no prefix or suffix:
§{"type": "function", "name": "FUNCTION_NAME", "parameters": {"PARAMETER_NAME": PARAMETER_VALUE}}

Here is an example of the output I desiere when performing function call:
§{"type": "function", "name": "python_repl_ast", "parameters": {"query": "print(df.shape)"}}

Reminder:
- Function calls MUST follow the specified format.
- Only call one function at a time.
- NEVER call more than one function at a time.
- Required parameters MUST be specified.
- Put the entire function call reply on one line.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
- If you have already called a function and got the response for the user's question, please reply with the response.
"""

        assert result.strip() == expected.strip()


class TestBuildLlamaConversation:
    def test_build_llama_conversation_with_nested_messages(self, mock_azure_provider):
        mock_azure_provider.format_message = MagicMock(
            side_effect=lambda msg: f"[formatted:{msg['content']}]"
        )

        openai_message = [
            {
                "role": "user",
                "content": "[{'content': 'nested message 1'}, {'content': 'nested message 2'}]",
            },
            {"role": "assistant", "content": "assistant reply"},
        ]
        llama_message = "Initial message: "

        result = mock_azure_provider.build_llama_conversation(
            openai_message, llama_message
        )

        expected = "Initial message: [formatted:nested message 1][formatted:nested message 2][formatted:assistant reply]"

        assert result == expected
        mock_azure_provider.format_message.assert_any_call(
            {"content": "nested message 1"}
        )
        mock_azure_provider.format_message.assert_any_call(
            {"content": "nested message 2"}
        )
        mock_azure_provider.format_message.assert_any_call(
            {"role": "assistant", "content": "assistant reply"}
        )

    def test_build_llama_conversation_with_invalid_nested_content(
        self, mock_azure_provider
    ):
        mock_azure_provider.format_message = MagicMock(
            side_effect=lambda msg: f"[formatted:{msg['content']}]"
        )

        openai_message = [
            {"role": "user", "content": "[invalid json/dict]"},
            {"role": "assistant", "content": "assistant reply"},
        ]
        llama_message = "Initial message: "

        with patch("ast.literal_eval", side_effect=ValueError) as mock_literal_eval:
            result = mock_azure_provider.build_llama_conversation(
                openai_message, llama_message
            )

            expected = "Initial message: [formatted:[invalid json/dict]][formatted:assistant reply]"

            assert result == expected
            mock_azure_provider.format_message.assert_any_call(
                {"role": "user", "content": "[invalid json/dict]"}
            )
            mock_azure_provider.format_message.assert_any_call(
                {"role": "assistant", "content": "assistant reply"}
            )

            mock_literal_eval.assert_called_once_with("[invalid json/dict]")

    def test_build_llama_conversation_skipping_system_messages(
        self, mock_azure_provider
    ):
        mock_azure_provider.format_message = MagicMock(
            side_effect=lambda msg: f"[formatted:{msg['content']}]"
        )

        openai_message = [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "user message"},
        ]
        llama_message = "Initial message: "

        result = mock_azure_provider.build_llama_conversation(
            openai_message, llama_message
        )

        expected = "Initial message: [formatted:user message]"

        assert result == expected
        mock_azure_provider.format_message.assert_any_call(
            {"role": "user", "content": "user message"}
        )
