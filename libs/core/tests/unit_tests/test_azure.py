from unittest.mock import MagicMock


class TestParseResponse:
    def test_tool_response_handling(self, mock_azure_provider):

        mock_azure_provider.is_llama = True
        mock_azure_provider.has_tools = True
        mock_azure_provider.has_functions = False

        mock_azure_provider.handle_tool_response = MagicMock(
            return_value=iter(["chunk1", None, "chunk2", None, "chunk3"])
        )

        response = iter(["irrelevant"])

        results = list(mock_azure_provider.parse_response(response))

        assert results == ["chunk1", "chunk2", "chunk3"]
        mock_azure_provider.handle_tool_response.assert_called_once_with(response)

    def test_direct_response_handling_with_choices(self, mock_azure_provider):
        mock_azure_provider.is_llama = False

        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {"choices": ["choice1", "choice2"]}
        chunk2 = MagicMock()
        chunk2.model_dump.return_value = {"choices": ["choice2"]}
        response = iter([chunk1, chunk2])

        results = list(mock_azure_provider.parse_response(response))

        assert results == [
            {"choices": ["choice1", "choice2"]},
            {"choices": ["choice2"]},
        ]
        chunk1.model_dump.assert_called_once()
        chunk2.model_dump.assert_called_once()

    def test_direct_response_handling_without_choices(self, mock_azure_provider):
        mock_azure_provider.is_llama = False

        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {"key": "value"}
        chunk2 = MagicMock()
        chunk2.model_dump.return_value = {"another_key": "another_value"}
        response = iter([chunk1, chunk2])

        results = list(mock_azure_provider.parse_response(response))

        assert results == []
        chunk1.model_dump.assert_called_once()
        chunk2.model_dump.assert_called_once()


class TestFormatMessage:
    def test_format_message_tool_calls(self, mock_azure_provider):
        message = {
            "tool_calls": [
                {
                    "function": {
                        "name": "example_tool",
                        "arguments": '{"arg1": "value1"}',
                    }
                }
            ]
        }
        result = mock_azure_provider.format_message(message)
        expected = """
        <|start_header_id|>assistant<|end_header_id|>
        <function=example_tool>{"arg1": "value1"}</function>
        <|eom_id|>
            """
        assert result.strip() == expected.strip()

    def test_format_message_tool_call_id(self, mock_azure_provider):
        message = {"tool_call_id": "123", "content": "This is the tool response."}
        result = mock_azure_provider.format_message(message)
        expected = """
    <|start_header_id|>ipython<|end_header_id|>
    This is the tool response.
    <|eot_id|>
        """
        assert result.strip() == expected.strip()

    def test_format_message_function_call(self, mock_azure_provider):
        message = {
            "function_call": {
                "name": "example_function",
                "arguments": '{"arg1": "value1"}',
            }
        }
        result = mock_azure_provider.format_message(message)
        expected = """
        <|start_header_id|>assistant<|end_header_id|>
        <function=example_function>{"arg1": "value1"}</function>
        <|eom_id|>
            """
        assert result.strip() == expected.strip()

    def test_format_message_user_message(self, mock_azure_provider):
        message = {"role": "user", "content": "This is a user message."}
        result = mock_azure_provider.format_message(message)
        expected = """
    <|start_header_id|>user<|end_header_id|>
    This is a user message.
    <|eot_id|>
        """
        assert result.strip() == expected.strip()

    def test_format_message_assistant_message(self, mock_azure_provider):
        message = {"role": "assistant", "content": "This is an assistant message."}
        result = mock_azure_provider.format_message(message)
        expected = """
    <|start_header_id|>assistant<|end_header_id|>
    This is an assistant message.
    <|eot_id|>
        """
        assert result.strip() == expected.strip()

    def test_format_message_function_response(self, mock_azure_provider):
        message = {"role": "function", "content": "This is the function response."}
        result = mock_azure_provider.format_message(message)
        expected = """
    <|start_header_id|>ipython<|end_header_id|>
    This is the function response.
    <|eot_id|>
        """
        assert result.strip() == expected.strip()

    def test_format_message_empty_message(self, mock_azure_provider):
        message = {"role": "user", "content": None}
        result = mock_azure_provider.format_message(message)
        expected = ""
        assert result == expected


class TestGenerateClient:
    def test_generate_client_with_tools_and_functions(self, mock_azure_provider):
        mock_azure_provider.prepare_messages = MagicMock(
            return_value="prepared_messages"
        )
        mock_azure_provider._client.chat.completions.create = MagicMock(
            return_value="mock_response"
        )

        request = MagicMock()
        request.model = "gpt-4"
        request.parameters = {
            "tools": ["tool1", "tool2"],
            "functions": ["function1", "function2"],
            "other_param": "value",
        }

        result = mock_azure_provider.generate_client(request)

        expected_args = {
            "model": "gpt-4",
            "messages": "prepared_messages",
            "stream": True,
            "tools": ["tool1", "tool2"],
            "tool_choice": "auto",
            "functions": ["function1", "function2"],
            "function_call": "auto",
            "other_param": "value",
        }

        assert result == "mock_response"
        mock_azure_provider.prepare_messages.assert_called_once_with(request)
        mock_azure_provider._client.chat.completions.create.assert_called_once_with(
            **expected_args
        )

    def test_generate_client_without_tools_or_functions(self, mock_azure_provider):
        mock_azure_provider.prepare_messages = MagicMock(
            return_value="prepared_messages"
        )
        mock_azure_provider._client.chat.completions.create = MagicMock(
            return_value="mock_response"
        )

        request = MagicMock()
        request.model = "gpt-4"
        request.parameters = {"other_param": "value"}

        result = mock_azure_provider.generate_client(request)

        expected_args = {
            "model": "gpt-4",
            "messages": "prepared_messages",
            "stream": True,
            "other_param": "value",
        }

        assert result == "mock_response"
        mock_azure_provider.prepare_messages.assert_called_once_with(request)
        mock_azure_provider._client.chat.completions.create.assert_called_once_with(
            **expected_args
        )

    def test_generate_client_with_llama_model(self, mock_azure_provider):
        mock_azure_provider.prepare_messages = MagicMock(
            return_value="prepared_messages"
        )
        mock_azure_provider._client.chat.completions.create = MagicMock(
            return_value="mock_response"
        )

        request = MagicMock()
        request.model = "llama-2"
        request.parameters = {
            "tools": ["tool1"],
            "functions": ["function1"],
            "other_param": "value",
        }

        result = mock_azure_provider.generate_client(request)

        expected_args = {
            "model": "llama-2",
            "messages": "prepared_messages",
            "stream": True,
            "tools": ["tool1"],
            "functions": ["function1"],
            "other_param": "value",
        }

        assert result == "mock_response"
        mock_azure_provider.prepare_messages.assert_called_once_with(request)
        mock_azure_provider._client.chat.completions.create.assert_called_once_with(
            **expected_args
        )
