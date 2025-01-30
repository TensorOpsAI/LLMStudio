from unittest.mock import MagicMock

import pytest
from llmstudio_core.providers.provider import ChatRequest, ProviderError, time

request = ChatRequest(chat_input="Hello World", model="test_model")


def test_chat_response_non_stream(mock_provider):
    mock_provider.validate_request = MagicMock()
    mock_provider.validate_model = MagicMock()
    mock_provider.generate_client = MagicMock(return_value="mock_response")
    mock_provider._handle_response = MagicMock(return_value="final_response")

    response = mock_provider.chat(chat_input="Hello", model="test_model")

    assert response == "final_response"
    mock_provider.validate_request.assert_called_once()
    mock_provider.validate_model.assert_called_once()


def test_chat_streaming_response(mock_provider):
    mock_provider.validate_request = MagicMock()
    mock_provider.validate_model = MagicMock()
    mock_provider.generate_client = MagicMock(return_value="mock_response_stream")
    mock_provider._handle_response = MagicMock(
        return_value=iter(["streamed_response_1", "streamed_response_2"])
    )

    response_stream = mock_provider.chat(
        chat_input="Hello", model="test_model", is_stream=True
    )
    assert next(response_stream) == "streamed_response_1"
    assert next(response_stream) == "streamed_response_2"
    mock_provider.validate_request.assert_called_once()
    mock_provider.validate_model.assert_called_once()


def test_validate_model(mock_provider):
    request = ChatRequest(chat_input="Hello", model="test_model")
    mock_provider.validate_model(request)  # Should not raise

    request_invalid = ChatRequest(chat_input="Hello", model="invalid_model")
    with pytest.raises(ProviderError):
        mock_provider.validate_model(request_invalid)


def test_join_chunks_finish_reason_stop(mock_provider):
    current_time = int(time.time())
    chunks = [
        {
            "id": "test_id",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {"content": "Hello, "},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
        },
        {
            "id": "test_id",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {"content": "world!"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
        },
    ]
    response, output_string = mock_provider._join_chunks(chunks)

    assert output_string == "Hello, world!"
    assert response.choices[0].message.content == "Hello, world!"


def test_join_chunks_finish_reason_function_call(mock_provider):
    current_time = int(time.time())
    chunks = [
        {
            "id": "test_id",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {
                        "function_call": {"name": "my_function", "arguments": "arg1"}
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
        },
        {
            "id": "test_id",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {"function_call": {"arguments": "arg2"}},
                    "finish_reason": "function_call",
                    "index": 0,
                }
            ],
        },
        {
            "id": "test_id",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {"function_call": {"arguments": "}"}},
                    "finish_reason": "function_call",
                    "index": 0,
                }
            ],
        },
    ]
    response, output_string = mock_provider._join_chunks(chunks)

    assert output_string == "arg1arg2"
    assert response.choices[0].message.function_call.arguments == "arg1arg2"
    assert response.choices[0].message.function_call.name == "my_function"


def test_join_chunks_tool_calls(mock_provider):
    current_time = int(time.time())

    chunks = [
        {
            "id": "test_id_1",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "tool_1",
                                "index": 0,
                                "function": {
                                    "name": "search_tool",
                                    "arguments": '{"query": "weather',
                                },
                                "type": "function",
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
        },
        {
            "id": "test_id_2",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "tool_1",
                                "index": 0,
                                "function": {
                                    "name": "search_tool",
                                    "arguments": ' details"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                    "index": 0,
                }
            ],
        },
    ]

    response, output_string = mock_provider._join_chunks(chunks)

    assert output_string == "['search_tool', '{\"query\": \"weather details\"}']"

    assert response.object == "chat.completion"
    assert response.choices[0].finish_reason == "tool_calls"
    tool_call = response.choices[0].message.tool_calls[0]

    assert tool_call.id == "tool_1"
    assert tool_call.function.name == "search_tool"
    assert tool_call.function.arguments == '{"query": "weather details"}'
    assert tool_call.type == "function"


def test_input_to_string_with_string(mock_provider):
    input_data = "Hello, world!"
    assert mock_provider._input_to_string(input_data) == "Hello, world!"


def test_input_to_string_with_list_of_text_messages(mock_provider):
    input_data = [
        {"content": "Hello"},
        {"content": " world!"},
    ]
    assert mock_provider._input_to_string(input_data) == "Hello world!"


def test_input_to_string_with_list_of_text_and_url(mock_provider):
    input_data = [
        {"role": "user", "content": [{"type": "text", "text": "Hello "}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.jpg"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": " world!"}]},
    ]
    expected_output = "Hello http://example.com/image.jpg world!"
    assert mock_provider._input_to_string(input_data) == expected_output


def test_input_to_string_with_mixed_roles_and_missing_content(mock_provider):
    input_data = [
        {"role": "assistant", "content": "Admin text;"},
        {"role": "user", "content": [{"type": "text", "text": "User text"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/another.jpg"},
                }
            ],
        },
    ]
    expected_output = "Admin text;User texthttp://example.com/another.jpg"
    assert mock_provider._input_to_string(input_data) == expected_output


def test_input_to_string_with_missing_content_key(mock_provider):
    input_data = [
        {"role": "user"},
        {"role": "user", "content": [{"type": "text", "text": "Hello again"}]},
    ]
    expected_output = "Hello again"
    assert mock_provider._input_to_string(input_data) == expected_output


def test_input_to_string_with_empty_list(mock_provider):
    input_data = []
    assert mock_provider._input_to_string(input_data) == ""
