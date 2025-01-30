from unittest.mock import MagicMock


def test_calculate_metrics(mock_provider):

    metrics = mock_provider._calculate_metrics(
        input="Hello",
        output="Hello World",
        model="test_model",
        start_time=0.0,
        end_time=1.0,
        first_token_time=0.5,
        token_times=(0.1,),
        token_count=2,
        is_stream=True,
    )

    assert metrics["input_tokens"] == 1
    assert metrics["output_tokens"] == 2
    assert metrics["total_tokens"] == 3
    assert metrics["cost_usd"] == 0.01 * 1 + 0.02 * 2  # input_cost + output_cost
    assert metrics["latency_s"] == 1.0  # end_time - start_time
    assert (
        metrics["time_to_first_token_s"] == 0.5 - 0.0
    )  # first_token_time - start_time
    assert metrics["inter_token_latency_s"] == 0.1  # Average of token_times
    assert metrics["tokens_per_second"] == 2 / 1.0  # token_count / total_time


def test_calculate_metrics_single_token(mock_provider):

    metrics = mock_provider._calculate_metrics(
        input="Hello",
        output="World",
        model="test_model",
        start_time=0.0,
        end_time=1.0,
        first_token_time=0.5,
        token_times=(),
        token_count=1,
        is_stream=True,
    )

    assert metrics["input_tokens"] == 1
    assert metrics["output_tokens"] == 1
    assert metrics["total_tokens"] == 2
    assert metrics["cost_usd"] == 0.01 * 1 + 0.02 * 1
    assert metrics["latency_s"] == 1.0
    assert metrics["time_to_first_token_s"] == 0.5 - 0.0
    assert metrics["inter_token_latency_s"] == 0
    assert metrics["tokens_per_second"] == 1 / 1.0


def test_calculate_cost_fixed_cost(mock_provider):
    fixed_cost = 0.02
    token_count = 100
    expected_cost = token_count * fixed_cost
    assert mock_provider._calculate_cost(token_count, fixed_cost) == expected_cost


def test_calculate_cost_variable_cost(mock_provider):
    cost_range_1 = MagicMock()
    cost_range_1.range = (0, 50)
    cost_range_1.cost = 0.01

    cost_range_2 = MagicMock()
    cost_range_2.range = (51, 100)
    cost_range_2.cost = 0.02

    variable_cost = [cost_range_1, cost_range_2]
    token_count = 75
    expected_cost = token_count * 0.02
    assert mock_provider._calculate_cost(token_count, variable_cost) == expected_cost


def test_calculate_cost_variable_cost_higher_range(mock_provider):
    cost_range_1 = MagicMock()
    cost_range_1.range = (0, 50)
    cost_range_1.cost = 0.01

    cost_range_2 = MagicMock()
    cost_range_2.range = (51, 100)
    cost_range_2.cost = 0.02

    cost_range_3 = MagicMock()
    cost_range_3.range = (101, None)
    cost_range_3.cost = 0.03

    variable_cost = [cost_range_1, cost_range_2, cost_range_3]
    token_count = 150
    expected_cost = token_count * 0.03
    assert mock_provider._calculate_cost(token_count, variable_cost) == expected_cost


def test_calculate_cost_variable_cost_no_matching_range(mock_provider):
    cost_range_1 = MagicMock()
    cost_range_1.range = (0, 50)
    cost_range_1.cost = 0.01

    cost_range_2 = MagicMock()
    cost_range_2.range = (51, 100)
    cost_range_2.cost = 0.02

    cost_range_3 = MagicMock()
    cost_range_3.range = (101, 150)
    cost_range_3.cost = 0.03

    variable_cost = [cost_range_1, cost_range_2, cost_range_3]
    token_count = 200
    expected_cost = 0
    assert mock_provider._calculate_cost(token_count, variable_cost) == expected_cost


def test_calculate_cost_variable_cost_no_matching_range_inferior(mock_provider):
    cost_range_1 = MagicMock()
    cost_range_1.range = (10, 50)
    cost_range_1.cost = 0.01

    cost_range_2 = MagicMock()
    cost_range_2.range = (51, 100)
    cost_range_2.cost = 0.02

    cost_range_3 = MagicMock()
    cost_range_3.range = (101, 150)
    cost_range_3.cost = 0.03

    variable_cost = [cost_range_1, cost_range_2, cost_range_3]
    token_count = 5
    expected_cost = 0
    assert mock_provider._calculate_cost(token_count, variable_cost) == expected_cost
