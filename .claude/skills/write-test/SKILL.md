---
name: write-test
description: Write unit tests according to the following requirement.
---

Write unit tests according to the following requirement.

#$ARGUMENTS

# Guidelines on writing unit tests

## Use of mocks

✅ Mock when:

- The function to be tested relies on external dependency such as databases, web APIs, or file systems
  - Example dependency are MLFlow and Slack related function/classes.
- When a dependent procedure (not the procedure to be tested) in the function to be tested is slow to execute

Example

```python
from unittest.mock import MagicMock

@mock.patch("mlflow.log_params")  # mocked due to the need to interact with external mlflow server
@mock.patch("mlflow.log_artifact")  # similar as above
@mock.patch("mlflow.pyfunc.log_model")  # similar as above
@mock.patch("metaflow_pipeline.run_hyperparameter_tuning") # mocked because HP tuning can be heavy to run
def test_train_model_step(
    mock_hp_tuning: MagicMock,
    mock_log_model: MagicMock,
    mock_log_artifact: MagicMock,
    mock_log_params: MagicMock
) -> None:
    pass
```

❌ Don't mock when:

- The dependent function are basic calculations, has no side effect, and are efficient to run
  - Often these functions are data processing/transformation or metrics calculation procedures
- The function/method is to be tested by the unit tests

Example:

```python
from unittest.mock import MagicMock

# DO NOT mock the following since they are basic calculations

@mock.patch("metaflow_pipeline.compute_drivers")
@mock.patch("metaflow_pipeline.get_driver_min_and_max")
@mock.patch("metaflow_pipeline.create_loo_preds_df")
@mock.patch("metaflow_pipeline.compute_metrics_by_grouping")
def test_train_model_step(
    mock_compute_metrics: MagicMock,
    mock_create_loo_preds: MagicMock,
    mock_get_driver_min_max: MagicMock,
    mock_compute_drivers: MagicMock
) -> None:
    pass
```


## Group related unit tests under one class

✅ The recommended way

- Group test functions for the same target function under one class
- Omit the function name to be tested in the test method names.

Example:

```python
from typing import Any

class TestFoo:
    """Test function foo"""
    def shared_util(self) -> Any:
        ...

    def test_case_1(self) -> None:
        ...

    def test_case_2(self) -> None:
        ...

    def test_case_3(self) -> None:
        ...
```

❌ The undesirable way

Create a flat list of test functions with duplicate substring in the function names.

Example:

```python
def test_foo_case_1() -> None:
    ...

def test_foo_case_2() -> None:
    ...

def test_foo_case_3() -> None:
    ...

```

## Use pytest.parametrize to improve structure and code reuse

❌ The undesirable way

```python
  def test_process_data_error_foo() -> None:
      with pytest.raises(ValueError, match="Error: foo"):
          process_data("")

  def test_process_data_error_bar() -> None:
      with pytest.raises(ValueError, match="Error: bar"):
          process_data("invalid-input")

  def test_process_data_error_baz() -> None:
      with pytest.raises(ValueError, match="Error: baz"):
          process_data("incomplete@")

```

✅ The recommended way

```python
  @pytest.mark.parametrize(
      "input_value, expected_error_message",
      [
          ("", "Error: foo"),
          ("invalid-input", "Error: bar"),
          ("incomplete@", "Error: baz"),
      ],
  )
  def test_process_data_error_cases(input_value: str, expected_error_message: str):
      with pytest.raises(ValueError, match=expected_error_message):
          process_data(input_value)
```

## Reuse test data

- Try defining shared test data at the class level, to make tests more concise.
- To ensure immutability, define the shared data as class properties

❌ The undesirable way

```python
from typing import Dict, List, Any

class TestFoo:
    def test_case_1(self) -> None:
        test_data = {"key": "value", "items": [1, 2, 3]}
        # test logic...

    def test_case_2(self) -> None:
        # Same mock data duplicated again
        test_data = {"key": "value", "items": [1, 2, 3]}
        # test logic...
```

✅ The recommended way

```python
from typing import Dict, List, Any

class TestFoo:
    # Define shared test data as class property, to ensure immutability
    @property
    def test_data(self) -> Dict[str, Any]:
        return {"key": "value", "items": [1, 2, 3]}

    def test_case_1(self) -> None:
        # Reuse class attributes
        result = some_function(self.test_data)
        # test logic...

    def test_case_2(self) -> None:
        # Same shared data, no duplication
        result = some_function(self.test_data)
        # test logic...
```

## Share patch decorator at class level

When the same object need to be patched for all methods under the same class, avoid patching the object for each method.

Instead, patch it once at the class level.

❌ The undesirable way

```python
from unittest.mock import MagicMock

class TestFoo:
    @mock.patch("bar.baz")
    def test_case_1(self, mock_baz: MagicMock) -> None:
        ...

    @mock.patch("bar.baz")
    def test_case_2(self, mock_baz: MagicMock) -> None:
        ...
```

✅ The recommended way

```python
from unittest.mock import MagicMock

@mock.patch("bar.baz")
class TestFoo:
    def test_case_1(self, mock_baz: MagicMock) -> None:
        ...

    def test_case_2(self, mock_baz: MagicMock) -> None:
        ...
```

## Number(s) comparison

When scalers are compared

```python
# ❌ do not use math.close
assert math.isclose(result, expected)

# ✅ use pytest, which is more informative
assert result == pytest.approx(expected)
```

When arrays are compared:

```python
# ✅ use numpy utils
numpy.testing.assert_allclose(x_arr, y_arr)
```
