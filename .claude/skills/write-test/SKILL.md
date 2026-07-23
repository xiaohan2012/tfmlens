---
name: write-test
description: Write unit tests according to the following requirement.
---

Write unit tests according to the following requirement.

#$ARGUMENTS

# Guidelines on writing unit tests

## What not to test — keep it to the point

- **Test behaviour, not implementation details.** Assert the observable result of a function, not how it computes it internally.
- **Don't test the language or the framework.** Checks like "is the returned list really a `list`?" are near-tautologies — low signal, drop them.
- **One concept per test.** Keep each test focused on a single behaviour instead of packing unrelated assertions into one.
- **Few high-signal tests beat exhaustive enumeration.** Cover representative behaviours and boundaries; collapse repetitive cases with `parametrize` rather than adding tests for their own sake.

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

**Rule:** Group tests under a class as soon as a target function/class has **two or more** tests. A target with a **single** test can stay a flat function — don't add a class for its own sake.

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

## Share setup with fixtures

Use pytest **fixtures** for shared setup (put cross-file fixtures in `conftest.py`).
Plain immutable constants can live as class attributes, but anything that has to
be *constructed* — objects, state — should come from a fixture. Each test then
gets a fresh instance, so tests stay isolated and don't leak state into one another.

❌ The undesirable way — rebuilding the same object in every test

```python
class TestFoo:
    def test_case_1(self) -> None:
        data = {"key": "value", "items": [1, 2, 3]}
        # test logic...

    def test_case_2(self) -> None:
        # Same data duplicated again
        data = {"key": "value", "items": [1, 2, 3]}
        # test logic...
```

✅ The recommended way — a fixture injected per test

```python
import pytest


@pytest.fixture
def data() -> dict:
    return {"key": "value", "items": [1, 2, 3]}


class TestFoo:
    def test_case_1(self, data: dict) -> None:
        result = some_function(data)
        # test logic...

    def test_case_2(self, data: dict) -> None:
        result = some_function(data)
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

When torch tensors are compared:

```python
import torch.testing

# ✅ more informative failures than torch.allclose
torch.testing.assert_close(x, y)
```
