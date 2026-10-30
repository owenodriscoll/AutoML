import gc
import sys
import warnings

import pytest

from automl.utils.function_helper import FuncHelper


def add(a, b=10):
    return a + b


def noisy(value):
    print('noise')
    warnings.warn('careful', UserWarning)
    return value


def fail(value):
    raise ValueError(f'failed on {value}')


# --------------------------------------- run_with_argument ---------------------------------------
@pytest.mark.parametrize('args, expected', [(1, 11), (1.5, 11.5), ([1, 2], 3), ({'a': 1, 'b': 5}, 6)])
def test_run_with_argument_unpacks_by_type(args, expected):
    assert FuncHelper.run_with_argument(add, args) == expected


def test_run_with_argument_none_does_not_call():
    assert FuncHelper.run_with_argument(fail, None) is None


def test_run_with_argument_unsupported_type_reports_and_returns_none(capsys):
    assert FuncHelper.run_with_argument(add, 'text') is None
    assert 'Input argument not of type' in capsys.readouterr().out


# ------------------------------------ function_warning_catcher ------------------------------------
def test_function_catcher_returns_output():
    assert FuncHelper.function_warning_catcher(add, [1, 2], 'ignore') == 3


def test_function_catcher_silences_stdout_and_warnings_on_ignore(capsys):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert FuncHelper.function_warning_catcher(noisy, [1], 'ignore') == 1
    assert capsys.readouterr().out == ''
    assert not [warning for warning in caught if warning.category is UserWarning]


@pytest.mark.xfail(reason="BUG: the os.devnull handle opened to silence stdout is never closed, leaking one per call")
def test_function_catcher_closes_devnull():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        FuncHelper.function_warning_catcher(add, [1, 2], 'ignore')
        gc.collect()
    assert not [warning for warning in caught if warning.category is ResourceWarning]


def test_function_catcher_passes_stdout_and_warnings_otherwise(capsys):
    with pytest.warns(UserWarning, match='careful'):
        FuncHelper.function_warning_catcher(noisy, [1], 'always')
    assert 'noise' in capsys.readouterr().out


def test_function_catcher_restores_stdout_and_chains_exceptions():
    stdout = sys.stdout
    with pytest.raises(RuntimeError) as error:
        FuncHelper.function_warning_catcher(fail, [1], 'ignore')
    assert isinstance(error.value.__cause__, ValueError)
    assert sys.stdout is stdout


def interrupt(value):
    raise KeyboardInterrupt


@pytest.mark.xfail(reason="BUG: only Exception is caught, so an interrupted fit leaves sys.stdout on os.devnull, "
                          "which silences all further output of a notebook session")
def test_function_catcher_restores_stdout_after_keyboard_interrupt():
    stdout = sys.stdout
    try:
        with pytest.raises(KeyboardInterrupt):
            FuncHelper.function_warning_catcher(interrupt, [1], 'ignore')
        assert sys.stdout is stdout
    finally:
        sys.stdout = stdout


# ------------------------------------- method_warning_catcher -------------------------------------
class Worker:
    def __init__(self, warning_verbosity):
        self.warning_verbosity = warning_verbosity

    @FuncHelper.method_warning_catcher()
    def succeed(self, value):
        """docstring"""
        warnings.warn('careful', UserWarning)
        return value

    @FuncHelper.method_warning_catcher()
    def crash(self):
        raise ValueError('crashed')

    @FuncHelper.method_warning_catcher(warning_verbosity='always')
    def crash_loudly(self):
        raise ValueError('crashed')


def test_method_catcher_preserves_signature_and_output():
    worker = Worker('ignore')
    assert worker.succeed(5) == 5
    assert worker.succeed.__name__ == 'succeed' and worker.succeed.__doc__ == 'docstring'


def test_method_catcher_respects_instance_verbosity():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Worker('ignore').succeed(1)
    assert not caught

    with pytest.warns(UserWarning, match='careful'):
        Worker('always').succeed(1)


def test_method_catcher_converts_exception_to_warning():
    with pytest.warns(UserWarning, match='crashed'):
        assert Worker('always').crash() is None


def test_method_catcher_explicit_verbosity_overrides_instance():
    with pytest.warns(UserWarning, match='crashed'):
        assert Worker('ignore').crash_loudly() is None


def test_method_catcher_escalates_with_error_verbosity():
    with pytest.raises(UserWarning, match='crashed'):
        Worker('error').crash()


def test_method_catcher_restores_warning_filters():
    filters = list(warnings.filters)
    Worker('ignore').crash()
    assert warnings.filters == filters
