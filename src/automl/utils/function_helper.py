import functools, warnings, sys, os


class OptimisationFailureWarning(RuntimeWarning):
    """
    Issued when a trial or model fails during hyperparameter optimisation, which continues with what remains.
    Not a UserWarning, so that failures are visible regardless of `warning_verbosity`.
    """


class FuncHelper:
    @staticmethod
    def run_with_argument(f, args):

        if isinstance(args, (int, float)):
            return f(args)
        if isinstance(args, list):
            return f(*args)
        if isinstance(args, dict):
            return f(**args)
        elif isinstance(args, type(None)):
            return None
        else:
            print("Input argument not of type: int, float or dict")

    @staticmethod
    def function_warning_catcher(f, args, new_warning_verbosity, old_warning_verbosity = 'default', new_std_error = None):
        warnings.simplefilter(new_warning_verbosity, UserWarning)
        old_stdout = sys.stdout
        devnull = open(os.devnull, "w") if new_warning_verbosity == 'ignore' else None
        if devnull is not None:
            sys.stdout = devnull
        else:
            sys.stdout = old_stdout if new_std_error is None else new_std_error

        # -- add in try loop such that warnings and stdout are reset to original
        # even if the function is interrupted or errors occur in function
        try:
            return FuncHelper.run_with_argument(f, args)
        except Exception as e:
            raise RuntimeError("Error in function warning catcher") from e
        finally:
            warnings.simplefilter(old_warning_verbosity, UserWarning)
            sys.stdout = old_stdout
            if devnull is not None:
                devnull.close()

    @staticmethod
    def method_warning_catcher(warning_verbosity=None):
        def decorator(f):

            @functools.wraps(f)
            def wrapper(self, *args, **kwargs):

                verbosity = (
                    warning_verbosity
                    if warning_verbosity is not None
                    else self.warning_verbosity
                )

                with warnings.catch_warnings():
                    warnings.simplefilter(verbosity, UserWarning)
                    try:
                        return f(self, *args, **kwargs)

                    except Exception as e:
                        warnings.warn(
                            str(e),
                            OptimisationFailureWarning,
                            stacklevel=2
                        )
                        return None

            return wrapper

        return decorator
