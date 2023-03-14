import warnings, sys, os


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
    def function_warning_catcher(f, args, warning_verbosity):
        warnings.simplefilter(warning_verbosity, UserWarning)
        old_stdout = sys.stdout
        if warning_verbosity == 'ignore':
            sys.stdout = open(os.devnull, "w")
        else:
            sys.stdout = old_stdout

        # -- add in try loop such that warnings are reset to original even 
        # if errors occur in function
        try:
            out = FuncHelper.run_with_argument(f, args)
        except:
            warnings.simplefilter('default', UserWarning)
            sys.stdout = old_stdout
        
        warnings.simplefilter('default', UserWarning)
        sys.stdout = old_stdout

        return out
    
    @staticmethod
    def method_warning_catcher(f):
        def wrap_arguments(args):
            warnings.simplefilter(args.warning_verbosity, UserWarning)
            old_stdout = sys.stdout
            if args.warning_verbosity == 'ignore':
                sys.stdout = open(os.devnull, "w")
            else:
                sys.stdout = old_stdout
            
            # -- add in try loop such that warnings are reset to original even 
            # if errors occur in function
            try:
                f(args)
            except:
                warnings.simplefilter('default', UserWarning)
                sys.stdout = old_stdout
            
            warnings.simplefilter('default', UserWarning)
            sys.stdout = old_stdout

        return wrap_arguments
