import functools
import importlib
import os
import sys
from typing import Callable

import_error_info_string = (
    "Please re-install the 'c4' version of npyx by running 'pip install npyx[c4]' \n"
    "Alternatively, manually install the missing dependencies. \n"
    "For more information, please refer to: https://github.com/m-beau/NeuroPyxels#%EF%B8%8F-installation. \n"
)


def fix_autoreload(decorator):
    """
    Fix autoreload for decorators so decorated functions properly autoreload
    """
    if "autoreload" not in sys.modules:
        return decorator

    def autoreload_fixed_decorator(func: Callable[[any], any]):

        module = sys.modules[func.__module__]
        mdate = os.path.getmtime(module.__file__)
        decorated = decorator(func)

        @functools.wraps(func)
        def autoreload_func(*args, **kwargs):
            nonlocal mdate, decorated

            _mdate = os.path.getmtime(module.__file__)
            if _mdate != mdate:
                mdate = _mdate
                decorated = importlib.reload(module).__dict__[func.__name__].__wrapped__

            return decorated(*args, **kwargs)

        return autoreload_func

    return autoreload_fixed_decorator


@fix_autoreload
def require_advanced_deps(*deps):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing_deps = [dep for dep in deps if importlib.util.find_spec(dep) is None]
            if missing_deps:
                raise ImportError(
                    f"This function ('{func.__name__}') requires additional dependencies to run: {', '.join(missing_deps)}. \n"
                    f"{import_error_info_string}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


class MissingTorch:
    def __init__(self) -> None:
        raise ImportError(
            f"This class requires the 'torch' package to be initialised. \n" f"{import_error_info_string}"
        )
