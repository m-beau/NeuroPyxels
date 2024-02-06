import functools


def require_advanced_deps(*deps):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing_deps = [dep for dep in deps if not globals().get(dep)]
            if missing_deps:
                raise ImportError(
                    f"This function ('{func.__name__}') requires additional dependencies to run: {', '.join(missing_deps)}. \n"
                    "Please re-install the 'c4' version of npyx by running 'pip install npyx[c4]' \n"
                    "Alternatively, manually install the missing dependencies. \n"
                    "For more information, please refer to: https://github.com/m-beau/NeuroPyxels#%EF%B8%8F-installation. \n"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
