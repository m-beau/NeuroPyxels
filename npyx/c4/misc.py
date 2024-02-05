import functools


def require_advanced_deps(*deps):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing_deps = [dep for dep in deps if not globals().get(dep)]
            if missing_deps:
                raise ImportError(
                    f"This function ('{func.__name__}') requires additional dependencies to run: {', '.join(missing_deps)}. \n"
                    "Please re-install the 'c4' version of npyx or manually install the missing dependencies to run it."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
