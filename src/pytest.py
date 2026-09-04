"""Lightweight pytest compatibility shim for offline/standalone execution.

Provides standard pytest constructs:
- @pytest.fixture
- pytest.raises(Exc, match=...)
- pytest.approx(val)
"""

from __future__ import annotations

import contextlib
import math
import re
from typing import Any, Callable


def fixture(*args: Any, **kwargs: Any) -> Callable:
    """Fixture decorator."""
    if len(args) == 1 and callable(args[0]):
        fn = args[0]
        fn._is_fixture = True
        return fn

    def decorator(fn: Callable) -> Callable:
        fn._is_fixture = True
        return fn

    return decorator


class raises:
    """Context manager asserting that an exception is raised."""

    def __init__(self, exc_type: type[BaseException], match: str | None = None) -> None:
        self.exc_type = exc_type
        self.match = match

    def __enter__(self) -> raises:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is None:
            raise AssertionError(
                f"DID NOT RAISE {self.exc_type.__name__}"
            )
        if not issubclass(exc_type, self.exc_type):
            return False
        if self.match is not None:
            msg = str(exc_val)
            if not re.search(self.match, msg):
                raise AssertionError(
                    f"Pattern '{self.match}' does not match '{msg}'"
                )
        return True


class approx:
    """Approximate numerical comparison."""

    def __init__(self, expected: float, rel: float = 1e-6, abs: float = 1e-12) -> None:
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, actual: Any) -> bool:
        if isinstance(actual, (int, float)):
            return math.isclose(actual, self.expected, rel_tol=self.rel, abs_tol=self.abs)
        return False

    def __repr__(self) -> str:
        return f"approx({self.expected} ± {self.rel})"


# pytest.mark dummy
class _Mark:
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: (lambda f: f)


mark = _Mark()

