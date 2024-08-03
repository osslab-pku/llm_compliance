from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Tuple,
    TypeVar,
    Set,
    Callable,
)


T = TypeVar("T")


def chunks(lst: Iterable[T], n: int) -> Generator[List[T], None, None]:
    lst = lst[::-1]
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def merge_dict(
    left: Dict, right: Dict, fn: Callable[[Any, Any], Any] = lambda x, y: x + y
):
    for k, v in right.items():
        left[k] = fn(left.get(k, 0), v)
    return left
