from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from functools import partial, wraps
from itertools import (
    accumulate,
    chain,
    compress,
    dropwhile,
    filterfalse,
    islice,
    starmap,
    takewhile,
    zip_longest,
)
from operator import neg, sub
from random import randrange
from typing import (
    Any,
    Concatenate,
    Iterator,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    Self,
    TypeVar,
    cast,
    no_type_check,
    overload,
)


class SupportsAdd(Protocol):
    def __add__(self, other):
        ...


class SupportsSum(Protocol):
    def __add__(self, other):
        ...

    def __radd__(self, other):
        ...


T = TypeVar("T")
R = TypeVar("R")
K = TypeVar("K")
P = ParamSpec("P")
A = TypeVar("A", bound=SupportsAdd)
S = TypeVar("S", bound=SupportsSum)


class Sentinel(object):
    pass


NA = Sentinel()


def for_each(
    chainable: ChainableIterator[T],
    func: Callable[Concatenate[T, P], Any],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    for item in chainable:
        func(item, *args, **kwargs)


@overload
def sumall(
    iterable: Sentinel, /, start: A | Literal[0]
) -> Callable[[ChainableIterator[A]], A]:
    ...


@overload
def sumall(iterable: ChainableIterator[S]) -> S | Literal[0]:
    ...


def sumall(
    iterable: ChainableIterator[S] | Sentinel = NA, /, start: A | Literal[0] = 0
) -> Callable[[ChainableIterator[A]], A] | S | Literal[0]:
    @wraps(sumall)
    def wrapper(iterable: ChainableIterator[A]) -> A:
        return sum(iterable, start=cast(A, start))

    if iterable is NA:
        return wrapper
    assert isinstance(iterable, ChainableIterator)
    return sum(iterable)


def mean(iterable: Iterable[int | float]) -> float:
    total = 0
    count = 1
    for count, item in enumerate(iterable, start=1):
        total += item
    return total / count


class ChainableIterator(Iterator[T]):
    def __init__(self, iterable: Iterable[T]) -> None:
        self._iterable: Iterator[T] = iter(iterable)

    def __iter__(self) -> Iterator[T]:
        return ChainableIterator(self._iterable)

    def __next__(self) -> T:
        return next(self._iterable)

    def map(self, func: Callable[[T], R]) -> ChainableIterator[R]:
        """
        ChainableIterator(range(5)).map(lambda x: x**2)
        -> 0 1 4 9 16
        """
        return ChainableIterator(map(func, self))

    def filter(self, func: Callable[[T], bool], filter_false: bool = False) -> Self:
        """
        ChainableIterator(range(5)).filter(lambda x: x%2 == 0)
        -> 0 2 4
        """
        if filter_false:
            self._iterable = filterfalse(func, self)
        self._iterable = filter(func, self)
        return self

    def accumulate(
        self, func: Callable[[T, T], T], *, initial: object | T = NA
    ) -> Self:
        """
        ChainableIterator(range(5)).accumulate(add)
        -> 0 1 3 6 10
        """
        if initial is NA:
            self._iterable = accumulate(self, func)
        self._iterable = accumulate(self, func, initial=cast(T, NA))
        return self

    def append(self, *iterable: Iterable[T]) -> Self:
        """
        ChainableIterator(range(5)).append(range(5, 10))
        -> 0 1 2 3 4 5 6 7 8 9
        """
        self._iterable = chain(self, *iterable)
        return self

    def prepend(self, *iterable: Iterable[T]) -> Self:
        """
        ChainableIterator(range(5)).append(range(5, 10))
        -> 5 6 7 8 9 0 1 2 3 4
        """
        self._iterable = chain(*iterable, self)
        return self

    def compress(self, selectors: Iterable[bool]) -> Self:
        """
        ChainableIterator(range(5)).compress([1, 0, 0, 1, 0])
        -> 0 3
        """
        self._iterable = compress(self, selectors)
        return self

    def dropwhile(self, predicate: Callable[[T], bool]) -> Self:
        """
        ChainableIterator(range(5)).dropwhile(lambda x: x < 3)
        -> 3 4
        """
        self._iterable = dropwhile(predicate, self)
        return self

    def takewhile(self, predicate: Callable[[T], bool]) -> Self:
        """
        ChainableIterator(range(5)).takewhile(lambda x: x < 3)
        -> 0 1 2
        """
        self._iterable = takewhile(predicate, self)
        return self

    @overload
    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["strict", "shortest"] = "shortest",
        other_first: Literal[False] = False,
    ) -> ChainableIterator[tuple[T, R]]:
        ...

    @overload
    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["strict", "shortest"] = "shortest",
        other_first: Literal[True] = True,
    ) -> ChainableIterator[tuple[R, T]]:
        ...

    @overload
    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["longest"],
        fillvalue: K,
        other_first: Literal[False] = False,
    ) -> ChainableIterator[tuple[T | K, R | K]]:
        ...

    @overload
    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["longest"],
        fillvalue: K,
        other_first: Literal[True] = True,
    ) -> ChainableIterator[tuple[R | K, T | K]]:
        ...

    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["strict", "shortest", "longest"] = "shortest",
        fillvalue: Optional[K] = None,
        other_first: bool = False,
    ) -> (
        ChainableIterator[tuple[T, R]]
        | ChainableIterator[tuple[R, T]]
        | ChainableIterator[tuple[R | K, T | K]]
        | ChainableIterator[tuple[T | K, R | K]]
    ):
        if policy == "strict":
            if other_first:
                return ChainableIterator(zip(iterable, self, strict=True))
            return ChainableIterator(zip(self, iterable, strict=True))
        elif policy == "shortest":
            if other_first:
                return ChainableIterator(zip(iterable, self))
            return ChainableIterator(zip(self, iterable))
        elif policy == "longest":
            if other_first:
                return ChainableIterator(
                    zip_longest(iterable, self, fillvalue=cast(K, fillvalue))
                )
            return ChainableIterator(
                zip_longest(self, iterable, fillvalue=cast(K, fillvalue))
            )
        raise NotImplementedError(
            "Available policy values are: 'strict', 'shortest', 'longest'"
        )

    def zip(
        self,
        iterable: Iterable,
        *its: Iterable,
        policy: Literal["strict", "shortest", "longest"] = "shortest",
        fillvalue: Any = None,
    ) -> ChainableIterator[tuple[Any, ...]]:
        """
        ChainableIterator(range(5)).zip(range(1, 6), range(10, 15))
        -> (0, 1, 10) (1, 2, 11) (2, 3, 12) (3, 4, 13) (4, 5, 14)
        """
        if policy == "longest":
            return ChainableIterator(
                zip_longest(self, iterable, *its, fillvalue=fillvalue)
            )
        elif policy == "shortest":
            return ChainableIterator(zip(self, iterable, *its))
        elif policy == "strict":
            return ChainableIterator(zip(self, iterable, *its, strict=True))
        else:
            raise NotImplementedError()

    def filter_index(
        self, predicate: Callable[[T], bool], filter_false: bool = False
    ) -> ChainableIterator[int]:
        """
        ChainableIterator(range(5, 10)).filter_index(lambda x: x % 3 == 0)
        -> 1 4
        """

        def _helper(filter_false: bool) -> Iterator[int]:
            for i, item in enumerate(self):
                cond = predicate(item)
                if cond == filter_false:
                    yield i

        return ChainableIterator(_helper(filter_false))

    def sliding_window(
        self, n: int = 2, step: int = 0
    ) -> ChainableIterator[tuple[T, ...]]:
        """
        ChainableIterator(range(5)).sliding_window(2)
        -> (5, 6) (6, 7) (7, 8) (8, 9)
        ChainableIterator(range(5)).sliding_window(3, step=2)
        -> (5, 6, 7) (7, 8, 9)
        """
        if n < 2:
            raise ValueError()

        def _helper() -> Iterator[tuple[T, ...]]:
            window = deque(islice(self, n), maxlen=n)
            if len(window) == n:
                yield tuple(window)
            self.skip(step)
            for item in self:
                window.append(item)
                yield tuple(window)
                self.skip(step)

        return ChainableIterator(_helper())

    @overload
    def batched(
        self,
        n: int = 1,
        *,
        policy: Literal["fill"],
        fillvalue: K,
    ) -> ChainableIterator[tuple[T | K, ...]]:
        ...

    @overload
    def batched(
        self,
        n: int = 1,
        *,
        policy: Literal["strict", "ignore"] = "strict",
    ) -> ChainableIterator[tuple[T, ...]]:
        ...

    def batched(
        self,
        n: int = 1,
        *,
        policy: Literal["fill", "strict", "ignore"] = "ignore",
        fillvalue: K = None,
    ) -> ChainableIterator[tuple[T, ...]] | ChainableIterator[tuple[T | K, ...]]:
        """
        ChainableIterator(range(5)).batched(2)
        -> (5, 6) (7, 8) (9,)
        """

        def _helper() -> Iterator[tuple[T | K, ...]]:
            if n < 1:
                raise ValueError()
            while batch := tuple(islice(self, n)):
                if 0 < (length := len(batch)) < n:
                    if policy == "fill":
                        batch += (fillvalue,) * (n - length)
                    elif policy == "strict":
                        raise ValueError("Uneven size")
                    elif policy == "ignore":
                        pass
                    else:
                        raise NotImplementedError()
                yield batch

        return ChainableIterator(_helper())

    def islice(self, start: int | None, stop: int | None, step: int | None = 1) -> Self:
        """
        ChainableIterator(range(50)).islice(0, 10, 10)
        -> 0 10 20 30 40
        """
        if start is None:
            raise ValueError("At least one of start and stop must be not None.")
        self._iterable = islice(self, start, stop, step)
        return self

    def skip(self, n: int | None = None) -> Self:
        """
        ChainableIterator(range(50)).skip(45)
        -> 45 46 47 48 49
        """
        self._iterable = islice(self, n, None)
        return self

    def getitems(self, indices: Iterable[int]) -> ChainableIterator[T]:
        """
        ChainableIterator(range(10, 20)).take([2, 5])
        -> 12 15
        """

        def _helper() -> Iterator[T]:
            it = iter(indices)
            ix = next(it)
            for i, item in enumerate(self):
                if i == ix:
                    yield item
                    temp = next(it, NA)
                    if temp is NA:
                        break
                    ix = cast(int, temp)

        self._iterable = _helper()
        return self

    def starmap(self, func: Callable[P, R], *args: Iterable) -> ChainableIterator[R]:
        if len(args) == 0:
            return ChainableIterator(
                starmap(func, cast(ChainableIterator[Iterable[T]], self))
            )
        return ChainableIterator(starmap(func, self.zip(*args)))

    @no_type_check
    def split_at(self, *indices) -> ChainableIterator[tuple[T, ...]]:
        assert len(indices) > 0, "No index provided"
        assert sorted(indices) == list(
            indices
        ), "indices must be strictly increasing sequence"
        iter_ind = ChainableIterator(indices)
        iterator = (
            iter_ind.prepend([0])
            .sliding_window(2)
            .starmap(sub)
            .map(neg)
            .map(partial(islice, self))
            .append([islice(self, None)])
            .map(tuple)
        )
        return iterator

    def enumerate(self, start: int = 0) -> ChainableIterator[tuple[int, T]]:
        return ChainableIterator(enumerate(self, start=start))

    def transpose(self) -> ChainableIterator[tuple[T, ...]]:
        its = (iter(item) for item in cast(ChainableIterator[Iterable[T]], self))
        transposed = zip(*its)
        return ChainableIterator(transposed)

    def __gt__(self, func: Callable[[ChainableIterator[Any]], R]) -> R:
        return func(self)

    def consume_with(self, func: Callable[[T], Any]) -> None:
        r"""For each item, apply the given function.

        NOTE: This collapses the Iterable and Chainable status.
        """
        for item in self:
            func(item)

    def feed_to(self, func: Callable[[Any], R]) -> R:
        return func(self)


if __name__ == "__main__":
    # Markov Chain Monte Carlo:
    # probability of getting a mean higher than 80
    # out of 10 random variables in range 1 to 100
    ITERATION = 100_000

    values: Iterator[int] = (randrange(1, 100) for _ in range(ITERATION))
    iter_values = ChainableIterator(values)
    prob = iter_values.batched(10).map(mean).map(lambda x: x > 1) > partial(
        sum, start=15
    )
    print(prob)
    values = (randrange(1, 100) for _ in range(ITERATION))
    iter_values = ChainableIterator(values)
    total = iter_values.batched(10).map(mean).map(bool) > sum
    print(total)
    iter_range = ChainableIterator(range(100))
    for_each(iter_range.islice(20, None, 25), print)
    iter_range = ChainableIterator(range(15))
    for_each(iter_range.split_at(3, 5, 9, 12), print)
    iter_range = ChainableIterator(range(20))
    for_each(iter_range.batched(5).transpose(), print)

    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month_dict = (
        ChainableIterator(months).enumerate(start=1).map(reversed).feed_to(dict)
    )
    print(month_dict)
