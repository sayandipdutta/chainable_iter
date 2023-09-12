from __future__ import annotations
from abc import ABC, abstractmethod

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from functools import partial
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
from statistics import mean
from typing import (
    Any,
    Generic,
    Iterator,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    Self,
    TypeVar,
    cast,
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


class Sentinel(object):
    pass


NA = Sentinel()


class Chainable(ABC, Iterator[T]):
    @abstractmethod
    def map(self, func: Callable[[T], R]) -> Chainable[R]:
        ...

    @abstractmethod
    def filter(self, func: Callable[[T], bool], filter_false: bool = False) -> Self:
        ...

    @abstractmethod
    def accumulate(
        self, func: Callable[[T, T], T], *, initial: object | T = NA
    ) -> Self:
        ...

    @abstractmethod
    def append(self, *iterable: Iterable[T]) -> Self:
        ...

    @abstractmethod
    def prepend(self, *iterable: Iterable[T]) -> Self:
        ...

    @abstractmethod
    def compress(self, selectors: Iterable[bool]) -> Self:
        ...

    @abstractmethod
    def dropwhile(self, predicate: Callable[[T], bool]) -> Self:
        ...

    @abstractmethod
    def takewhile(self, predicate: Callable[[T], bool]) -> Self:
        ...

    @abstractmethod
    @overload
    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["strict", "shortest"] = "shortest",
        other_first: Literal[False] = False,
    ) -> Chainable[tuple[T, R]]:
        ...

    @abstractmethod
    @overload
    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["strict", "shortest"] = "shortest",
        other_first: Literal[True] = True,
    ) -> Chainable[tuple[R, T]]:
        ...

    @abstractmethod
    @overload
    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["longest"],
        fillvalue: K,
        other_first: Literal[False] = False,
    ) -> Chainable[tuple[T | K, R | K]]:
        ...

    @abstractmethod
    @overload
    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["longest"],
        fillvalue: K,
        other_first: Literal[True] = True,
    ) -> Chainable[tuple[R | K, T | K]]:
        ...

    @abstractmethod
    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["strict", "shortest", "longest"] = "shortest",
        fillvalue: Optional[K] = None,
        other_first: bool = False,
    ) -> (
        Chainable[tuple[T, R]]
        | Chainable[tuple[R, T]]
        | Chainable[tuple[R | K, T | K]]
        | Chainable[tuple[T | K, R | K]]
    ):
        ...

    @abstractmethod
    def zip(
        self,
        iterable: Iterable,
        *its: Iterable,
        policy: Literal["strict", "shortest", "longest"] = "shortest",
        fillvalue: Any = None,
    ) -> NestedIterator[Any]:
        ...

    @abstractmethod
    def filter_index(
        self, predicate: Callable[[T], bool], filter_false: bool = False
    ) -> Chainable[int]:
        """
        Chainable(range(5, 10)).filter_index(lambda x: x % 3 == 0)
        -> 1 4
        """
        ...

    # TODO: make Nested
    @abstractmethod
    def sliding_window(self, n: int = 2, step: int = 0) -> Chainable[tuple[T, ...]]:
        ...

    # TODO: make Nested
    @abstractmethod
    @overload
    def batched(
        self,
        n: int = 1,
        *,
        policy: Literal["fill"],
        fillvalue: K,
    ) -> Chainable[tuple[T | K, ...]]:
        ...

    # TODO: make Nested
    @abstractmethod
    @overload
    def batched(
        self,
        n: int = 1,
        *,
        policy: Literal["strict", "ignore"] = "strict",
    ) -> Chainable[tuple[T, ...]]:
        ...

    # TODO: make Nested
    @abstractmethod
    def batched(
        self,
        n: int = 1,
        *,
        policy: Literal["fill", "strict", "ignore"] = "ignore",
        fillvalue: K = None,
    ) -> Chainable[tuple[T, ...]] | Chainable[tuple[T | K, ...]]:
        ...

    @abstractmethod
    def islice(self, start: int | None, stop: int | None, step: int | None = 1) -> Self:
        ...

    @abstractmethod
    def skip(self, n: int | None = None) -> Self:
        ...

    @abstractmethod
    def getitems(self, indices: Iterable[int]) -> Chainable[T]:
        ...

    @abstractmethod
    def starmap(self, func: Callable[P, R], *args: Iterable) -> Chainable[R]:
        ...

    # TODO: Make Nested
    @abstractmethod
    def split_at(self, *indices) -> Chainable[tuple[T, ...]]:
        ...

    # TODO: Make Nested
    @abstractmethod
    def enumerate(self, start: int = 0) -> Chainable[tuple[int, T]]:
        ...

    @abstractmethod
    def make_consumable(self) -> Consumable[T]:
        ...

    @abstractmethod
    def make_collapsible(self) -> Collapsible[T]:
        ...


class ChainableIterator(Chainable[T]):
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
            return self
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
            return self
        self._iterable = accumulate(self, func, initial=cast(T, initial))
        return self

    def append(self, *iterable: Iterable[T]) -> Self:
        """
        ChainableIterator(range(5)).append(range(5, 10))
        -> 0 1 2 3 4 5 6 7 8 9
        """
        self._iterable = chain(self._iterable, *iterable)
        return self

    def prepend(self, *iterable: Iterable[T]) -> Self:
        """
        ChainableIterator(range(5)).append(range(5, 10))
        -> 5 6 7 8 9 0 1 2 3 4
        """
        self._iterable = iter(chain(*iterable, self._iterable))
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

    # TODO: Try to retain type information inside the tuple
    def zip2(
        self,
        iterable: Iterable[R],
        *,
        policy: Literal["strict", "shortest", "longest"] = "shortest",
        fillvalue: Optional[K] = None,
        other_first: bool = False,
    ) -> NestedIterator[T | R] | NestedIterator[T | R | K]:
        if policy == "strict":
            if other_first:
                return NestedIterator(zip(iterable, self, strict=True))
            return NestedIterator(zip(self, iterable, strict=True))
        elif policy == "shortest":
            if other_first:
                return NestedIterator(zip(iterable, self))
            return NestedIterator(zip(self, iterable))
        elif policy == "longest":
            if other_first:
                return NestedIterator(
                    zip_longest(iterable, self, fillvalue=cast(K, fillvalue))
                )
            return NestedIterator(
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
    ) -> NestedIterator[Any]:
        """
        ChainableIterator(range(5)).zip(range(1, 6), range(10, 15))
        -> (0, 1, 10) (1, 2, 11) (2, 3, 12) (3, 4, 13) (4, 5, 14)
        """
        if policy == "longest":
            return NestedIterator(
                zip_longest(self, iterable, *its, fillvalue=fillvalue)
            )
        elif policy == "shortest":
            return NestedIterator(zip(self, iterable, *its))
        elif policy == "strict":
            return NestedIterator(zip(self, iterable, *its, strict=True))
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

    def sliding_window(self, n: int = 2, step: int = 1) -> NestedIterator[T]:
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
            step_counter = 0
            for item in self:
                window.append(item)
                step_counter += 1
                if step_counter == step:
                    yield tuple(window)
                    step_counter = 0

        return NestedIterator(_helper())

    @overload
    def batched(
        self,
        n: int = 1,
        *,
        policy: Literal["fill"],
        fillvalue: K,
    ) -> NestedIterator[T | K]:
        ...

    @overload
    def batched(
        self,
        n: int = 1,
        *,
        policy: Literal["strict", "ignore"] = "strict",
    ) -> NestedIterator[T]:
        ...

    def batched(
        self,
        n: int = 1,
        *,
        policy: Literal["fill", "strict", "ignore"] = "ignore",
        fillvalue: K = None,
    ) -> NestedIterator[T | K] | NestedIterator[T]:
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

        return NestedIterator(_helper())

    def islice(
        self, *, start: int | None, stop: int | None, step: int | None = 1
    ) -> Self:
        """
        ChainableIterator(range(50)).islice(start=0, stop=10, step=10)
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
            for i, item in enumerate(self._iterable):
                if i == ix:
                    yield item
                    temp = next(it, NA)
                    if temp is NA:
                        break
                    ix = cast(int, temp)

        return ChainableIterator(_helper())

    def starmap(self, func: Callable[P, R], *args: Iterable) -> ChainableIterator[R]:
        if len(args) == 0:
            raise ValueError()
        return ChainableIterator(starmap(func, self.zip(*args)))

    def split_at(self, *indices: int) -> NestedIterator[T]:
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
        return NestedIterator[T](iterator)

    def enumerate(self, start: int = 0) -> ChainableIterator[tuple[int, T]]:
        return ChainableIterator(enumerate(self, start=start))

    def make_consumable(self) -> Consumable:
        return Consumable(self)

    def make_collapsible(self) -> Collapsible:
        return Collapsible(self)


class NestedIterator(ChainableIterator[Sequence[T]]):
    def __init__(self, iterable: Iterable[Sequence[T]]) -> None:
        self._iterable: Iterator[Sequence[T]] = iter(iterable)

    def __next__(self) -> Sequence[T]:
        val = next(self._iterable)
        if not isinstance(val, Sequence):
            raise TypeError("Each item of NestedIterator must be a sequence")
        return val

    def starmap(self, func: Callable[P, R], *args: Iterable) -> ChainableIterator[R]:
        if len(args) == 0:
            return ChainableIterator(starmap(func, self._iterable))
        return ChainableIterator(starmap(func, self.zip(*args)))

    def reverse_each(self) -> Self:
        self._iterable = (cast(Sequence[T], reversed(item)) for item in self)
        return self

    def first_of_each(self) -> ChainableIterator[T]:
        return ChainableIterator((item[0] for item in self))

    def last_of_each(self) -> Chainable[T]:
        return ChainableIterator((item[-1] for item in self))

    def transpose(self) -> Self:
        its = (iter(item) for item in self)
        transposed = zip(*its)
        self._iterable = transposed
        return self


class Collapsible(Generic[T]):
    def __init__(self, iterable: Iterator[T]) -> None:
        self._iterable = iterable

    def collapse(self, function: Callable[[Iterable[T]], R]) -> R:
        return function(self._iterable)

    def __rshift__(self, function: Callable[[Iterable[T]], R]) -> R:
        return function(self._iterable)


class Consumable(Generic[T]):
    def __init__(self, iterable: Iterator[T]) -> None:
        self._iterable = iterable

    def consume(self, function: Callable[[T], Any]) -> None:
        for item in self._iterable:
            function(item)

    # @abstractmethod
    # def __rshift__(self, func: Callable[[T], Any]) -> None:
    #     ...


if __name__ == "__main__":
    # Markov Chain Monte Carlo:
    # probability of getting a mean higher than 80
    # out of 10 random variables in range 1 to 100
    ITERATION = 100_000

    values: Iterator[int] = (randrange(1, 100) for _ in range(ITERATION))
    iter_values = ChainableIterator(values)
    prob = (
        iter_values.batched(10).map(mean).map(lambda x: x > 1).make_collapsible()
    ) >> partial(sum, start=15)
    print(prob)
    values = (randrange(1, 100) for _ in range(ITERATION))
    iter_values = ChainableIterator(values)
    total = iter_values.batched(10).map(mean).map(bool).make_collapsible() >> sum
    print(total)
    iter_range = ChainableIterator(range(100))
    iter_range.islice(start=20, stop=None, step=25).make_consumable().consume(print)
    iter_range = ChainableIterator(range(15))
    iter_range.split_at(3, 5, 9, 12).make_consumable().consume(print)
    iter_range.batched(5).transpose().make_consumable().consume(print)

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
        ChainableIterator(months)
        .enumerate(start=1)
        .map(reversed)
        .make_collapsible()
        .collapse(dict)
    )
    print(month_dict)
