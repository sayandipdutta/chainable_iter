from operator import add
import pytest

from chainable import ChainableIterator, NestedIterator


@pytest.fixture
def short_chainable_iterator():
    return ChainableIterator(range(5))


@pytest.fixture
def long_chainable_iterator():
    return ChainableIterator(range(100))


@pytest.fixture
def random_chainable_iterator():
    return ChainableIterator([3, 2, 5, 6, 9])


@pytest.fixture
def nested_iterator():
    return NestedIterator([(start, end) for start, end in zip([2] * 5, range(5))])


@pytest.mark.chainable
def test_map(short_chainable_iterator):
    result = list(short_chainable_iterator.map(lambda x: x**2))
    assert result == [0, 1, 4, 9, 16]


@pytest.mark.chainable
def test_filter(short_chainable_iterator):
    result = list(short_chainable_iterator.filter(lambda x: x % 2 == 0))
    assert result == [0, 2, 4]


@pytest.mark.chainable
def test_accumulate(short_chainable_iterator):
    result = list(short_chainable_iterator.accumulate(add))
    assert result == [0, 1, 3, 6, 10]


@pytest.mark.chainable
def test_accumulate_initial(short_chainable_iterator):
    result = list(short_chainable_iterator.accumulate(add, initial=5))
    assert result == [5, 5, 6, 8, 11, 15]


@pytest.mark.chainable
def test_append(short_chainable_iterator):
    result = list(short_chainable_iterator.append([5, 6]))
    assert result == [0, 1, 2, 3, 4, 5, 6]


@pytest.mark.chainable
def test_prepend(short_chainable_iterator):
    result = list(short_chainable_iterator.prepend([-2, -1]))
    assert result == [-2, -1, 0, 1, 2, 3, 4]


@pytest.mark.chainable
def test_compress(short_chainable_iterator):
    result = list(short_chainable_iterator.compress([0, 1, 1, 0, 0]))
    assert result == [1, 2]


@pytest.mark.chainable
def test_dropwhile(short_chainable_iterator):
    result = list(short_chainable_iterator.dropwhile(lambda x: x < 2))
    assert result == [2, 3, 4]


@pytest.mark.chainable
def test_takewhile(short_chainable_iterator):
    result = list(short_chainable_iterator.takewhile(lambda x: x < 2))
    assert result == [0, 1]


@pytest.mark.chainable
def test_zip2(short_chainable_iterator):
    result = list(short_chainable_iterator.zip2([0, 1, 1, 0, 0]))
    assert result == [(0, 0), (1, 1), (2, 1), (3, 0), (4, 0)]


@pytest.mark.chainable
def test_zip2_shortest(short_chainable_iterator):
    result = list(short_chainable_iterator.zip2([0, 1, 1, 0, 0], policy="strict"))
    assert result == [(0, 0), (1, 1), (2, 1), (3, 0), (4, 0)]


@pytest.mark.chainable
def test_zip2_longest(short_chainable_iterator):
    result = list(short_chainable_iterator.zip2([0, 1, 1, 0, 0, 9], policy="longest"))
    assert result == [(0, 0), (1, 1), (2, 1), (3, 0), (4, 0), (None, 9)]


@pytest.mark.chainable
def test_zip2_strict(short_chainable_iterator):
    with pytest.raises(ValueError) as excinfo:
        list(short_chainable_iterator.zip2([0, 1, 1, 0], policy="strict"))
    assert excinfo.value.args[0] == "zip() argument 2 is shorter than argument 1"


@pytest.mark.chainable
def test_zip_strict(short_chainable_iterator):
    with pytest.raises(ValueError):
        list(
            short_chainable_iterator.zip([0, 1, 1, 0], [4, 3, 2, 1, 0], policy="strict")
        )


@pytest.mark.chainable
def test_zip_shortest(short_chainable_iterator):
    result = list(
        short_chainable_iterator.zip([0, 1, 1, 0, 0], [4, 3], policy="shortest")
    )
    assert result == [(0, 0, 4), (1, 1, 3)]


@pytest.mark.chainable
def test_zip_longest(short_chainable_iterator):
    result = list(
        short_chainable_iterator.zip(
            [0, 1, 1, 0, 0, -1], [4, 3, 2, 1, 0], policy="longest"
        )
    )
    assert result == [
        (0, 0, 4),
        (1, 1, 3),
        (2, 1, 2),
        (3, 0, 1),
        (4, 0, 0),
        (None, -1, None),
    ]


@pytest.mark.chainable
def test_filter_index(random_chainable_iterator):
    result = list(random_chainable_iterator.filter_index(lambda x: x % 3))
    assert result == [0, 3, 4]


@pytest.mark.chainable
def test_sliding_window(short_chainable_iterator):
    result = list(short_chainable_iterator.sliding_window(2))
    assert result == [(0, 1), (1, 2), (2, 3), (3, 4)]


@pytest.mark.chainable
def test_sliding_window_step(short_chainable_iterator):
    result = list(short_chainable_iterator.sliding_window(2, step=2))
    assert result == [(0, 1), (2, 3)]


@pytest.mark.chainable
def test_batched_ignore(short_chainable_iterator):
    result = list(short_chainable_iterator.batched(n=3))
    assert result == [(0, 1, 2), (3, 4)]


@pytest.mark.chainable
def test_batched_strict(short_chainable_iterator):
    with pytest.raises(ValueError) as excinfo:
        list(short_chainable_iterator.batched(n=3, policy="strict"))
    assert excinfo.value.args[0] == "Uneven size"


@pytest.mark.chainable
def test_batched_fill(short_chainable_iterator):
    result = list(short_chainable_iterator.batched(n=3, policy="fill"))
    assert result == [(0, 1, 2), (3, 4, None)]


@pytest.mark.chainable
def test_islice(long_chainable_iterator):
    result = list(long_chainable_iterator.islice(start=35, stop=50, step=4))
    assert result == [35, 39, 43, 47]


@pytest.mark.chainable
def test_islice_negative(long_chainable_iterator):
    with pytest.raises(ValueError) as excinfo:
        list(long_chainable_iterator.islice(start=50, stop=35, step=-4))
    assert (
        excinfo.value.args[0] == "Step for islice() must be a positive integer or None."
    )


@pytest.mark.chainable
def test_skip(long_chainable_iterator):
    result = list(long_chainable_iterator.skip(95))
    assert result == [95, 96, 97, 98, 99]


@pytest.mark.chainable
def test_getitems(long_chainable_iterator):
    result = list(long_chainable_iterator.getitems([5, 15, 91]))
    assert result == [5, 15, 91]


@pytest.mark.chainable
def test_starmap(short_chainable_iterator):
    result = list(short_chainable_iterator.starmap(add, range(5, 10)))
    assert result == [5, 7, 9, 11, 13]


@pytest.mark.chainable
def test_starmap_no_args(short_chainable_iterator):
    with pytest.raises(ValueError):
        list(short_chainable_iterator.starmap(lambda x: x**2))


@pytest.mark.chainable
def test_split_at(short_chainable_iterator):
    result = list(short_chainable_iterator.split_at(2, 3))
    assert result == [(0, 1), (2,), (3, 4)]


@pytest.mark.chainable
def test_enumerate(short_chainable_iterator):
    result = list(short_chainable_iterator.enumerate(start=2))
    assert result == [(2, 0), (3, 1), (4, 2), (5, 3), (6, 4)]


@pytest.mark.nested
def test_starmap_nested_no_args(nested_iterator):
    result = list(nested_iterator.starmap(pow))
    assert result == [1, 2, 4, 8, 16]


@pytest.mark.nested
def test_starmap_nested(nested_iterator):
    result = list(nested_iterator.starmap(sum, [1] * 5))
    assert result == [3, 4, 5, 6, 7]


@pytest.mark.nested
def reverse_each(nested_iterator):
    result = list(nested_iterator.reverse_each())
    assert result == [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]


@pytest.mark.nested
def first_of_each(nested_iterator):
    result = list(nested_iterator.first_of_each())
    assert result == [2, 2, 2, 2, 2]


@pytest.mark.nested
def last_of_each(nested_iterator):
    result = list(nested_iterator.first_of_each())
    assert result == [0, 1, 2, 3, 4]


@pytest.mark.nested
def transpose(nested_iterator):
    result = list(nested_iterator.transpose())
    assert result == [(2, 2, 2, 2, 2), (0, 1, 2, 3, 4)]
