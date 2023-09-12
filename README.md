# Chainable Iterator

## Install instructions:
```shell
$ pip install "git+https://github.com/sayandipdutta/chainable_iter"
```

This module provides a convenient way to chain useful methods on an iterator.

## Example:
```python
from chainable_iter import ChainableIterator

ch_iter = ChainableIterator(range(10))

result = (
  ch_iter
    .skip(1)
    .map(lambda x: x**2)  # 1 4 9 16 25 36 49 64 81
    .takewhile(lambda x: x < 10)  # 1 4 9
    .getitems([0, 2]) # 1 9
    .make_collapsible() # it can no longer be used as a ChainableIterator
    .collapse(sum) # 10
)

print(result)
# 10
```

## Iterator Types:
- ChainableIterator
- NestedIterator

## Methods for `ChainableIterator`:
- map
- filter
- accumulate
- append
- prepend
- compress
- dropwhile
- takewhile
- zip2
- zip
- filter_index
- sliding_window
- batched
- islice
- skip
- getitems
- starmap
- split_at
- enumerate
- make_consumable
- make_collapsible


## Methods for `ChainableIterator`:
- starmap
- reverse_each
- first_of_each
- last_of_each
- transpose

## Consumer Types:
This module provides two types, that consume an iterator.
1. Consumable
  Consumable class has a single method, named collapse, that consumes the iterator
  by applying the provided function on each item of the iterator.
2. Collapsible
  Collapsible class has a single method, named collapse, that collapses the iterator
  to a single item, by applying the provided function as it's argument.

