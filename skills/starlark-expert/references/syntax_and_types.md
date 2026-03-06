# Starlark Syntax and Data Types

## Table of Contents

* [Data Types](#data-types) (Lines 24 - 83)
  * [None, Booleans, Integers, Floats](#none-booleans-integers-floats)
  * [Strings and Bytes](#strings-and-bytes)
  * [Lists and Tuples](#lists-and-tuples)
  * [Dictionaries and Sets](#dictionaries-and-sets)
  * [Functions](#functions)
* [Value Concepts & Mutability](#value-concepts--mutability) (Lines 84 - 116)
  * [Identity and Mutation](#identity-and-mutation)
  * [Freezing a Value](#freezing-a-value)
  * [Hashing](#hashing)
* [Core Syntax](#core-syntax) (Lines 117 - 181)
  * [Indexing and Slicing](#indexing-and-slicing)
  * [Comprehensions](#comprehensions)
  * [Assignments](#assignments)
  * [Control Flow (If, For)](#control-flow)
  * [Function Definitions](#function-definitions)

---

## Data Types

### None, Booleans, Integers, Floats

* **None**: `None` is a distinguished value used to indicate the absence of any other value. Its truth value is `False`.
* **Booleans**: `True` and `False`. Any value may be explicitly converted to a Boolean using the built-in `bool` function. `None`, `0`, `""`, `b""`, `[]`, `()`, `{}`, and `set()` are considered `False`.
* **Integers**: Arbitrarily large signed integers.
* **Floats**: IEEE 754 double-precision floating-point numbers. Includes `+Inf`, `-Inf`, and `NaN`.

```python
100 // 5 * 9 + 32               # 212
int("0xffff", 16)               # 65535
3.0 // 2.0                      # 1.0
"A" if 1 + 1 else "B"           # "A"
```

### Strings and Bytes

* **Strings**: An immutable array of elements that encode Unicode text.
* **Bytes**: An immutable array of integers in the range 0-255. Indicated by a `b` prefix.

```python
"abc\ndef"                      # String literal with escape
r"a\nb"                         # Raw string literal
b"abc"                          # Bytes literal
```

### Lists and Tuples

* **Lists**: A mutable sequence of values.
* **Tuples**: An immutable sequence of values.

```python
[1, 2] + [3, 4]                 # [1, 2, 3, 4]
(1,)                            # a 1-tuple (trailing comma required)
(1, 2, 3)                       # a 3-tuple
```

### Dictionaries and Sets

* **Dictionaries**: A mutable mapping from keys to values. Keys must be hashable.
* **Sets**: A mutable collection of unique, hashable values.

```python
coins = {"penny": 1, "dime": 10}
coins["quarter"] = 25

s = set(["a", "b", "c"])
set([1, 2]) | set([3, 2])       # set([1, 2, 3]) (Union)
set([1, 2]) & set([2, 3])       # set([2]) (Intersection)
set([1, 2]) - set([2, 3])       # set([1]) (Difference)
set([1, 2]) ^ set([2, 3])       # set([1, 3]) (Symmetric Difference)
```

### Functions

A function value represents a function defined in Starlark. Function values used in a Boolean context are always considered true.

---

## Value Concepts & Mutability

### Identity and Mutation

Starlark is an imperative language. Values of some data types (`NoneType`, `bool`, `int`, `float`, `string`, `bytes`) are **immutable**. Values of other data types (`list`, `dict`, `set`) are **mutable**. `tuple` and `function` values are not directly mutable, but may refer to mutable values indirectly.

### Freezing a Value

Starlark has a unique feature: a mutable value may be **frozen** so that all subsequent attempts to mutate it fail with a dynamic error; the value, and all other values reachable from it, become immutable.
Immediately after execution of a Starlark module (like a `.bzl` file), all values in its top-level environment are frozen. This allows parallel execution without race conditions.

```python
# module a.sky
var = [] # declare a list
def fct():
    var.append(5) # fct can mutate var within this context

# module b.sky
load("a.sky", "var", "fct")
var.append(6)     # runtime error: the list stored in var is frozen
fct()             # runtime error: fct() attempts to modify a frozen list
```

### Hashing

Only **hashable** values are suitable as keys of a `dict` or elements of a `set`.

* Immutable types (`NoneType`, `bool`, `int`, `float`, `string`, `bytes`) are hashable.
* Tuples are hashable only if all their elements are hashable.
* Mutable types (`list`, `dict`, `set`) are **not hashable**, unless they have become immutable due to freezing.

---

## Core Syntax

### Indexing and Slicing

Indexing is zero-based. Slicing uses half-open intervals `a[start:stop:stride]`. Negative indices are relative to the end of the sequence.

```python
"hello"[1:4]                    # "ell"
"hello"[-3:-1]                  # "ll"
"banana"[1::2]                  # "aaa" (stride)
```

### Comprehensions

List and dict comprehensions construct new collections by looping over iterables.

```python
[x*x for x in range(5) if x%2 == 0]      # [0, 4, 16]
{x: len(x) for x in ["able", "baker"]}   # {"baker": 5, "able": 4}
```

### Assignments

Supports standard and augmented assignments (`+=`, `-=`, etc.). Starlark also supports tuple unpacking.

```python
a, b = 2, 3
[zero, one, two] = range(3)
x += 1
```

### Control Flow

* **If Statements**: Only permitted within a function definition.
* **For Statements**: Iterates over a finite sequence. Only permitted within a function definition. Modifying a collection during iteration is an error.

```python
if score >= 100:
    print("You win!")
elif score > 50:
    print("Getting there.")
else:
    print("Keep trying...")

for x in range(10):
    if x % 2 == 1:
        continue
    print(x)
```

### Function Definitions

Defined using `def`. Supports required parameters, optional parameters, `*args`, keyword-only parameters, and `**kwargs`.

```python
def f(a, b, c=1, *args, **kwargs):
    pass

def g(a, *, b=2, c):  # 'b' and 'c' are keyword-only
    print(a, b, c)

# Lambda expressions
twice = lambda x: x * 2
```
