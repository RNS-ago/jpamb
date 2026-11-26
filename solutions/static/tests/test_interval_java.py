from hypothesis import given, settings
from hypothesis import strategies as st

import jvm
from abstractions import Interval
from interval_domain import IntervalDomain

MIN, MAX = -(2**31), 2**31 - 1
# Small values hit zero and sign changes, the edges hit overflow.
INT = st.one_of(st.integers(-3, 3), st.integers(MIN, MAX), st.sampled_from([MIN, MIN + 1, MAX - 1, MAX]))
OPS = [jvm.BinaryOpr.Add, jvm.BinaryOpr.Sub, jvm.BinaryOpr.Mul, jvm.BinaryOpr.Div, jvm.BinaryOpr.Rem]


def wrap(x):
    return (x - MIN) % 2**32 + MIN


def java(opr, x, y):
    """Concrete Java int arithmetic. Div and Rem need y != 0."""
    match opr:
        case jvm.BinaryOpr.Add:
            return wrap(x + y)
        case jvm.BinaryOpr.Sub:
            return wrap(x - y)
        case jvm.BinaryOpr.Mul:
            return wrap(x * y)
        case jvm.BinaryOpr.Div:
            # Java rounds toward zero, Python's // rounds down.
            q = abs(x) // abs(y)
            return wrap(q if (x < 0) == (y < 0) else -q)
        case jvm.BinaryOpr.Rem:
            return wrap(x - y * java(jvm.BinaryOpr.Div, x, y))


def test_java_reference():
    D, R = jvm.BinaryOpr.Div, jvm.BinaryOpr.Rem
    assert java(D, -7, 2) == -3
    assert java(R, -7, 2) == -1
    assert java(R, 7, -2) == 1
    assert java(jvm.BinaryOpr.Add, MAX, 1) == MIN
    assert java(jvm.BinaryOpr.Mul, MIN, -1) == MIN
    assert java(D, MIN, -1) == MIN
    assert java(R, MIN, -1) == 0


@st.composite
def interval_and_member(draw):
    lo, hi = sorted([draw(INT), draw(INT)])
    return Interval(lo, hi), draw(st.integers(lo, hi) | st.sampled_from([lo, hi]))


@settings(max_examples=3000)
@given(interval_and_member(), interval_and_member(), st.sampled_from(OPS))
def test_arithmetic_is_sound(left, right, opr):
    (a, x), (b, y) = left, right
    results = list(IntervalDomain.arithmetic(a, b, opr))
    if opr in (jvm.BinaryOpr.Div, jvm.BinaryOpr.Rem) and y == 0:
        assert "divide by zero" in results
        return
    z = java(opr, x, y)
    assert any(isinstance(r, Interval) and r.min <= z <= r.max for r in results)


def test_exact_edge_cases():
    D, R = jvm.BinaryOpr.Div, jvm.BinaryOpr.Rem
    exact = Interval.exact
    assert list(IntervalDomain.arithmetic(exact(MIN), exact(-1), D)) == [exact(MIN)]
    assert list(IntervalDomain.arithmetic(exact(MIN), exact(-1), R)) == [exact(0)]
    assert list(IntervalDomain.arithmetic(exact(MAX), exact(1), jvm.BinaryOpr.Add)) == [exact(MIN)]
    assert list(IntervalDomain.arithmetic(exact(10), exact(2), D)) == [exact(5)]
