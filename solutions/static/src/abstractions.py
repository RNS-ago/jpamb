from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, replace
from operator import or_
from typing import Literal, Self

import jvm
import jvm.state as jvms


class Poset(ABC):
    @abstractmethod
    def __eq__(self, other: Self) -> bool: ...

    @abstractmethod
    def __lt__(self, other: Self) -> bool: ...

    def __le__(self, other: Self) -> bool:
        return self == other or self < other

    @abstractmethod
    def __gt__(self, other: Self) -> bool: ...

    def __ge__(self, other: Self) -> bool:
        return self == other or self > other


def is_reflexive(a: Poset):
    # Checking if the relation is reflexive (a == a)
    assert a == a, f"Reflexivity check failed: Element {a} is not related to itself."  # noqa: PLR0124


def is_antisymmetric(a: Poset, b: Poset):
    if a != b:
        # Checking that if a <= b and b <= a, then a must equal b
        assert not (a <= b <= a), (
            f"Antisymmetry violation: Found elements where {a} <= {b} and {b} <= {a}, "
            f"but {a} != {b}."
        )


def is_transitive(a: Poset, b: Poset, c: Poset):
    if a <= b <= c:
        # Checking that if a <= b and b <= c, then a <= c
        assert a <= c, (
            f"Transitivity violation: Found path {a} <= {b} and {b} <= {c}, "
            f"but {a} is not related to {c}."
        )


def is_poset(a: Poset, b: Poset, c: Poset):
    for x in [a, b, c]:
        is_reflexive(x)

    for x in [b, c]:
        is_antisymmetric(a, x)

    is_transitive(a, b, c)


class Lattice(Poset):
    @classmethod
    @abstractmethod
    def top(cls) -> Self:
        raise NotImplementedError(f"top: {cls}")

    @classmethod
    def bot(cls) -> Self:
        return cls.abstract([])

    @abstractmethod
    def __or__(self, other: Self) -> Self: ...

    @abstractmethod
    def __and__(self, other: Self) -> Self: ...


def is_lattice_identities(a: Lattice):
    # Top element identity checks
    assert (a | a.top()) == a.top(), (
        f"Top identity fail: ({a} | {a.top()}) -> {a | a.top()} != {a.top()}"
    )
    assert (a.top() | a) == a.top(), (
        f"Top identity fail (commute): ({a.top()} | {a}) != {a.top()}"
    )
    assert (a & a.top()) == a, f"Top identity fail (meet): ({a} & {a.top()}) != {a}"
    assert (a.top() & a) == a, (
        f"Top identity fail (meet commute): ({a.top()} & {a}) != {a}"
    )

    # Bottom element identity checks
    assert (a & a.bot()) == a.bot(), (
        f"Bottom identity fail: ({a} & {a.bot()}) != {a.bot()}"
    )
    assert (a.bot() & a) == a.bot(), (
        f"Bottom identity fail (commute): ({a.bot()} & {a}) != {a.bot()}"
    )
    assert (a.bot() | a) == a, f"Bottom identity fail (join): ({a.bot()} | {a}) != {a}"
    assert (a | a.bot()) == a, (
        f"Bottom identity fail (join commute): ({a} | {a.bot()}) != {a}"
    )


def is_lattice(a: Lattice, b: Lattice, c: Lattice):
    for x in [a, b, c]:
        is_lattice_identities(x)

    if not (a | b <= c):
        # Calculate values for the error message
        assert not (a <= c and b <= c), (
            f"Upper bound check failed: ({a} | {b}) -> {a | b} is not <= {c}. "
            f"Check individual bounds: {a <= c=}, {b <= c=}."
        )

    if not (c <= a & b):
        # Calculate values for the error message
        assert not (c <= a and c <= b), (
            f"Lower bound check failed: {c} is not <= ({a} & {b}) -> {a & b}. "
            f"Check individual bounds: {c <= a=}, {c <= b=}."
        )


class Abstraction(ABC):
    """This is the minimal interface for an abstraction."""

    @classmethod
    @abstractmethod
    def abstract(cls, values: Iterable[jvms.StackValue]) -> Self:
        raise NotImplementedError(f"abstract: {cls}")

    @abstractmethod
    def __contains__(self, value: jvms.StackValue) -> bool: ...


def is_galoi(a: set[jvms.StackValue], b: Abstraction):
    alpha_a = b.abstract(a)

    for value in a:
        assert value in alpha_a, f"{value} not in {a}"

    if alpha_a <= b:
        for value in a:
            assert value in b, f"{value} not in {b}"


type Sign = Literal[-1, 0, 1]


def to_sign(value: jvms.StackValue) -> Sign:
    match value:
        case jvms.StackInt(v):
            return (v > 0) - (v < 0)
        case a:
            raise NotImplementedError(f"Unsupported value {value!r}")


@dataclass(frozen=True, order=True)
class SignSet(Abstraction, Lattice):
    signs: frozenset[Sign]

    def __str__(self):
        return f"{{{''.join(z for z, b in zip('-0+', [-1, 0, 1]) if b in self.signs)}}}"

    def __sexpr__(self):
        return str(self)

    @classmethod
    def from_sign(cls, code):
        return cls(frozenset(b for z, b in zip("-0+", [-1, 0, 1]) if z in code))

    @classmethod
    def abstract(cls, values: Iterable[jvms.StackValue]) -> Self:
        signs = set()
        for value in values:
            signs.add(to_sign(value))
        return cls(frozenset(signs))

    def __contains__(self, value: jvms.StackValue) -> bool:
        return to_sign(value) in self.signs

    @classmethod
    def top(cls) -> bool:
        return cls(frozenset([-1, 0, 1]))

    def __or__(self, other: "SignSet") -> "SignSet":
        if not isinstance(other, SignSet):
            return NotImplemented

        return SignSet(self.signs | other.signs)

    def __and__(self, other: "SignSet") -> "SignSet":
        if not isinstance(other, SignSet):
            return NotImplemented

        return SignSet(self.signs & other.signs)

    def arithmetic(
        self, other: "SignSet", opr: jvm.BinaryOpr
    ) -> tuple["SignSet", set[str]]:
        match opr:
            case jvm.BinaryOpr.Div:
                signs = set()
                for numerator in self.signs:
                    for divisor in other.signs - {0}:
                        signs.add(0)  # Integer division can truncate to zero.
                        signs.add(numerator * divisor)
                        if numerator == divisor == -1:
                            signs.add(-1)  # JVM MIN_VALUE / -1 overflows.
                errors = {"divide by zero"} if self.signs and 0 in other.signs else set()
                return SignSet(frozenset(signs)), errors
            case jvm.BinaryOpr.Add:
                output = set()
                if 1 in self.signs:
                    output.add(1)
                    if -1 in other.signs:
                        output.update([0, -1])

                if -1 in self.signs:
                    output.add(-1)
                    if 1 in other.signs:
                        output.update([0, 1])

                if 0 in self.signs:
                    output.update(other.signs)

                return (SignSet(output), set())
            case _:
                raise NotImplementedError(f"TODO: {opr}")

    def compare(self, other: "SignSet", opr: jvm.CmpOpr) -> Iterable[bool]:
        match opr:
            case jvm.CmpOpr.Le:
                cases = set()
                for x in self.signs:
                    for y in other.signs:
                        if x == 0 or y == 0:
                            cases.add(x <= y)
                            continue
                        if x <= y:
                            cases.add(True)
                        if x >= y:
                            cases.add(False)
                return cases
            case _:
                raise NotImplementedError(f"TODO: {opr}")


INT_MIN = -(2**31)
INT_MAX = 2**31 - 1


@dataclass(frozen=True)
class Interval(Abstraction, Lattice):
    """Closed signed JVM int interval; [1,0] is the empty interval."""
    min: int
    max: int

    def __post_init__(self):
        if self.min > self.max:
            object.__setattr__(self, "min", 1)
            object.__setattr__(self, "max", 0)
        elif self.min < INT_MIN or self.max > INT_MAX:
            raise ValueError("Interval endpoint outside JVM int range")

    @property
    def empty(self):
        return self.min > self.max

    @classmethod
    def bot(cls):
        return cls(1, 0)

    @classmethod
    def top(cls):
        return cls(INT_MIN, INT_MAX)

    @classmethod
    def exact(cls, value):
        return cls(value, value)

    @classmethod
    def abstract(cls, values):
        result = cls.bot()
        for value in values:
            if not isinstance(value, jvms.StackInt):
                raise TypeError(value)
            result |= cls.exact(value.value)
        return result

    def __contains__(self, value):
        return isinstance(value, jvms.StackInt) and self.may_equal(value.value)

    def may_equal(self, value):
        return self.min <= value <= self.max

    def may_be_outside(self, low, high):
        return not self.empty and (self.min < low or self.max > high)

    def __or__(self, other):
        if self.empty:
            return other
        if other.empty:
            return self
        return Interval(min(self.min, other.min), max(self.max, other.max))

    def __and__(self, other):
        return Interval(max(self.min, other.min), min(self.max, other.max))

    def __lt__(self, other):
        return self != other and (self.empty or other.min <= self.min <= self.max <= other.max)

    def __gt__(self, other):
        return other < self

    def __sexpr__(self):
        return "bottom" if self.empty else f"[{self.min},{self.max}]"

    def widen(self, incoming, thresholds):
        if self.empty or incoming.empty:
            return self | incoming
        lo, hi = self.min, self.max
        if incoming.min < lo:
            lo = max((k for k in thresholds if k <= incoming.min), default=INT_MIN)
        if incoming.max > hi:
            hi = min((k for k in thresholds if k >= incoming.max), default=INT_MAX)
        return Interval(lo, hi)

    @classmethod
    def wrapped_range(cls, lo, hi):
        # Wrap precisely when the entire range lies in one signed 32-bit cycle.
        width = 2**32
        cycle = (lo - INT_MIN) // width
        if cycle != (hi - INT_MIN) // width:
            return cls.top()
        return cls(lo - cycle * width, hi - cycle * width)


@dataclass(frozen=True)
class ArrayRef:
    sites: frozenset[int]
    nullable: bool

    def __or__(self, other):
        if isinstance(other, Interval) and other.empty:
            return self
        return ArrayRef(self.sites | other.sites, self.nullable or other.nullable)

    def __and__(self, other):
        return ArrayRef(self.sites & other.sites, self.nullable and other.nullable)


@dataclass(frozen=True)
class LengthOrigin:
    site: int


@dataclass(frozen=True)
class IntArray:
    """Length, tracked prefix cells, and a summary interval for remaining cells."""

    PREFIX_LIMIT = 32

    length: Interval
    cells: tuple[Interval, ...]
    default: Interval

    @classmethod
    def input(cls, value):
        if value is None:
            return cls(Interval(0, INT_MAX), (), Interval.top())
        # Keep a bounded prefix exact, and summarise all remaining cells.
        cells = tuple(Interval.exact(x) for x in value.values[:cls.PREFIX_LIMIT])
        default = Interval.bot()
        for x in value.values[cls.PREFIX_LIMIT:]:
            default |= Interval.exact(x)
        return cls(Interval.exact(len(value.values)), cells, default)

    def read(self, valid):
        result = self.default if valid.max >= len(self.cells) else Interval.bot()
        for cell in self.cells[valid.min:valid.max + 1]:
            result |= cell
        return result

    def write(self, valid, value):
        """Return the updated object on the successful (in-bounds) edge."""
        end = min(valid.max + 1, self.PREFIX_LIMIT)
        cells = list(self.cells)
        cells.extend([self.default] * max(0, end - len(cells)))
        for i in range(valid.min, end):
            cells[i] = value if valid.min == valid.max else cells[i] | value
        default = self.default
        if valid.max >= self.PREFIX_LIMIT:
            # Updating one possible tail element preserves the other tail values.
            default |= value
        return replace(self, cells=tuple(cells), default=default)

    def merge(self, other, join_value):
        size = max(len(self.cells), len(other.cells))
        left = self.cells + (self.default,) * (size - len(self.cells))
        right = other.cells + (other.default,) * (size - len(other.cells))
        return IntArray(join_value(self.length, other.length),
                        tuple(join_value(a, b) for a, b in zip(left, right)),
                        join_value(self.default, other.default))

    def __or__(self, other):
        return self.merge(other, or_)
