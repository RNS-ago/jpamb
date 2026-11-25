"""Abstract strings: a set of at most K templates, or top (any string).

A template is a sequence of known text and holes. A hole stands for any
string, and is tainted if user input may flow into it.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Self

from abstractions import Abstraction, Lattice


@dataclass(frozen=True, order=True)
class Hole:
    tainted: bool

    def __str__(self):
        return "<input>" if self.tainted else "<any>"


type Template = tuple[str | Hole, ...]


def normalize(chunks: Iterable[str | Hole]) -> Template:
    # One spelling per template: "a" + "b" is "ab", and two holes in a row are one hole.
    out: list[str | Hole] = []
    for chunk in chunks:
        if chunk == "":
            continue
        if out and isinstance(chunk, str) and isinstance(out[-1], str):
            out[-1] += chunk
        elif out and isinstance(chunk, Hole) and isinstance(out[-1], Hole):
            out[-1] = Hole(out[-1].tainted or chunk.tainted)
        else:
            out.append(chunk)
    return tuple(out)


def _pattern(template: Template) -> re.Pattern:
    return re.compile(
        "".join(".*" if isinstance(c, Hole) else re.escape(c) for c in template),
        re.DOTALL,
    )


@dataclass(frozen=True)
class Str(Abstraction, Lattice):
    # None is top. Top may contain user input.
    templates: frozenset[Template] | None

    K: ClassVar[int] = 8

    @classmethod
    def _of(cls, templates: Iterable[Template]) -> Self:
        ts = frozenset(templates)
        return cls(None) if len(ts) > cls.K else cls(ts)

    @classmethod
    def const(cls, s: str) -> Self:
        return cls._of([normalize([s])])

    @classmethod
    def input(cls) -> Self:
        return cls._of([(Hole(True),)])

    @classmethod
    def abstract(cls, values: Iterable[str]) -> Self:
        # The concrete values of this domain are Python strings.
        return cls._of(normalize([v]) for v in values)

    @classmethod
    def top(cls) -> Self:
        return cls(None)

    def __contains__(self, value: str) -> bool:
        if self.templates is None:
            return True
        return any(_pattern(t).fullmatch(value) for t in self.templates)

    def __le__(self, other: Self) -> bool:
        if other.templates is None:
            return True
        return self.templates is not None and self.templates <= other.templates

    def __lt__(self, other: Self) -> bool:
        return self <= other and self != other

    def __ge__(self, other: Self) -> bool:
        return other <= self

    def __gt__(self, other: Self) -> bool:
        return other < self

    def __or__(self, other: Self) -> Self:
        if not isinstance(other, Str):
            return NotImplemented
        if self.templates is None or other.templates is None:
            return Str.top()
        return Str._of(self.templates | other.templates)

    def __and__(self, other: Self) -> Self:
        # The meet of the template order. Do not use it to refine a branch:
        # {"a"} & {<any>} is bottom although "a" matches <any>.
        if not isinstance(other, Str):
            return NotImplemented
        if self.templates is None:
            return other
        if other.templates is None:
            return self
        return Str(self.templates & other.templates)

    def concat(self, other: Self) -> Self:
        if self == Str.bot() or other == Str.bot():
            return Str.bot()
        if self.templates is None or other.templates is None:
            return Str.top()
        return Str._of(
            normalize(a + b) for a in self.templates for b in other.templates
        )

    def equals(self, other: Self) -> frozenset[bool]:
        if self == Str.bot() or other == Str.bot():
            return frozenset()
        mine, theirs = self.possible_strings(), other.possible_strings()
        if mine is None or theirs is None:
            return frozenset({True, False})
        return frozenset(a == b for a in mine for b in theirs)

    def possible_strings(self) -> frozenset[str] | None:
        """The exact texts, None if unknown, or the empty set for bottom."""
        if self.templates is None:
            return None
        if any(isinstance(c, Hole) for t in self.templates for c in t):
            return None
        return frozenset("".join(t) for t in self.templates)

    def tainted(self) -> bool:
        if self.templates is None:
            return True
        return any(isinstance(c, Hole) and c.tainted for t in self.templates for c in t)

    def __str__(self):
        if self.templates is None:
            return "T"
        shown = (
            "·".join(f'"{c}"' if isinstance(c, str) else str(c) for c in t) or '""'
            for t in self.templates
        )
        return "{" + ", ".join(sorted(shown)) + "}"

    def __sexpr__(self):
        return str(self)
