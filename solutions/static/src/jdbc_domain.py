"""Reference JDBC state abstraction for one PreparedStatement.

This module is independent of the interpreter. The integrator owns statement
references, heap updates, stub registration, return values and SQL parsing.
An operation returns normal successor states plus possible errors/warnings.
Batch operations model H2 2.5.252 eager validation and local batch execution.
SQL syntax, types, database contents and exception handlers are outside this domain.
A retained value is a reuse warning, not by itself an SQL error.
"""
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable, Protocol

from abstractions import Lattice


MAX_ALTERNATIVES = 32
MAX_TRACKED_PARAMETERS = 64
MAX_BATCH_SHAPES = 16


class Binding(Enum):
    UNSET = "unset"
    FRESH = "fresh"
    RETAINED = "retained"


@dataclass(frozen=True)
class QueryShape:
    text: str | None
    arity: int | None

    def __post_init__(self):
        if self.arity is not None and self.arity < 0:
            raise ValueError("Query arity must be nonnegative")


class StringValue(Protocol):
    def possible_strings(self) -> frozenset[str] | None: ...


# None means unknown bindings or unknown pending batch contents.
# Batch summaries discard multiplicity and order, except for the last snapshot.
# A None batch is an overflow summary of a nonempty batch.
type Bindings = tuple[Binding, ...]
type Batch = frozenset[Bindings | None] | None


@dataclass(frozen=True)
class Statement:
    query: QueryShape
    opened: bool
    bindings: Bindings | None
    batch: Batch = frozenset()
    last_batch: Bindings | None = None


def _fresh_epoch(bindings: Bindings | None) -> Bindings | None:
    if bindings is None:
        return None
    return tuple(Binding.RETAINED if b is Binding.FRESH else b for b in bindings)


def _initial_bindings(arity: int | None) -> Bindings | None:
    if arity is None or arity > MAX_TRACKED_PARAMETERS:
        return None
    return (Binding.UNSET,) * arity


@dataclass(frozen=True)
class JDBCState(Lattice):
    """Bounded powerset of summary records; None is top, empty is bottom.

    Ordering is set inclusion on records, not inclusion of their concrete
    meanings. Unknown fields are opaque summary tokens. In particular, meet
    is structural intersection; do not use it as a concrete branch-refinement
    operator. Forward transfer and join are the intended analysis operations.
    The ascending chain height is at most MAX_ALTERNATIVES + 2 elements.
    """
    alternatives: frozenset[Statement] | None

    def __post_init__(self):
        if self.alternatives is not None:
            alternatives = frozenset(self.alternatives)
            object.__setattr__(self, "alternatives",
                               None if len(alternatives) > MAX_ALTERNATIVES else alternatives)

    @classmethod
    def top(cls):
        return cls(None)

    @classmethod
    def bot(cls):
        return cls(frozenset())

    @classmethod
    def prepare(cls, queries: Iterable[QueryShape] | None):
        if queries is None:
            queries = (QueryShape(None, None),)
        return cls(frozenset(Statement(q, True, _initial_bindings(q.arity)) for q in queries))

    @classmethod
    def prepare_from_strings(cls, value: StringValue, arity_of: Callable[[str], int | None]):
        """Adapter for the shared StringValue protocol; arity comes from SQL analysis.

        The callback must return None for unsupported or ambiguous SQL. A raw
        count of question marks is not generally a sound SQL parameter parser.
        """
        texts = value.possible_strings()
        if texts is None:
            return cls.prepare(None)
        return cls.prepare(QueryShape(text, arity_of(text)) for text in texts)

    def __or__(self, other):
        if not isinstance(other, JDBCState):
            return NotImplemented
        if self.alternatives is None or other.alternatives is None:
            return self.top()
        return JDBCState(self.alternatives | other.alternatives)

    def __and__(self, other):
        """Structural meet only; not a concrete feasibility test."""
        if not isinstance(other, JDBCState):
            return NotImplemented
        if self.alternatives is None:
            return other
        if other.alternatives is None:
            return self
        return JDBCState(self.alternatives & other.alternatives)

    def __lt__(self, other):
        if not isinstance(other, JDBCState):
            return NotImplemented
        if other.alternatives is None:
            return self.alternatives is not None
        return self.alternatives is not None and self.alternatives < other.alternatives

    def __gt__(self, other):
        if not isinstance(other, JDBCState):
            return NotImplemented
        return other < self

    def __sexpr__(self):
        return str(self)

    def __str__(self):
        if self.alternatives is None:
            return "JDBC_TOP"
        if not self.alternatives:
            return "JDBC_BOTTOM"
        return "JDBC{" + ", ".join(sorted(map(repr, self.alternatives))) + "}"

    def bind(self, index: int | None):
        """None denotes an unknown integer index, not the Java null value."""
        return self._apply("bind", index)

    def bind_indices(self, indices: Iterable[int] | None):
        """Transfer over possible integer indices; None is unknown, empty is bottom."""
        if indices is None:
            return self.bind(None)
        state = self.bot()
        errors = set()
        warnings = set()
        for index in frozenset(indices):
            result = self.bind(index)
            state = state | result.state
            errors.update(result.errors)
            warnings.update(result.warnings)
        return Transition(state, frozenset(errors), frozenset(warnings))

    def clear_parameters(self):
        return self._apply("clear_parameters")

    def execute(self):
        return self._apply("execute")

    def add_batch(self):
        return self._apply("add_batch")

    def execute_batch(self):
        return self._apply("execute_batch")

    def clear_batch(self):
        return self._apply("clear_batch")

    def close(self):
        return self._apply("close")

    def _apply(self, operation: str, index: int | None = None):
        normal = set()
        errors = set()
        warnings = set()
        if self.alternatives is None:
            # Overflow loses object details. Keep both success and error possible.
            if operation == "close":
                return Transition(self.top())
            errors.add("closed-statement")
            if operation == "bind":
                errors.add("invalid-parameter-index")
                if index is not None and index <= 0:
                    return Transition(self.bot(), frozenset(errors))
            if operation in ("execute", "add_batch", "execute_batch"):
                errors.add("missing-parameter-binding")
                warnings.add("retained-parameter")
            return Transition(self.top(), frozenset(errors), frozenset(warnings))

        for s in self.alternatives:
            if operation == "close":
                normal.add(replace(s, opened=False, batch=frozenset(), last_batch=None))
                continue
            if not s.opened:
                errors.add("closed-statement")
                continue

            if operation == "bind":
                arity = s.query.arity
                if index is not None and (index <= 0 or (arity is not None and index > arity)):
                    errors.add("invalid-parameter-index")
                    continue
                if index is None or arity is None:
                    errors.add("invalid-parameter-index")
                if arity == 0:
                    continue
                if s.bindings is None:
                    normal.add(s)
                else:
                    indices = range(1, arity + 1) if index is None else (index,)
                    for i in indices:
                        values = s.bindings[:i-1] + (Binding.FRESH,) + s.bindings[i:]
                        normal.add(replace(s, bindings=values))
            elif operation == "clear_parameters":
                normal.add(replace(s, bindings=_initial_bindings(s.query.arity)))
            elif operation == "execute":
                if _may_execute(s.bindings, errors, warnings):
                    normal.add(replace(s, bindings=_fresh_epoch(s.bindings)))
            elif operation == "add_batch":
                # H2 checks all required parameters before queuing the row.
                if not _may_execute(s.bindings, errors, warnings):
                    continue
                batch = None if s.batch is None else s.batch | frozenset((s.bindings,))
                if batch is not None and len(batch) > MAX_BATCH_SHAPES:
                    batch = None
                # Treat each queued row as a freshness boundary for reuse checks.
                normal.add(replace(s, batch=batch, last_batch=s.bindings,
                                   bindings=_fresh_epoch(s.bindings)))
            elif operation == "execute_batch":
                if s.batch is None:
                    # Successful H2 addBatch calls have already checked bindings.
                    # Overflow loses freshness information, not that invariant.
                    warnings.add("retained-parameter")
                    normal.add(replace(s, batch=frozenset(), last_batch=None,
                                       bindings=_fresh_epoch(s.last_batch)))
                else:
                    valid = True
                    for snapshot in s.batch:
                        if snapshot is None:
                            # The successful enqueue establishes completeness even
                            # when individual parameter states are unknown.
                            warnings.add("retained-parameter")
                        elif not _may_execute(snapshot, errors, warnings):
                            valid = False
                    if valid:
                        # H2 local execution loads each row into the statement;
                        # the last row remains bound. An empty batch changes nothing.
                        bindings = _fresh_epoch(s.last_batch) if s.batch else s.bindings
                        normal.add(replace(s, batch=frozenset(), last_batch=None,
                                           bindings=bindings))
            elif operation == "clear_batch":
                normal.add(replace(s, batch=frozenset(), last_batch=None))
            else:
                raise ValueError(f"Unknown operation: {operation}")
        return Transition(JDBCState(frozenset(normal)), frozenset(errors), frozenset(warnings))


@dataclass(frozen=True)
class Transition:
    state: JDBCState
    errors: frozenset[str] = frozenset()
    warnings: frozenset[str] = frozenset()


def _may_execute(bindings: Bindings | None, errors: set[str], warnings: set[str]) -> bool:
    """Check binding completeness; report retained values separately."""
    if bindings is None:
        errors.add("missing-parameter-binding")
        warnings.add("retained-parameter")
        return True
    if Binding.UNSET in bindings:
        errors.add("missing-parameter-binding")
        return False
    if Binding.RETAINED in bindings:
        warnings.add("retained-parameter")
    return True
