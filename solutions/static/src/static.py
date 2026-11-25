from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from operator import or_
from typing import Protocol, runtime_checkable

from abstractions import ArrayRef, IntArray, Lattice, LengthOrigin, SignSet

import jpamb
import sexpr
from jpamb import jvm
from jvm.state import PC, StackInt


@dataclass(frozen=True)
class State(sexpr.AsSExpr):
    locals: tuple[Lattice, ...]
    stack: tuple[Lattice, ...]

    memory: Lattice | None = None
    # A guard can refine the local or array length that produced a stack value.
    origins: tuple[int | LengthOrigin | None, ...] = ()
    arrays: tuple[IntArray, ...] = ()

    def __post_init__(self):
        assert isinstance(self.locals, tuple)
        assert isinstance(self.stack, tuple)
        if self.stack and not self.origins:
            object.__setattr__(self, "origins", (None,) * len(self.stack))
        assert len(self.stack) == len(self.origins)

    def __str__(self):
        return f"{', '.join(map(str, self.locals))}/{':'.join(map(str, self.stack))}"

    def __or__(self, other):
        return self.merge(other, or_)

    def merge(self, other, join_value):
        assert isinstance(other, State), f"Expected State but got {other!r}"
        assert len(self.stack) == len(other.stack), "Stack heights must match"
        assert len(self.locals) == len(other.locals), "Local counts must match"

        if (self.memory is None) != (other.memory is None):
            raise ValueError("Joined states must use the same memory domain")

        return State(
            tuple(join_value(s1, s2) for s1, s2 in zip(self.locals, other.locals)),
            tuple(join_value(s1, s2) for s1, s2 in zip(self.stack, other.stack)),
            self.memory | other.memory if self.memory is not None else other.memory,
            tuple(a if a == b else None for a, b in zip(self.origins, other.origins)),
            tuple(a.merge(b, join_value) for a, b in zip(self.arrays, other.arrays, strict=True)),
        )

    def push(self, value: Lattice, origin: int | LengthOrigin | None = None):
        return replace(self, stack=self.stack + (value,), origins=self.origins + (origin,))

    def pop(self, number=1):
        if not 0 <= number <= len(self.stack):
            raise ValueError("Operand stack underflow")
        split = len(self.stack) - number
        return self.stack[split:], replace(self, stack=self.stack[:split], origins=self.origins[:split])

    def load(self, index):
        return self.locals[index]

    def store(self, index, value):
        origins = tuple(None if source == index else source for source in self.origins)
        return replace(self, locals=self.locals[:index] + (value,) + self.locals[index + 1:], origins=origins)

    def with_array(self, site, array):
        return replace(self, arrays=self.arrays[:site] + (array,) + self.arrays[site + 1:])

    def narrow(self, index, value):
        if index is None:
            return self
        if isinstance(index, LengthOrigin):
            site = index.site
            array = self.arrays[site]
            narrowed = array.length & value
            state = self.with_array(site, replace(array, length=narrowed))
            stack = tuple(narrowed if source == index else old for old, source in zip(self.stack, self.origins))
            return replace(state, stack=stack)
        narrowed = self.locals[index] & value
        locals_ = self.locals[:index] + (narrowed,) + self.locals[index + 1:]
        stack = tuple(narrowed if source == index else old for old, source in zip(self.stack, self.origins))
        return replace(self, locals=locals_, stack=stack)


@dataclass(frozen=True)
class Finding:
    pc: PC
    kind: str
    message: str


@runtime_checkable
class StringValue(Protocol):
    """Read-only handoff from the string domain to JDBC stubs."""

    def possible_strings(self) -> frozenset[str] | None:
        """Return possible texts, None for unknown, or the empty set for bottom."""
        ...


class DomainAdapter(Protocol):
    """Bytecode transfer functions supplied to AbstractInterpreter.initial."""

    def bottom(self) -> Lattice: ...
    def parameter(self, typ: jvm.Type, value: jpamb.case.Value | None) -> Lattice: ...
    def literal(self, typ: jvm.Type, value: int | float | str) -> Lattice: ...
    def arithmetic(self, left: Lattice, right: Lattice, op: jvm.BinaryOpr) -> Iterable[Lattice | str]: ...
    def compare(self, left: Lattice, right: Lattice, op: jvm.CmpOpr) -> Iterable[bool | str]: ...
    def allocate(self, pc: PC, classname: jvm.ClassName, state: State) -> tuple[State, Lattice]: ...


class SignDomain:
    """Default JVM transfers. Supply a domain adapter for mixed value types."""
    bottom = staticmethod(SignSet.bot)

    @staticmethod
    def arithmetic(left, right, op):
        value, errors = left.arithmetic(right, op)
        yield from errors
        if value != SignSet.bot():
            yield value

    @staticmethod
    def branch_values(left, right, op):
        """Project feasible sign pairs back onto each operand for each edge."""
        accepted = {
            jvm.CmpOpr.Eq: {0}, jvm.CmpOpr.Ne: {-1, 1}, jvm.CmpOpr.Lt: {-1},
            jvm.CmpOpr.Le: {-1, 0}, jvm.CmpOpr.Gt: {1}, jvm.CmpOpr.Ge: {0, 1},
        }[op]
        pairs = {True: [], False: []}
        for a in left.signs:
            for b in right.signs:
                # Two positives (or two negatives) can occur in any order.
                orders = {-1, 0, 1} if a == b != 0 else {(a > b) - (a < b)}
                for taken in {order in accepted for order in orders}:
                    pairs[taken].append((a, b))
        for taken, feasible in pairs.items():
            if feasible:
                yield taken, SignSet(frozenset(a for a, _ in feasible)), SignSet(frozenset(b for _, b in feasible))

    @staticmethod
    def compare(left, right, op):
        return (taken for taken, _, _ in SignDomain.branch_values(left, right, op))

    @staticmethod
    def literal(typ, value):
        if isinstance(typ, jvm.Int):
            return SignSet.abstract([StackInt(value)])
        raise NotImplementedError(f"Unsupported literal {typ}")

    @staticmethod
    def allocate(pc, classname, state):
        raise NotImplementedError(f"Unsupported allocation of {classname} at {pc}")

    @staticmethod
    def parameter(typ, value):
        if value is None:
            return SignSet.top()
        if isinstance(value, (jpamb.case.Boolean, jpamb.case.Int)):
            return SignSet.abstract([StackInt(int(value.value))])
        if isinstance(value, jpamb.case.Array):
            return SignSet.from_sign("+")
        raise NotImplementedError(f"Unsupported input {value!r}")


@dataclass(frozen=True)
class CallResult:
    state: State
    value: Lattice | None = None


# Registry keys use JVM internal owner names and descriptors.
type StubKey = tuple[str, str, str]
type Stub = Callable[[PC, State, Lattice | None, tuple[Lattice, ...]], Iterable[CallResult | Finding | str]]


def dispatch_call(pc, op, state, stubs: Mapping[StubKey, Stub]):
    """Pass the residual state, receiver (None for static), and arguments in Java order."""
    method = op.method.extension
    descriptor = method.encode().split(":", 1)[1]
    key = (op.method.classname.slashed(), method.name, descriptor)
    if key not in stubs:
        raise NotImplementedError(f"No stub registered for {key}")
    args, after = state.pop(len(method.params))
    receiver = None
    if not isinstance(op, jvm.InvokeStatic):
        (receiver,), after = after.pop()
    for result in stubs[key](pc, after, receiver, args):
        if isinstance(result, (Finding, str)):
            yield result
            continue
        if (method.return_type is None) != (result.value is None):
            raise ValueError(f"Stub return value disagrees with {key}")
        successor = result.state
        if result.value is not None:
            successor = successor.push(result.value)
        yield pc + 1, successor


def successors(
    bc: jpamb.Bytecode,
    pc: PC,
    state: State,
    domain: DomainAdapter = SignDomain,
    stubs=None,
) -> Iterable[tuple[PC, State] | Finding | str]:
    opr = bc[pc]
    match opr:
        case jvm.InvokeStatic() | jvm.InvokeVirtual() | jvm.InvokeInterface() | jvm.InvokeSpecial():
            yield from dispatch_call(pc, opr, state, stubs or {})
        case jvm.Push(type=typ, value=value):
            yield pc + 1, state.push(domain.literal(typ, value))
        case jvm.ArrayLength() | jvm.ArrayLoad(type=jvm.Int()) | jvm.ArrayStore(type=jvm.Int()):
            handler = getattr(domain, "array_access", None)
            if handler is None:
                raise NotImplementedError(f"Unsupported array instruction at {pc}")
            yield from handler(pc, state, opr)
        case jvm.Store(index=i):
            (value,), after = state.pop()
            yield pc + 1, after.store(i, value)
        case jvm.Pop(words=1):
            _, after = state.pop()
            yield pc + 1, after
        case jvm.Dup(words=1):
            yield pc + 1, state.push(state.stack[-1], state.origins[-1])
        case jvm.Incr(index=i, amount=amount):
            for res in domain.arithmetic(state.load(i), domain.literal(jvm.Int(), amount), jvm.BinaryOpr.Add):
                yield res if isinstance(res, str) else (pc + 1, state.store(i, res))
        case jvm.If(condition=op, target=target) | jvm.Ifz(condition=op, target=target):
            if isinstance(opr, jvm.Ifz):
                (left,), after = state.pop()
                right = domain.literal(jvm.Int(), 0)
                origins = (state.origins[-1], None)
            else:
                (left, right), after = state.pop(2)
                origins = state.origins[-2:]
            refine = getattr(domain, "branch_values", None)
            if refine is None:
                for taken in domain.compare(left, right, op):
                    yield taken if isinstance(taken, str) else (pc % target if taken else pc + 1, after)
            else:
                for taken, lhs, rhs in refine(left, right, op):
                    edge = after.narrow(origins[0], lhs).narrow(origins[1], rhs)
                    yield (pc % target if taken else pc + 1), edge
        case jvm.Get(static=True, field=field):
            # Hack - Only handle the assertion case
            assert field.extension.name == "$assertionsDisabled"

            # Hack - Assuming assertions are never disabled
            va = domain.literal(jvm.Int(), 0)

            yield (pc + 1, state.push(va))

        case jvm.Load(index=i):
            va = state.load(i)
            yield (pc + 1, state.push(va, i))

        case jvm.Goto(target=t):
            yield (pc % t, state)

        case jvm.Binary(operant=op):
            [v1, v2], after = state.pop(2)
            for res in domain.arithmetic(v1, v2, op):
                if isinstance(res, str):
                    yield res
                else:
                    yield (pc + 1, after.push(res))

        case jvm.Return():
            yield "ok"

        case jvm.New(classname=jvm.ClassName("java.lang.AssertionError")):
            # Hack -- if we create an assertion error, we probably also throw it.
            yield "assertion error"

        case jvm.New(classname=classname):
            after, reference = domain.allocate(pc, classname, state)
            yield pc + 1, after.push(reference)

        case _:
            raise NotImplementedError(f"Unsupported instruction at {pc}: {opr}")


def initialstate(
    bc: jpamb.Bytecode,
    methodid: jvm.AbsMethodID,
    inputs: jpamb.Input | None,
    domain: DomainAdapter = SignDomain,
    memory=None,
) -> dict[PC, State]:
    method = bc.getmethod(methodid)
    locals = [domain.bottom()] * method.max_locals
    arrays = []
    for i, typ in enumerate(methodid.extension.params):
        value = None if inputs is None else inputs.values[i]
        if isinstance(typ, jvm.Array) and isinstance(typ.contains, jvm.Int) and hasattr(domain, "array_input"):
            site = len(arrays)
            # Unknown arguments can alias earlier arrays of the same type.
            sites = range(site + 1) if value is None else (site,)
            locals[i] = ArrayRef(frozenset(sites), nullable=value is None)
            arrays.append(domain.array_input(value))
        else:
            locals[i] = domain.parameter(typ, value)
    state = State(tuple(locals), (), memory, arrays=tuple(arrays))
    return {PC(methodid, 0): state}


@dataclass
class AbstractInterpreter:
    bc: jpamb.Bytecode
    worklist: deque[PC]
    states: dict[PC, State]

    domain: DomainAdapter = SignDomain
    stubs: Mapping[StubKey, Stub] = field(default_factory=dict)

    findings: set[Finding] = field(default_factory=set)

    @staticmethod
    def initial(bc: jpamb.Bytecode, methodid: jvm.AbsMethodID, inputs, *, domain: DomainAdapter = SignDomain, stubs=None, memory=None):
        states = initialstate(bc, methodid, inputs, domain, memory)
        worklist = deque(states.keys())

        return AbstractInterpreter(bc, worklist, states, domain, {} if stubs is None else stubs)

    def step(self) -> tuple[PC, set[str]]:
        pc = self.worklist.pop()

        finals = set()

        for res in successors(self.bc, pc, self.states[pc], self.domain, self.stubs):
            if isinstance(res, Finding):
                self.findings.add(res)
            elif isinstance(res, str):
                finals.add(res)
            else:
                pc_, st = res

                before = self.states.get(pc_, None)
                if before is None:
                    after = st
                elif pc_.offset <= pc.offset and hasattr(self.domain, "widen"):
                    # Widen on backward edges; ordinary branch joins keep their precision.
                    after = before.merge(st, self.domain.widen)
                else:
                    after = before | st
                if before is None or after != before:
                    self.states[pc_] = after
                    self.worklist.append(pc_)

        return pc, finals


def interpret():
    """The static analysis"""
    methodid, input, steps = jpamb.getcase(
        "static",
        "1.0",
        "The Rice Theorem Cookers",
        ["static", "python"],
        for_science=True,
    )
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})

    ai = AbstractInterpreter.initial(bc, methodid, input)

    x = jpamb.emit_init(ai.states)

    while steps > 0 and ai.worklist:
        pc, final = ai.step()
        for f in final:
            jpamb.emit_step(x, pc, f, depth=1)
            steps -= 1

        x = jpamb.emit_step(x, pc, ai.states, depth=1)
        steps -= 1


def analyse():
    _analyse(use_intervals=False)


def analyse_intervals():
    _analyse(use_intervals=True)


def _analyse(use_intervals):
    methodid = jpamb.getmethodid(
        "static",
        "1.0",
        "The Rice Theorem Cookers",
        ["static", "python"],
        for_science=True,
    )

    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})

    steps = 300

    final = set()
    domain = SignDomain
    if use_intervals:
        from interval_domain import IntervalDomain
        from jdbc_scan import scan
        domain = IntervalDomain(scan(bc.getmethod(methodid), budget=0).constants)
    ai = AbstractInterpreter.initial(bc, methodid, None, domain=domain)
    while steps > 0 and ai.worklist:
        _, outcomes = ai.step()
        final.update(outcomes)
        steps -= 1

    incomplete = bool(ai.worklist)
    for outcome in jpamb.QUERIES:
        answer = "maybe" if incomplete or outcome in final else "no"
        print(f"{outcome};{answer}")
