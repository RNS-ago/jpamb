from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace

from typing import Protocol, runtime_checkable

from abstractions import Lattice, SignSet

import jpamb
import sexpr
from jpamb import jvm
from jvm.state import PC, StackInt


@dataclass(frozen=True)
class State(sexpr.AsSExpr):
    locals: tuple[Lattice, ...]
    stack: tuple[Lattice, ...]

    memory: Lattice | None = None

    def __post_init__(self):
        assert isinstance(self.locals, tuple)
        assert isinstance(self.stack, tuple)

    def __str__(self):
        return f"{', '.join(map(str, self.locals))}/{':'.join(map(str, self.stack))}"

    def __or__(self, other):
        assert isinstance(other, State), f"Expected State but got {other!r}"
        assert len(self.stack) == len(other.stack), "Stack heights must match"
        assert len(self.locals) == len(other.locals), "Local counts must match"

        if (self.memory is None) != (other.memory is None):
            raise ValueError("Joined states must use the same memory domain")

        return State(
            tuple(s1 | s2 for s1, s2 in zip(self.locals, other.locals)),
            tuple(s1 | s2 for s1, s2 in zip(self.stack, other.stack)),
            self.memory | other.memory if self.memory is not None else other.memory,
        )

    def push(self, value: Lattice):
        return replace(self, stack=self.stack + (value,))

    def pop(self, number=1):
        if not 0 <= number <= len(self.stack):
            raise ValueError("Operand stack underflow")
        split = len(self.stack) - number
        return self.stack[split:], replace(self, stack=self.stack[:split])

    def load(self, index):
        return self.locals[index]

    def store(self, index, value):
        return replace(self, locals=self.locals[:index] + (value,) + self.locals[index + 1:])


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
    def compare(left, right, op):
        if op in (jvm.CmpOpr.Eq, jvm.CmpOpr.Ne):
            for a in left.signs:
                for b in right.signs:
                    if a == b:
                        yield op == jvm.CmpOpr.Eq
                    if a != b or a != 0:
                        yield op == jvm.CmpOpr.Ne
            return
        yield from left.compare(right, op)

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
        case jvm.Store(index=i):
            (value,), after = state.pop()
            yield pc + 1, after.store(i, value)
        case jvm.Pop(words=1):
            _, after = state.pop()
            yield pc + 1, after
        case jvm.Dup(words=1):
            yield pc + 1, state.push(state.stack[-1])
        case jvm.Incr(index=i, amount=amount):
            for res in domain.arithmetic(state.load(i), domain.literal(jvm.Int(), amount), jvm.BinaryOpr.Add):
                yield res if isinstance(res, str) else (pc + 1, state.store(i, res))
        case jvm.If(condition=op, target=target):
            (left, right), after = state.pop(2)
            for res in domain.compare(left, right, op):
                yield res if isinstance(res, str) else (pc % target if res else pc + 1, after)
        case jvm.Get(static=True, field=field):
            # Hack - Only handle the assertion case
            assert field.extension.name == "$assertionsDisabled"

            # Hack - Assuming assertions are never disabled
            va = domain.literal(jvm.Int(), 0)

            yield (pc + 1, state.push(va))

        case jvm.Ifz(condition=op, target=target):
            [val], after = state.pop(1)

            for res in domain.compare(val, domain.literal(jvm.Int(), 0), op):
                match res:
                    case True:
                        yield (pc % target, after)
                    case False:
                        yield (pc + 1, after)
                    case err:
                        yield err

        case jvm.Load(index=i):
            va = state.load(i)
            yield (pc + 1, state.push(va))

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
    for i, typ in enumerate(methodid.extension.params):
        value = None if inputs is None else inputs.values[i]
        locals[i] = domain.parameter(typ, value)
    state = State(tuple(locals), (), memory)
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
    """The static analysis"""

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
    ai = AbstractInterpreter.initial(bc, methodid, None)
    while steps > 0 and ai.worklist:
        _, outcomes = ai.step()
        final.update(outcomes)
        steps -= 1

    incomplete = bool(ai.worklist)
    for outcome in jpamb.QUERIES:
        answer = "maybe" if incomplete or outcome in final else "no"
        print(f"{outcome};{answer}")
