"""Transfer functions for JVM integers and integer arrays."""
import jvm
from abstractions import INT_MAX, INT_MIN, ArrayRef, IntArray, Interval, LengthOrigin


def quotient(a, b):
    magnitude = abs(a) // abs(b)
    return -magnitude if (a < 0) != (b < 0) else magnitude


class IntervalDomain:
    bottom = staticmethod(Interval.bot)
    array_input = staticmethod(IntArray.input)

    def __init__(self, thresholds=()):
        self.thresholds = tuple(sorted({k for k in thresholds if INT_MIN <= k <= INT_MAX}))

    def widen(self, old, incoming):
        if isinstance(old, Interval) and isinstance(incoming, Interval):
            return old.widen(incoming, self.thresholds)
        return old | incoming

    @staticmethod
    def literal(typ, value):
        if isinstance(typ, jvm.Int):
            return Interval.exact(value)
        raise NotImplementedError(f"Unsupported interval literal {typ}")

    @staticmethod
    def parameter(typ, value):
        bounds = {jvm.Boolean: (0, 1), jvm.Char: (0, 65535), jvm.Byte: (-128, 127),
                  jvm.Short: (-32768, 32767), jvm.Int: (INT_MIN, INT_MAX)}
        if type(typ) not in bounds:
            raise NotImplementedError(f"Unsupported interval input {typ}")
        if value is None:
            return Interval(*bounds[type(typ)])
        raw = value.value
        return Interval.exact(ord(raw) if isinstance(raw, str) else int(raw))

    @staticmethod
    def allocate(pc, classname, state):
        raise NotImplementedError(f"Unsupported interval allocation {classname}")

    @staticmethod
    def arithmetic(a, b, op):
        if a.empty or b.empty:
            return
        operators = jvm.BinaryOpr
        if op == operators.Add:
            yield Interval.wrapped_range(a.min + b.min, a.max + b.max)
        elif op == operators.Sub:
            yield Interval.wrapped_range(a.min - b.max, a.max - b.min)
        elif op == operators.Mul:
            products = [x*y for x in (a.min, a.max) for y in (b.min, b.max)]
            yield Interval.wrapped_range(min(products), max(products))
        elif op in (operators.Div, operators.Rem):
            if b.may_equal(0):
                yield "divide by zero"
            result = Interval.bot()
            for divisor in (b & Interval(INT_MIN, -1), b & Interval(1, INT_MAX)):
                if divisor.empty:
                    continue
                if op == operators.Div:
                    quotients = [quotient(x, y) for x in (a.min, a.max)
                                 for y in (divisor.min, divisor.max)]
                    part = Interval.wrapped_range(min(quotients), max(quotients))
                elif a.min == a.max and divisor.min == divisor.max:
                    part = Interval.exact(a.min - quotient(a.min, divisor.min) * divisor.min)
                else:
                    limit = max(abs(divisor.min), abs(divisor.max)) - 1
                    part = Interval(max(min(a.min, 0), -limit), min(max(a.max, 0), limit))
                result |= part
            if not result.empty:
                yield result
        else:
            raise NotImplementedError(op)

    @staticmethod
    def branch_values(a, b, op):
        if a.empty or b.empty:
            return
        for order, left, right in (
            ("lt", Interval(a.min, min(a.max, b.max-1)), Interval(max(b.min, a.min+1), b.max)),
            ("eq", a & b, a & b),
            ("gt", Interval(max(a.min, b.min+1), a.max), Interval(b.min, min(b.max, a.max-1))),
        ):
            if left.empty or right.empty:
                continue
            accepted = {"Eq": {"eq"}, "Ne": {"lt", "gt"}, "Lt": {"lt"}, "Le": {"lt", "eq"},
                        "Gt": {"gt"}, "Ge": {"gt", "eq"}}
            yield order in accepted[op.name], left, right

    @classmethod
    def compare(cls, a, b, op):
        return {taken for taken, _, _ in cls.branch_values(a, b, op)}

    @staticmethod
    def array_access(pc, state, opcode):
        length_read = isinstance(opcode, jvm.ArrayLength)
        store = isinstance(opcode, jvm.ArrayStore)
        count = 1 if length_read else (3 if store else 2)
        operands, after = state.pop(count)
        ref = operands[0]
        if not isinstance(ref, ArrayRef):
            raise NotImplementedError("Expected interval-domain int[] reference")
        if ref.nullable:
            yield "null pointer"
        origin = state.origins[-count]
        for site in ref.sites:
            # Split by target object. Other possible targets are unchanged on this edge.
            edge = after.narrow(origin, ArrayRef(frozenset({site}), False))
            array = edge.arrays[site]
            if length_read:
                yield pc + 1, edge.push(array.length, LengthOrigin(site))
                continue

            index = operands[1]
            known_below = (state.origins[1-count], site) in state.below and index.min >= 0
            if not known_below and index.may_be_outside(0, array.length.min - 1):
                yield "out of bounds"
            valid = index & Interval(0, array.length.max - 1)
            if valid.empty:
                continue
            edge = edge.narrow(state.origins[1-count], valid)
            edge = edge.narrow(LengthOrigin(site), Interval(valid.min + 1, INT_MAX))
            if store:
                updated = edge.arrays[site].write(valid, operands[2])
                yield pc + 1, edge.with_array(site, updated)
            else:
                yield pc + 1, edge.push(array.read(valid))
