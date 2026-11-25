"""Syntactic hints for static JPAMB methods. All PCs are instruction indices."""
import argparse
from collections import defaultdict
from dataclasses import dataclass

import jpamb
import jvm


CALL = (jvm.InvokeInterface, jvm.InvokeVirtual, jvm.InvokeStatic, jvm.InvokeSpecial)
BRANCH = (jvm.If, jvm.Ifz)


@dataclass(frozen=True, order=True)
class Predicate:
    parameter: int
    operator: str
    constant: int


@dataclass(frozen=True)
class Candidate:
    predicate: Predicate
    query_sites: tuple[int, ...]
    jdbc_sites: tuple[int, ...]


@dataclass(frozen=True)
class ScanResult:
    selected: tuple[Candidate, ...]
    constants: frozenset[int]


def successors(code, pc):
    op = code[pc]
    if isinstance(op, jvm.Goto):
        return (op.target,)
    if isinstance(op, BRANCH):
        return (op.target, pc + 1)
    if isinstance(op, (jvm.Throw, jvm.Return)) or pc + 1 == len(code):
        return ()
    return (pc + 1,)


def reachable(edges, start, stop):
    seen = set()
    pending = [start]
    while pending:
        pc = pending.pop()
        if pc != stop and pc not in seen:
            seen.add(pc)
            pending.extend(edges[pc])
    return seen


def query_slice(code, predecessors):
    """Trace SQL arguments back through stack producers and reaching local stores.

    Handles local copies and fluent StringBuilder chains. Heap aliases, separate
    builder mutations and application helper bodies are outside this heuristic.
    """
    pending = []
    for pc, op in enumerate(code):
        if not isinstance(op, CALL):
            continue
        method = op.method.extension
        if (op.method.classname.slashed().startswith("java/sql/")
                and method.name in {"prepareStatement", "prepareCall", "execute", "executeQuery", "executeUpdate", "addBatch"}
                and len(method.params) and method.params[0].encode() == "Ljava/lang/String;"):
            pending.append((pc, "stack", len(method.params) - 1))

    relevant = set()
    seen = set()
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        pc, kind, position = item
        for prev in predecessors[pc]:
            op = code[prev]
            if kind == "local":
                if isinstance(op, jvm.Store) and op.index == position:
                    relevant.add(prev)
                    pending.append((prev, "stack", 0))
                else:
                    pending.append((prev, kind, position))
                continue

            # Stack effects for the supported category-one expression patterns.
            if isinstance(op, CALL):
                consumed = len(op.method.extension.params) + int(not isinstance(op, jvm.InvokeStatic))
                produced = int(op.method.extension.return_type is not None)
            elif isinstance(op, (jvm.Load, jvm.Push, jvm.New)):
                consumed, produced = 0, 1
            elif isinstance(op, jvm.Store):
                consumed, produced = 1, 0
            elif isinstance(op, (jvm.Goto, jvm.Incr)):
                consumed, produced = 0, 0
            elif isinstance(op, jvm.Ifz):
                consumed, produced = 1, 0
            elif isinstance(op, jvm.If):
                consumed, produced = 2, 0
            elif isinstance(op, jvm.Dup) and op.words == 1:
                consumed, produced = 1, 2
            elif isinstance(op, jvm.Pop) and op.words == 1:
                consumed, produced = 1, 0
            else:
                continue
            if position >= produced:
                pending.append((prev, "stack", position - produced + consumed))
            else:
                relevant.add(prev)
                if isinstance(op, jvm.Load):
                    pending.append((prev, "local", op.index))
                else:
                    pending.extend((prev, "stack", depth) for depth in range(consumed))
    return relevant


def guard(code, pc, parameters):
    op = code[pc]
    if pc < 1 or not isinstance(op, BRANCH):
        return None
    comparison = op.condition.name
    if isinstance(op, jvm.Ifz):
        load, constant = code[pc - 1], 0
    elif pc >= 2:
        load, literal = code[pc - 2:pc]
        if isinstance(literal, jvm.Load):
            load, literal = literal, load
            comparison = {"Lt": "Gt", "Ge": "Le", "Gt": "Lt", "Le": "Ge"}.get(comparison, comparison)
        if not isinstance(literal, jvm.Push) or not isinstance(literal.type, jvm.Int):
            return None
        constant = literal.value
    else:
        return None
    if not isinstance(load, jvm.Load) or load.index not in parameters:
        return None
    # Complementary branches cut the state space in exactly the same way.
    operator = {"Eq": "==", "Ne": "==", "Lt": "<", "Ge": "<", "Le": "<=", "Gt": "<="}[comparison]
    return Predicate(parameters[load.index], operator, constant)


def scan(method: jvm.Method, budget: int = 1) -> ScanResult:
    """Rank guards for partitioning and collect integer thresholds for widening."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    code = method.opcodes
    edges = [successors(code, pc) for pc in range(len(code))]
    predecessors = defaultdict(set)
    for pc, targets in enumerate(edges):
        for target in targets:
            predecessors[target].add(pc)

    written = {op.index for op in code if isinstance(op, (jvm.Store, jvm.Incr))}
    parameters = {}
    slot = 0
    for index, typ in enumerate(method.id.extension.params):
        if slot not in written and isinstance(typ, (jvm.Int, jvm.Boolean, jvm.Byte, jvm.Char, jvm.Short)):
            parameters[slot] = index
        slot += 2 if isinstance(typ, (jvm.Long, jvm.Double)) else 1

    constants = {op.value for op in code if isinstance(op, jvm.Push) and isinstance(op.type, jvm.Int)}
    constants.update(op.amount for op in code if isinstance(op, jvm.Incr))
    if any(isinstance(op, jvm.Ifz) for op in code):
        constants.add(0)
    query_ops = query_slice(code, predecessors)
    jdbc_ops = {pc for pc, op in enumerate(code)
                if isinstance(op, CALL) and op.method.classname.slashed().startswith("java/sql/")}
    evidence = {}
    for pc, op in enumerate(code):
        predicate = guard(code, pc, parameters)
        if predicate is None:
            continue
        arm = reachable(edges, op.target, pc) ^ reachable(edges, pc + 1, pc)
        query, jdbc = bool(arm & query_ops), bool(arm & jdbc_ops)
        if query or jdbc:
            query_sites, jdbc_sites = evidence.setdefault(predicate, ([], []))
            if query:
                query_sites.append(pc)
            if jdbc:
                jdbc_sites.append(pc)

    candidates = [Candidate(p, tuple(q), tuple(j)) for p, (q, j) in evidence.items()]
    candidates.sort(key=lambda c: (-(bool(c.query_sites) + bool(c.jdbc_sites)),
                                   -len(set(c.query_sites + c.jdbc_sites)), c.predicate))
    return ScanResult(tuple(candidates[:budget]), frozenset(constants))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", help="JPAMB static benchmark method identifier")
    parser.add_argument("--budget", type=int, default=1)
    args = parser.parse_args()
    bc = jpamb.Bytecode(*jpamb.setup())
    result = scan(bc.getmethod(jpamb.parse_methodid(args.method)), args.budget)
    for candidate in result.selected:
        p = candidate.predicate
        print(f"arg[{p.parameter}] {p.operator} {p.constant}; query guards {candidate.query_sites}; JDBC guards {candidate.jdbc_sites}")
    print("K =", sorted(result.constants))


if __name__ == "__main__":
    main()
