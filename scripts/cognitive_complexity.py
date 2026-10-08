"""Cognitive complexity of every function, measured from the stdlib ``ast``.

Cognitive complexity is the measure of SonarSource's rule S3776 ("Cognitive
Complexity of functions should not be too high"), and its limit is that
rule's: 15, and 15 itself passes. This repository runs no Sonar analysis;
this script is the check, run by ``mise run complexity`` and in CI.

Usage: ``python scripts/cognitive_complexity.py [--limit N] [paths...]``. Each
path is a ``.py`` file or a directory walked for them; with no paths the
package, ``scripts/`` and ``tests/`` are walked. Every function over the limit
is printed as ``path:line name complexity`` and the exit status is non-zero
when there is one.

The rules are the analyser's, read from sonar-python's
``CognitiveComplexityVisitor`` and ``CognitiveComplexityFunctionCheck``
(``tests/test_cognitive_complexity.py`` pins each one):

- ``if``, ``for``, ``while`` and each ``except`` cost 1 plus the nesting depth;
  ``elif`` and every ``else`` (on ``if``, on a loop, on ``try``) cost 1 flat.
- A ternary costs 1 plus the nesting depth and raises the depth for all three
  of its operands.
- Each boolean-operator expression costs 1 flat, however many operands it
  chains; ``not`` costs nothing.
- Comprehensions, ``with``, ``assert``, ``lambda`` and recursion cost nothing.
- The body of a nested function starts one level deeper than the statement
  defining it, except in a wrapper — an enclosing function whose other
  statements are all ``return <name>`` (the decorator shape), where it starts
  at the same level. A class body starts back at zero.
- A nested function is not reported on its own: its cost belongs to the
  outermost function that carries it, which is what the analyser flags.

One shape the ``ast`` cannot see: the analyser flattens an unparenthesised
mixed chain into one operator sequence and charges every operator change, so
``a or b and c or d`` costs 3 there and 2 here; parentheses around the inner
chain make the two agree.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = ("memex_mcp", "scripts", "tests")
DEFAULT_LIMIT = 15

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class _Counter:
    """One measurement: the running total and the scope the walk is in.

    ``nesting`` travels as a parameter because it is a property of the
    position in the tree; ``scope`` is state because a nested function's
    starting depth depends on the function *enclosing* it, not on the branch
    it sits in.
    """

    def __init__(self) -> None:
        self.total: int = 0
        self._scope: FunctionNode | None = None

    # -- statements ---------------------------------------------------------

    def statements(self, body: Iterable[ast.stmt], nesting: int) -> None:
        for statement in body:
            self.statement(statement, nesting)

    def statement(self, node: ast.stmt, nesting: int) -> None:
        if isinstance(node, ast.If):
            self._if(node, nesting, elif_branch=False)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            self._loop(node, nesting)
        elif isinstance(node, (ast.Try, ast.TryStar)):
            self._try(node, nesting)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.function(node, nesting)
        elif isinstance(node, ast.ClassDef):
            self._class(node)
        elif isinstance(node, ast.Match):
            self._match(node, nesting)
        else:
            self._children(node, nesting)

    def _if(self, node: ast.If, nesting: int, *, elif_branch: bool) -> None:
        self.total += 1 if elif_branch else 1 + nesting
        self.expression(node.test, nesting)
        self.statements(node.body, nesting + 1)
        branch = _elif_branch(node)
        if branch is not None:
            self._if(branch, nesting, elif_branch=True)
        elif node.orelse:
            self._else(node.orelse, nesting)

    def _else(self, body: list[ast.stmt], nesting: int) -> None:
        self.total += 1
        self.statements(body, nesting + 1)

    def _loop(self, node: ast.For | ast.AsyncFor | ast.While, nesting: int) -> None:
        self.total += 1 + nesting
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.expression(child, nesting)
        self.statements(node.body, nesting + 1)
        if node.orelse:
            self._else(node.orelse, nesting)

    def _try(self, node: ast.Try | ast.TryStar, nesting: int) -> None:
        self.statements(node.body, nesting)
        for handler in node.handlers:
            self.total += 1 + nesting
            if handler.type is not None:
                self.expression(handler.type, nesting)
            self.statements(handler.body, nesting + 1)
        if node.orelse:
            self._else(node.orelse, nesting)
        self.statements(node.finalbody, nesting)

    def _match(self, node: ast.Match, nesting: int) -> None:
        self.expression(node.subject, nesting)
        for case in node.cases:
            if case.guard is not None:
                self.expression(case.guard, nesting)
            self.statements(case.body, nesting + 1)

    def _class(self, node: ast.ClassDef) -> None:
        self._header(node, 0)
        enclosing = self._scope
        self._scope = None
        self.statements(node.body, 0)
        self._scope = enclosing

    def function(self, node: FunctionNode, nesting: int) -> None:
        enclosing = self._scope
        base = self._function_base(node, enclosing, nesting)
        self._header(node, base)
        self._scope = node
        self.statements(node.body, base)
        self._scope = enclosing

    @staticmethod
    def _function_base(node: FunctionNode, enclosing: FunctionNode | None, nesting: int) -> int:
        if enclosing is None:
            return 0
        if all(
            _is_simple_return(statement) for statement in enclosing.body if statement is not node
        ):
            return nesting
        return nesting + 1

    def _header(self, node: ast.ClassDef | FunctionNode, nesting: int) -> None:
        """Decorators, bases, parameter defaults and annotations: outside the body, at its depth.

        The analyser enters the definition's own nesting level before it reads
        the head, so a ternary in a nested function's default is as deep as
        one in its body, and one in a class decorator is back at zero.
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.expression(child, nesting)
            elif not isinstance(child, ast.stmt):
                self._children(child, nesting)

    def _children(self, node: ast.AST, nesting: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.expression(child, nesting)
            elif isinstance(child, ast.stmt):
                self.statement(child, nesting)
            else:
                self._children(child, nesting)

    # -- expressions --------------------------------------------------------

    def expression(self, node: ast.expr, nesting: int) -> None:
        if isinstance(node, ast.BoolOp):
            self.total += 1
        elif isinstance(node, ast.IfExp):
            self.total += 1 + nesting
            nesting += 1
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.expression(child, nesting)
            else:
                self._children(child, nesting)


def _elif_branch(node: ast.If) -> ast.If | None:
    """``elif`` and ``else: if`` parse alike; only the keyword column tells them apart."""
    if len(node.orelse) != 1:
        return None
    branch = node.orelse[0]
    if isinstance(branch, ast.If) and branch.col_offset == node.col_offset:
        return branch
    return None


def _is_simple_return(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Return) and isinstance(statement.value, ast.Name)


def complexity(node: FunctionNode) -> int:
    """The cognitive complexity of one function, nested functions included."""
    counter = _Counter()
    counter.function(node, 0)
    return counter.total


def outer_functions(node: ast.AST) -> Iterator[FunctionNode]:
    """Every function the analyser measures: not enclosed by another function.

    Methods count, and so does a function defined under a module-level
    ``if``; a function inside a function is part of its enclosing function's
    number and never its own.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child
        else:
            yield from outer_functions(child)


def over_limit(tree: ast.AST, limit: int) -> list[tuple[int, str, int]]:
    """``(line, name, complexity)`` for every function over *limit*, in file order."""
    findings: list[tuple[int, str, int]] = []
    for function in outer_functions(tree):
        score = complexity(function)
        if score > limit:
            findings.append((function.lineno, function.name, score))
    return findings


def default_paths() -> list[Path]:
    """The code directories under the repository root — what a bare run walks."""
    return [REPO_ROOT / name for name in DEFAULT_PATHS]


def python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(child for child in path.rglob("*.py") if child.is_file())
        else:
            yield path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Report every function whose cognitive complexity is over the limit."
    )
    parser.add_argument(
        "paths", nargs="*", type=Path, help="files or directories (default: the code directories)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"highest complexity that passes (default {DEFAULT_LIMIT})",
    )
    args = parser.parse_args(argv)
    limit = cast("int", args.limit)
    paths = cast("list[Path]", args.paths) or default_paths()
    findings = 0
    for path in python_files(paths):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            raise SystemExit(f"cannot measure {path}: {error}") from None
        for line, name, score in over_limit(tree, limit):
            sys.stdout.write(f"{os.path.relpath(path)}:{line} {name} {score}\n")
            findings += 1
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
