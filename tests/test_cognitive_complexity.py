"""The cognitive-complexity check: its rules, pinned one construct at a time.

What ``scripts/cognitive_complexity.py`` must match is the analyser behind
rule S3776, so it is pinned here by construct, not by the functions in the
tree that happen to be near the limit today — those get split, and a pin on
them would go with them. The fixture module below is a source string, not a
file under ``tests/``: the check walks ``tests/`` for ``.py`` files, and a
fixture that has to carry a function at 16 would fail the check it exists to
prove.

Each fixture function carries one construct and the cost the analyser
charges for it, as sonar-python's ``CognitiveComplexityVisitor`` computes it
(``visitElseClause`` for the loop and ``try`` else,
``isStmtListIncrementsNestingLevel`` for what does not nest,
``NestingLevel`` for the nested-function base, and the absence of any
``lambda`` handling).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import cognitive_complexity as gate

SCRIPT = Path(gate.__file__)

FIXTURE = """\
def if_statement(a):  # if 1
    if a:
        return 1
    return 0


def elif_branch(a, b):  # if 1, elif 1
    if a:
        return 1
    elif b:
        return 2
    return 0


def else_branch(a):  # if 1, else 1
    if a:
        return 1
    else:
        return 0


def else_holding_an_if(a, b):  # if 1, else 1, the if inside it 1 + nesting 1
    if a:
        return 1
    else:
        if b:
            return 2
    return 0


def for_loop(xs):  # for 1
    for x in xs:
        x()


def while_loop(a):  # while 1
    while a:
        a -= 1


def except_clause(f):  # except 1
    try:
        f()
    except ValueError:
        pass


def two_except_clauses(f):  # except 1, except 1
    try:
        f()
    except ValueError:
        pass
    except KeyError:
        pass


def elif_at_depth_one(a, b, c):  # if 1, if 1 + 1, elif 1
    if a:
        if b:
            return 1
        elif c:
            return 2
    return 0


def else_at_depth_one(a, b):  # if 1, if 1 + 1, else 1
    if a:
        if b:
            return 1
        else:
            return 2
    return 0


def for_at_depth_one(a, xs):  # if 1, for 1 + 1
    if a:
        for x in xs:
            x()


def while_at_depth_one(a):  # if 1, while 1 + 1
    if a:
        while a:
            a -= 1


def except_at_depth_one(a, f):  # if 1, except 1 + 1
    if a:
        try:
            f()
        except ValueError:
            pass


def nested_if_at_depth_two(a, b, c):  # if 1, if 1 + 1, if 1 + 2
    if a:
        if b:
            if c:
                return 1
    return 0


def ternary_at_depth_one(a, b):  # if 1, ternary 1 + 1
    if a:
        return 1 if b else 0
    return 0


def and_chain(a, b, c):  # one sequence 1
    return a and b and c


def mixed_operators(a, b, c):  # `and` sequence 1, `or` sequence 1
    return a and b or c


def parenthesised_inner_sequence(a, b, c, d):  # `or` sequence 1, `and` sequence 1
    return a or (b and c) or d


def negation(a):  # nothing: `not` is no sequence
    return not a


def comprehension(xs):  # nothing
    return [x for x in xs if x]


def recursive(n):  # if 1; the call to itself nothing
    if n:
        return recursive(n - 1)
    return 0


def nested_function(a):  # if 1; the inner body is one level deeper, so its if 1 + 1
    def inner(b):
        if b:
            return 1
        return 0

    if a:
        return inner(a)
    return 0


def wrapper_function(f):  # the decorator shape: the inner body is at the same level, its if 1
    def inner(b):
        if b:
            return f(b)
        return 0

    return inner


def class_in_a_function(a, b):  # if 1; the class body starts at zero: its if 1, its method's if 1
    if a:

        class Local:
            if b:
                flag = True

            def method(self, c):
                if c:
                    return 1
                return 0

        return Local
    return None


def loop_else(xs):  # for 1, else 1
    for x in xs:
        if x:  # if 1 + 1
            break
    else:
        return 0
    return 1


def try_else(f):  # except 1, else 1
    try:
        f()
    except ValueError:
        return 0
    else:
        return 1


def try_and_finally_bodies_do_not_nest(f):  # if 1, except 1, if 1 in finally
    try:
        if f:
            f()
    except ValueError:
        pass
    finally:
        if f:
            f()


def with_body_does_not_nest(f):  # if 1
    with f:
        if f:
            return 1
    return 0


def lambda_body_does_not_nest():  # ternary 1
    return lambda key: 1 if key else 0


def ternary_operands_nest(a, b, c):  # ternary 1, the ternary in its condition 1 + 1
    return 1 if (b if a else c) else 0


def decorator_expressions_count(a):
    return a


@decorator_expressions_count(1 or 2)  # `or` sequence 1
def decorated():
    return 0


def parameter_defaults_count(a=1 or 2):  # `or` sequence 1
    return a


def nested_head_is_as_deep_as_its_body(z):  # the inner head is one level deeper: ternary 1 + 1
    z = z or None  # `or` sequence 1, and what keeps the outer from being a wrapper

    def inner(x=1 if z else 2):
        return x

    return inner


def class_head_starts_at_zero(a):  # if 1; the class head starts at zero: its decorator's ternary 1
    if a:

        @decorator_expressions_count(1 if a else 2)
        class Local:
            pass

        return Local
    return None


def unparenthesised_mixed_chain(a, b, c, d):  # 2 here; the analyser charges 3, one per change
    # Pinned at the counter's number on purpose: the ast folds the shape and the
    # 3 is unreachable from it, which the module docstring states. A rewrite
    # that moves this number has changed what the docstring promises.
    return a or b and c or d


def at_limit(a, b, c, d, e):  # if 1, 2, 3, 4, 5
    if a:
        if b:
            if c:
                if d:
                    if e:
                        return 1
    return 0


def over_limit(a, b, c, d, e):  # if 1, 2, 3, 4, 5 and one `and` sequence 1
    if a:
        if b:
            if c:
                if d:
                    if e and a:
                        return 1
    return 0
"""

EXPECTED = {
    "if_statement": 1,
    "elif_branch": 2,
    "else_branch": 2,
    "else_holding_an_if": 4,
    "for_loop": 1,
    "while_loop": 1,
    "except_clause": 1,
    "two_except_clauses": 2,
    "elif_at_depth_one": 4,
    "else_at_depth_one": 4,
    "for_at_depth_one": 3,
    "while_at_depth_one": 3,
    "except_at_depth_one": 3,
    "nested_if_at_depth_two": 6,
    "ternary_at_depth_one": 3,
    "and_chain": 1,
    "mixed_operators": 2,
    "parenthesised_inner_sequence": 2,
    "negation": 0,
    "comprehension": 0,
    "recursive": 1,
    "nested_function": 3,
    "wrapper_function": 1,
    "class_in_a_function": 3,
    "loop_else": 4,
    "try_else": 2,
    "try_and_finally_bodies_do_not_nest": 3,
    "with_body_does_not_nest": 1,
    "lambda_body_does_not_nest": 1,
    "ternary_operands_nest": 3,
    "decorator_expressions_count": 0,
    "decorated": 1,
    "parameter_defaults_count": 1,
    "nested_head_is_as_deep_as_its_body": 3,
    "class_head_starts_at_zero": 2,
    "unparenthesised_mixed_chain": 2,
    "at_limit": 15,
    "over_limit": 16,
}


def _fixture_functions() -> dict[str, gate.FunctionNode]:
    return {function.name: function for function in gate.outer_functions(ast.parse(FIXTURE))}


def test_the_fixture_and_the_expectation_name_the_same_functions() -> None:
    assert set(_fixture_functions()) == set(EXPECTED)


@pytest.mark.parametrize(("name", "expected"), sorted(EXPECTED.items()))
def test_each_construct_costs_what_the_analyser_charges(name: str, expected: int) -> None:
    assert gate.complexity(_fixture_functions()[name]) == expected


def test_a_nested_function_is_measured_inside_its_outer_function_only() -> None:
    names = [function.name for function in gate.outer_functions(ast.parse(FIXTURE))]
    assert "inner" not in names
    assert "method" not in names
    assert names.count("nested_function") == 1


def test_a_method_is_measured_on_its_own() -> None:
    module = ast.parse("class C:\n    def method(self):\n        return 0\n")
    assert [function.name for function in gate.outer_functions(module)] == ["method"]


def test_the_limit_is_over_fifteen_not_at_it() -> None:
    findings = gate.over_limit(ast.parse(FIXTURE), gate.DEFAULT_LIMIT)
    assert [(name, score) for _, name, score in findings] == [("over_limit", 16)]


def test_the_limit_is_a_parameter() -> None:
    assert gate.over_limit(ast.parse(FIXTURE), 16) == []
    names = {name for _, name, _ in gate.over_limit(ast.parse(FIXTURE), 14)}
    assert names == {"at_limit", "over_limit"}


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def test_the_command_names_each_finding_and_fails(tmp_path: Path) -> None:
    (tmp_path / "fixture.py").write_text(FIXTURE, encoding="utf-8")
    line = next(f.lineno for f in _fixture_functions().values() if f.name == "over_limit")
    result = _run("fixture.py", cwd=tmp_path)
    assert result.returncode == 1
    assert result.stdout == f"fixture.py:{line} over_limit 16\n"
    assert result.stderr == ""


def test_the_command_walks_a_directory_and_passes_under_a_raised_limit(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "fixture.py").write_text(FIXTURE, encoding="utf-8")
    (tmp_path / "pkg" / "notes.txt").write_text("def not_python(): pass\n", encoding="utf-8")
    failing = _run("pkg", cwd=tmp_path)
    assert failing.returncode == 1
    assert failing.stdout.startswith("pkg/fixture.py:")
    passing = _run("--limit", "16", "pkg", cwd=tmp_path)
    assert passing.returncode == 0
    assert passing.stdout == ""


def test_a_bare_run_walks_the_three_code_directories_under_the_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert gate.default_paths() == [
        gate.REPO_ROOT / "memex_mcp",
        gate.REPO_ROOT / "scripts",
        gate.REPO_ROOT / "tests",
    ]
    assert (gate.REPO_ROOT / "pyproject.toml").is_file()
    root = tmp_path / "repo"
    for directory in ("memex_mcp", "scripts", "tests", "elsewhere"):
        (root / directory).mkdir(parents=True)
        (root / directory / "module.py").write_text(FIXTURE, encoding="utf-8")
    # Run from outside the root: the walk must start at the root, not at the
    # working directory, and paths print relative to where the run started.
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.chdir(tmp_path / "run")
    assert gate.main([]) == 1
    reported = [line.split(":")[0] for line in capsys.readouterr().out.splitlines()]
    assert reported == [
        "../repo/memex_mcp/module.py",
        "../repo/scripts/module.py",
        "../repo/tests/module.py",
    ]


def test_a_file_that_does_not_parse_stops_the_check(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def (:\n", encoding="utf-8")
    result = _run("broken.py", cwd=tmp_path)
    assert result.returncode != 0
    assert "cannot measure broken.py" in result.stderr
