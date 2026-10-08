"""How log lines and error messages name the type of a value parsed from JSON."""

from __future__ import annotations

_KINDS: dict[type, str] = {
    bool: "a boolean",
    int: "a number",
    float: "a number",
    str: "a string",
    list: "a list",
    dict: "an object",
    type(None): "null",
}


def json_kind(value: object) -> str:
    """``"a list"``, ``"null"`` and so on; the Python type name for anything JSON cannot hold."""
    return _KINDS.get(type(value), type(value).__name__)
