"""The text side of writes: secrets, edits, index lines and sections, commit messages."""

from __future__ import annotations

import re
import subprocess

import pytest

from memex_mcp.notes import (
    MAX_INDEX_LINE_CHARS,
    appended,
    check_composed,
    check_secrets,
    checked_line,
    checked_section,
    checked_topic,
    commit_message,
    edited,
    index_entry,
    index_spot,
    is_daily,
    master_entry,
    new_index,
    section_names,
    version_of,
    with_entry,
    with_topic,
)
from memex_mcp.rights import InvalidRequest

# Built from pieces, so that no scanner takes this file for a leak.
SECRETS: dict[str, tuple[str, str]] = {
    "rsa key": ("-----BEGIN " + "RSA PRIVATE KEY-----\nMIIEow\n", "a private key"),
    "openssh key": ("-----BEGIN " + "OPENSSH PRIVATE KEY-----", "a private key"),
    "pgp key": ("-----BEGIN " + "PGP PRIVATE KEY BLOCK-----", "a private key"),
    "github classic": ("token: " + "gh" + "p_" + "a1B2" * 9, "a GitHub token"),
    "github fine-grained": ("github" + "_pat_" + "11ABC" * 6, "a GitHub token"),
    "sk key": ("key=" + "sk" + "-proj-" + "Ab3_" * 10, "an API key (sk-)"),
    "aws key": ("id " + "AK" + "IA" + "ABCDEFGH12345678", "an AWS access key"),
    "slack bot": ("xo" + "xb-" + "1234567890-abcdef", "a Slack token"),
    "slack user": ("xo" + "xp-" + "1234567890-abcdef", "a Slack token"),
    "gitlab": ("glp" + "at-" + "abcdefghij0123456789", "a GitLab token"),
}


@pytest.mark.parametrize(("text", "kind"), SECRETS.values(), ids=list(SECRETS))
def test_each_kind_of_secret_is_refused_by_name_without_its_value(text: str, kind: str) -> None:
    with pytest.raises(InvalidRequest) as caught:
        check_secrets(["harmless", text])
    message = str(caught.value)
    assert kind in message
    # No part of the value comes back, not even its prefix.
    assert text.rsplit(maxsplit=1)[-1][:8] not in message


@pytest.mark.parametrize(
    "text",
    [
        "The guest wifi password is hunter2-orange-42.",
        "Wi-Fi: SSID household, Passwort Sommer2026!",
        "desk-organizer-with-a-very-long-hyphenated-name-for-shelves",
        "the sk-learn library and the ghp_ prefix in prose",
        "AKIA is how AWS key ids begin",
        "-----BEGIN PUBLIC KEY-----",
        "-----BEGIN CERTIFICATE-----",
    ],
)
def test_ordinary_text_and_passwords_in_prose_pass(text: str) -> None:
    check_secrets([text])


def test_version_is_gits_blob_id() -> None:
    data = "# Note\n\nä text\n".encode()
    expected = subprocess.run(
        ["git", "hash-object", "--stdin"], input=data, capture_output=True, check=True
    ).stdout.decode()
    assert version_of(data) == expected.strip()


def test_edit_replaces_the_one_occurrence() -> None:
    assert edited("a b c\n", "b", "B") == "a B c\n"
    assert edited("a b c\n", " b", "") == "a c\n"


@pytest.mark.parametrize(
    ("current", "old", "message"),
    [
        ("a b c\n", "x", "does not occur in the note; read the note again"),
        ("b a b\n", "b", "occurs 2 times in the note; include more"),
        # Overlapping occurrences are two as well.
        ("aaa\n", "aa", "occurs 2 times"),
        ("a b c\n", "", "must not be empty"),
    ],
)
def test_edit_refuses_anything_but_exactly_one_occurrence(
    current: str, old: str, message: str
) -> None:
    with pytest.raises(InvalidRequest, match=message):
        edited(current, old, "new")


def test_edit_that_changes_nothing_is_refused() -> None:
    with pytest.raises(InvalidRequest, match="changes nothing"):
        edited("a b c\n", "b", "b")


def test_append_puts_the_text_on_a_line_of_its_own() -> None:
    assert appended("# Day\n", "- one") == "# Day\n- one\n"
    assert appended("# Day", "- one\n") == "# Day\n- one\n"
    assert appended("", "- one") == "- one\n"
    with pytest.raises(InvalidRequest, match="must not be empty"):
        appended("# Day\n", "  \n")


@pytest.mark.parametrize(
    ("parts", "daily"),
    [
        (("alice", "memory", "daily", "2026-10-10.md"), True),
        (("alice", "memory", "global", "daily", "2026-10-10.md"), True),
        (("alice", "memory", "daily.md"), False),
        (("alice", "memory", "daily-notes", "x.md"), False),
    ],
)
def test_daily_notes_are_those_in_a_daily_folder(parts: tuple[str, ...], daily: bool) -> None:
    assert is_daily(parts) is daily


def test_the_index_is_always_the_topics() -> None:
    spot = index_spot(("alice", "memory", "homelab", "domain", "dns.md"))
    assert (spot.name, spot.topic) == ("MEMORY.md", ("alice", "memory", "homelab"))
    assert index_spot(("alice", "memory", "x.md")).topic == ("alice", "memory")
    wiki = index_spot(("alice", "wiki", "networking", "deep", "dhcp.md"))
    assert (wiki.name, wiki.topic) == ("_index.md", ("alice", "wiki", "networking"))
    with pytest.raises(InvalidRequest, match="wiki notes go into a topic folder"):
        index_spot(("alice", "wiki", "x.md"))


def test_index_lines_follow_the_format_of_their_index() -> None:
    note = ("alice", "memory", "homelab", "domain", "dns.md")
    assert index_entry("memory", note, ("alice", "memory", "homelab"), "DNS notes") == (
        "- [domain/dns.md](domain/dns.md) — DNS notes"
    )
    spaced = ("alice", "memory", "homelab", "my note.md")
    assert index_entry("memory", spaced, ("alice", "memory", "homelab"), "d") == (
        "- [my note.md](<my note.md>) — d"
    )
    wiki = ("alice", "wiki", "networking", "dhcp leases.md")
    assert index_entry("wiki", wiki, ("alice", "wiki", "networking"), "leases") == (
        "- [[dhcp leases]] — leases"
    )
    assert master_entry(("alice", "wiki", "networking"), "the home network") == (
        "- [[wiki/networking/_index|networking]] — the home network"
    )


def test_a_new_index_is_headed_by_its_topic() -> None:
    assert new_index(index_spot(("alice", "memory", "homelab", "x.md"))) == "# homelab\n"
    assert new_index(index_spot(("household", "memory", "x.md"))) == "# household memory\n"


INDEX = """\
# Index

## Tools

- [a.md](a.md) — a

## Places
- [b.md](b.md) — b

```
## Not a section
```

## Empty
"""


def test_sections_are_the_level_two_headings_outside_code() -> None:
    assert section_names(INDEX) == ["Tools", "Places", "Empty"]


def test_an_entry_goes_to_the_end_of_its_section() -> None:
    result = with_entry(INDEX, "Tools", "- [c.md](c.md) — c")
    assert "- [a.md](a.md) — a\n- [c.md](c.md) — c\n\n## Places" in result
    # The code block belongs to its section, so the entry comes after it.
    places = with_entry(INDEX, "Places", "- x")
    assert "## Not a section\n```\n- x\n\n## Empty" in places


def test_a_section_matching_ignoring_case_is_used() -> None:
    result = with_entry(INDEX, "tOOLS", "- [c.md](c.md) — c")
    assert section_names(result) == ["Tools", "Places", "Empty"]
    assert "- [a.md](a.md) — a\n- [c.md](c.md) — c\n" in result


def test_an_entry_in_an_empty_section_follows_a_blank_line() -> None:
    assert with_entry(INDEX, "Empty", "- e").endswith("## Empty\n\n- e\n")


def test_an_unknown_section_is_added_at_the_end() -> None:
    assert with_entry(INDEX, "Recipes", "- r").endswith("## Empty\n\n## Recipes\n\n- r\n")
    # Blank lines at the end stay: no line but the new ones changes.
    result = with_entry(INDEX + "\n\n", "Recipes", "- r")
    assert result.endswith("## Empty\n\n\n## Recipes\n\n- r\n")
    assert section_names(result)[-1] == "Recipes"


def test_an_index_keeps_its_line_endings() -> None:
    crlf = INDEX.replace("\n", "\r\n")
    result = with_entry(crlf, "Tools", "- c")
    assert "\n" not in result.replace("\r\n", "")


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (None, "needs index_section"),
        ("  ## ", "needs index_section"),
        ("two\nlines", "one line of at most"),
        ("x" * 101, "one line of at most"),
    ],
)
def test_a_malformed_section_names_the_existing_ones(section: str | None, message: str) -> None:
    with pytest.raises(InvalidRequest, match=message) as caught:
        checked_section(section, ["Tools", "Places"])
    assert "Tools; Places" in str(caught.value)


def test_a_section_loses_its_heading_marks() -> None:
    assert checked_section("## Tools ", []) == "Tools"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (None, "needs index_line"),
        ("", "needs index_line"),
        ("two\nlines", "one line of at most 300"),
        ("x" * (MAX_INDEX_LINE_CHARS + 1), "one line of at most 300"),
    ],
)
def test_a_malformed_index_line_is_refused(line: str | None, message: str) -> None:
    with pytest.raises(InvalidRequest, match=message):
        checked_line(line, ["Tools"])


def test_an_index_line_of_the_longest_length_passes() -> None:
    assert checked_line("x" * MAX_INDEX_LINE_CHARS, []) == "x" * MAX_INDEX_LINE_CHARS


def test_commit_message_for_one_file_and_for_several() -> None:
    assert commit_message("memex-mcp", [("append", "alice/memory/a.md")]) == (
        "memex-mcp: append alice/memory/a.md"
    )
    several = commit_message(
        "memex-mcp",
        [
            ("create", "alice/wiki/net/dhcp.md"),
            ("create", "alice/wiki/net/_index.md"),
            ("update", "alice/wiki/_master-index.md"),
        ],
    )
    assert several == (
        "memex-mcp: 3 files in alice/wiki\n\n"
        "create alice/wiki/net/dhcp.md\n"
        "create alice/wiki/net/_index.md\n"
        "update alice/wiki/_master-index.md"
    )
    apart = commit_message("memex-mcp", [("update", "alice/x.md"), ("update", "bob/y.md")])
    assert apart.splitlines()[0] == "memex-mcp: 2 files"


# Characters Python's splitlines() breaks at besides "\n" and "\r", and format characters.
LINE_BREAKERS = ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
FORMAT_CHARACTERS = ["\u202e", "\u200b", "\u2066"]


@pytest.mark.parametrize("char", ["\t", *LINE_BREAKERS, *FORMAT_CHARACTERS], ids=repr)
def test_index_texts_refuse_control_format_and_line_breaking_characters(char: str) -> None:
    text = f"before{char}after"
    with pytest.raises(InvalidRequest, match="index_line must be one line"):
        checked_line(text, [])
    with pytest.raises(InvalidRequest, match="index_section must be one line"):
        checked_section(text, [])
    with pytest.raises(InvalidRequest, match="topic_description must be one line"):
        checked_topic(text)


@pytest.mark.parametrize("text", ["see @notes.md", "@notes.md", "(@x)", "a @@b"])
def test_index_texts_refuse_an_at_sign_starting_a_word(text: str) -> None:
    for check in (lambda: checked_line(text, []), lambda: checked_section(text, [])):
        with pytest.raises(InvalidRequest, match="'@' at the start of a word"):
            check()
    with pytest.raises(InvalidRequest, match="'@' at the start of a word"):
        checked_topic(text)


def test_an_at_sign_inside_a_word_is_fine() -> None:
    assert checked_line("mail alice@example.org", []) == "mail alice@example.org"


def test_index_texts_refuse_an_html_comment() -> None:
    for check in (
        lambda: checked_line("a <!-- b", []),
        lambda: checked_section("a <!-- b", []),
        lambda: checked_topic("a <!-- b"),
    ):
        with pytest.raises(InvalidRequest, match="must not contain '<!--'"):
            check()


def test_a_topic_description_is_required_and_one_line() -> None:
    with pytest.raises(InvalidRequest, match="needs topic_description"):
        checked_topic(None)
    with pytest.raises(InvalidRequest, match="at most 300 characters"):
        checked_topic("x" * 301)
    assert checked_topic(" the home network ") == "the home network"


@pytest.mark.parametrize("char", LINE_BREAKERS, ids=repr)
def test_an_entry_changes_no_byte_of_the_other_lines(char: str) -> None:
    index = f"# Index\n\n## Tools\n\n- [a.md](a.md) — one{char}## not a heading\n- tail{char}\n"
    result = with_entry(index, "Tools", "- [c.md](c.md) — c")
    assert result == index + "- [c.md](c.md) — c\n"
    assert section_names(result) == ["Tools"]


def test_the_master_line_goes_to_the_end_of_the_first_section() -> None:
    master = "# Wiki\n\n## Topics\n\n- [[wiki/a/_index|a]] — a\n\n## Other\n\n- x\n"
    assert with_topic(master, "- new") == (
        "# Wiki\n\n## Topics\n\n- [[wiki/a/_index|a]] — a\n- new\n\n## Other\n\n- x\n"
    )
    assert with_topic("# Wiki\n\n- [[a]]\n", "- new") == "# Wiki\n\n- [[a]]\n- new\n"


@pytest.mark.parametrize(
    ("line", "problem"),
    [
        ("- [a\u202eb.md](a\u202eb.md) — x", "control or format characters"),
        ("# topic\x0cname", "control or format characters"),
        ("- [x <!-- y.md](<x <!-- y.md>) — x", "'<!--'"),
        ("- [x @y/n.md](<x @y/n.md>) — x", "'@' at the start of a word"),
    ],
)
def test_a_composed_index_line_is_checked(line: str, problem: str) -> None:
    with pytest.raises(InvalidRequest, match=re.escape(problem)):
        check_composed(["- fine", line])
    check_composed(["- [a.md](a.md) — mail alice@example.org", "# topic", "## Section"])
