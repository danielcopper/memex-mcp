"""Chunking and titles."""

from __future__ import annotations

import time

from memex_mcp.markdown import chunk_note, title_of


def test_chunks_follow_headings() -> None:
    text = "intro line\n\n# Title\n\nfirst\n\n## Part\n\nsecond\n"
    chunks = chunk_note(text, 1000)
    assert [(c.heading, c.body) for c in chunks] == [
        ("", "intro line"),
        ("Title", "first"),
        ("Part", "second"),
    ]


def test_headings_inside_code_fences_are_text() -> None:
    text = "# Real\n\n```bash\n# not a heading\necho hi\n```\n"
    chunks = chunk_note(text, 1000)
    assert [c.heading for c in chunks] == ["Real"]
    assert "# not a heading" in chunks[0].body
    assert title_of("```\n# fake\n```\n# Real\n", "fallback") == "Real"


def test_long_sections_split_into_bounded_pieces() -> None:
    paragraphs = "\n\n".join("word " * 60 for _ in range(10))
    chunks = chunk_note(f"# Long\n\n{paragraphs}\n", 500)
    assert len(chunks) > 1
    assert all(len(c.body) <= 500 for c in chunks)
    one_huge = "x" * 1300
    assert [len(c.body) for c in chunk_note(one_huge, 500)] == [500, 500, 300]


def test_empty_section_keeps_its_heading() -> None:
    assert [(c.heading, c.body) for c in chunk_note("# Only a title\n", 100)] == [
        ("Only a title", "")
    ]


def test_title_falls_back_to_the_file_name() -> None:
    assert title_of("no heading here\n## second level\n", "my-note") == "my-note"


def test_closing_hashes_are_dropped_but_not_part_of_the_text() -> None:
    assert title_of("# Title ##\n", "x") == "Title"
    assert title_of("# C#\n", "x") == "C#"
    assert title_of("#\tTabbed  \n", "x") == "Tabbed"
    assert [c.heading for c in chunk_note("## Part #\n\nbody\n", 100)] == ["Part"]


def test_pathological_lines_are_linear() -> None:
    """Lines a writer could commit to stall the index lock (each 100 000 characters)."""
    lines = [
        "# a" + " " * 100_000 + "x",
        "# a" + " #" * 50_000 + "x",
        "#" * 100_000,
        " " * 100_000,
        "# " + "a " * 50_000,
        "x\n" + " " * 100_000 + "y",
        "```" + " " * 100_000,
    ]
    started = time.perf_counter()
    for line in lines:
        title_of(line, "fallback")
        chunk_note(line + "\n\nbody\n", 1500)
    assert time.perf_counter() - started < 1.0
