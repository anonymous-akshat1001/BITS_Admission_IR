from pathlib import Path

import pytest

from backend.scripts.evaluate import load_answerable_cases, load_unanswerable_cases


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_stable_query_ids_survive_csv_reordering(tmp_path) -> None:
    queries = write(
        tmp_path / "queries.csv",
        "query_id,Question,Answer\nQ02,Second question,Second answer\nQ01,First question,First answer\n",
    )
    qrels = write(
        tmp_path / "qrels.csv",
        "query_id,category,relevant_documents\nQ01,one,one.pdf\nQ02,two,two.pdf\n",
    )
    gaps = write(
        tmp_path / "source_gaps.csv",
        "query_id,reason\nQ02,The answer is absent.\n",
    )

    cases = load_answerable_cases(queries, qrels, gaps)

    assert [case.query_id for case in cases] == ["Q02", "Q01"]
    assert [case.category for case in cases] == ["two", "one"]
    assert [case.corpus_supported for case in cases] == [False, True]


def test_query_csv_requires_explicit_ids(tmp_path) -> None:
    queries = write(tmp_path / "queries.csv", "Question,Answer\nQuestion,Answer\n")
    qrels = write(
        tmp_path / "qrels.csv",
        "query_id,category,relevant_documents\nQ01,one,one.pdf\n",
    )
    gaps = write(tmp_path / "source_gaps.csv", "query_id,reason\n")

    with pytest.raises(ValueError, match="must contain Question and Answer"):
        load_answerable_cases(queries, qrels, gaps)


def test_unanswerable_cases_require_a_category(tmp_path) -> None:
    cases = write(
        tmp_path / "unanswerable.csv",
        "query_id,category,Question\nU01,hard_negative,Unsupported research question\n",
    )

    loaded = load_unanswerable_cases(cases)

    assert loaded[0].category == "hard_negative"
