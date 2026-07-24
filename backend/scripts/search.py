"""Command-line search for quick demonstrations without the web interface."""

from __future__ import annotations

import argparse

from backend.ir_system.service import SearchService


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the bundled BITS research documents.")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11), metavar="1-10")
    parser.add_argument(
        "--answer-mode",
        choices=("auto", "extractive", "gemini"),
        default="extractive",
    )
    args = parser.parse_args()

    result = SearchService().query(
        args.query,
        top_k=args.top_k,
        answer_mode=args.answer_mode,
    )
    print(f"\nAnswer ({result.answer.answer_type}, {result.answer.confidence} confidence)\n")
    print(result.answer.answer)
    print("\nSources\n")
    for source in result.results:
        availability = "" if source.chunk.text_available else " [scan: no extracted text]"
        print(
            f"[{source.rank}] {source.chunk.title}, page {source.chunk.page_start} "
            f"(score {source.score:.3f}){availability}"
        )
    if result.warnings:
        print("\nWarnings")
        for warning in result.warnings:
            print(f"- {warning}")
    print(f"\nCompleted in {result.processing_time_ms:.1f} ms")


if __name__ == "__main__":
    main()
