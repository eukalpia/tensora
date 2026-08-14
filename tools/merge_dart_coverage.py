#!/usr/bin/env python3
"""Merge Dart LCOV reports and enforce complete owned-source line coverage."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--source-fragment",
        default="/packages/tensora/lib/",
        help="Only LCOV source paths containing this fragment are owned production sources.",
    )
    return parser.parse_args()


def parse_lcov(path: Path, source_fragment: str) -> dict[str, dict[int, int]]:
    records: dict[str, dict[int, int]] = defaultdict(dict)
    current_source: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("SF:"):
            candidate = raw_line[3:]
            current_source = candidate if source_fragment in candidate else None
            continue
        if raw_line.startswith("DA:") and current_source is not None:
            fields = raw_line[3:].split(",")
            line_number = int(fields[0])
            hit_count = int(fields[1])
            previous = records[current_source].get(line_number, 0)
            records[current_source][line_number] = previous + hit_count
            continue
        if raw_line == "end_of_record":
            current_source = None

    return records


def ignored_lines(source: Path) -> set[int]:
    if not source.is_file():
        return set()

    ignored: set[int] = set()
    inside_block = False
    lines = source.read_text(encoding="utf-8").splitlines()

    for index, text in enumerate(lines, start=1):
        if "coverage:ignore-start" in text:
            inside_block = True
            ignored.add(index)
            continue
        if inside_block:
            ignored.add(index)
            if "coverage:ignore-end" in text:
                inside_block = False
            continue
        if "coverage:ignore-line" in text:
            ignored.add(index)
        if "coverage:ignore-next-line" in text and index < len(lines):
            ignored.add(index + 1)

    if inside_block:
        raise RuntimeError(f"Unclosed coverage:ignore-start in {source}")

    return ignored


def merge_reports(
    coverage_dir: Path, source_fragment: str
) -> dict[str, dict[int, int]]:
    merged: dict[str, dict[int, int]] = defaultdict(dict)
    reports = sorted(
        path for path in coverage_dir.glob("*.info") if path.name != "merged.info"
    )
    if not reports:
        raise RuntimeError(f"No LCOV reports found in {coverage_dir}")

    for report in reports:
        for source, lines in parse_lcov(report, source_fragment).items():
            target = merged[source]
            for line_number, hit_count in lines.items():
                target[line_number] = target.get(line_number, 0) + hit_count

    if not merged:
        raise RuntimeError("LCOV reports contained no owned production Dart sources")
    return merged


def write_merged_lcov(path: Path, merged: dict[str, dict[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for source in sorted(merged):
            handle.write(f"SF:{source}\n")
            for line_number in sorted(merged[source]):
                handle.write(f"DA:{line_number},{merged[source][line_number]}\n")
            handle.write("end_of_record\n")


def main() -> int:
    args = parse_args()
    merged = merge_reports(args.coverage_dir, args.source_fragment)

    total = 0
    covered = 0
    excluded = 0
    summary_lines = ["# Dart production coverage", ""]
    all_uncovered: list[tuple[str, int]] = []

    for source_name in sorted(merged):
        source = Path(source_name)
        ignored = ignored_lines(source)
        executable = {
            line_number: hit_count
            for line_number, hit_count in merged[source_name].items()
            if line_number not in ignored
        }
        file_total = len(executable)
        file_covered = sum(hit_count > 0 for hit_count in executable.values())
        file_excluded = len(merged[source_name]) - file_total
        total += file_total
        covered += file_covered
        excluded += file_excluded

        uncovered = sorted(
            line_number
            for line_number, hit_count in executable.items()
            if hit_count == 0
        )
        all_uncovered.extend((source_name, line_number) for line_number in uncovered)
        percent = 100.0 if file_total == 0 else file_covered * 100.0 / file_total
        summary_lines.append(
            f"- `{source_name}`: {file_covered}/{file_total} = {percent:.4f}%"
            + (f" ({file_excluded} explicitly excluded)" if file_excluded else "")
        )

    if total == 0:
        raise RuntimeError("Merged coverage contained zero executable production lines")

    percent = covered * 100.0 / total
    summary_lines.extend(
        [
            "",
            f"**Total:** {covered}/{total} = {percent:.4f}%",
            f"**Explicitly excluded executable lines:** {excluded}",
        ]
    )

    if all_uncovered:
        summary_lines.extend(["", "## Uncovered lines", ""])
        for source_name, line_number in all_uncovered:
            summary_lines.append(f"- `{source_name}:{line_number}`")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_merged_lcov(args.output, merged)

    print("\n".join(summary_lines))
    if all_uncovered:
        print(
            f"Coverage gate failed: {len(all_uncovered)} production lines remain uncovered.",
            file=sys.stderr,
        )
        return 1

    print("Coverage gate passed: 100.0000% of included production Dart lines executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
