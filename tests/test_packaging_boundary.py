from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


REMOVED_COMPOSITES = {"Interview", "Gate", "Choose", "Respond"}


@dataclass(frozen=True)
class _CodeBlock:
    code: str
    start: int
    end: int


@dataclass(frozen=True)
class _Heading:
    title: str
    level: int
    start: int


def _python_code_blocks(text: str) -> list[_CodeBlock]:
    return [
        _CodeBlock(code=match.group(1), start=match.start(), end=match.end())
        for match in re.finditer(r"```python\n(.*?)```", text, flags=re.DOTALL)
    ]


def _markdown_headings(text: str) -> list[_Heading]:
    return [
        _Heading(
            title=match.group(2).strip(),
            level=len(match.group(1)),
            start=match.start(),
        )
        for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, flags=re.MULTILINE)
    ]


def _section_bounds(text: str, heading_title: str) -> tuple[int, int]:
    headings = _markdown_headings(text)
    for index, heading in enumerate(headings):
        if heading.title != heading_title:
            continue
        end = len(text)
        for later in headings[index + 1 :]:
            if later.level <= heading.level:
                end = later.start
                break
        return heading.start, end
    raise AssertionError(f"Missing heading: {heading_title}")


def _imported_iriai_compose_names(block: str) -> set[str]:
    names: set[str] = set()
    for match in re.finditer(
        r"from\s+iriai_compose\s+import\s*(\((.*?)\)|([^\n]+))",
        block,
        flags=re.DOTALL,
    ):
        raw = match.group(2) if match.group(2) is not None else match.group(3) or ""
        for part in raw.replace("\n", " ").split(","):
            name = part.strip()
            if name:
                names.add(name)
    return names


def _uses_removed_composites(code: str) -> bool:
    return bool(
        re.search(
            r"\b(?:Interview|Gate|Choose|Respond)\s*\(",
            code,
        )
        or re.search(r"^class\s+(?:Interview|Gate|Choose|Respond)\b", code, flags=re.MULTILINE)
    )


def _readme_block_is_explicitly_local(readme: str, block: _CodeBlock) -> bool:
    context = readme[max(0, block.start - 400) : block.start]
    return bool(
        re.search(
            r"assumes\s+your\s+application\s+defines\s+local",
            context,
            flags=re.IGNORECASE,
        )
        and "Ask" in context
    )


def test_docs_respect_installed_api_boundary() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    tutorial = (REPO_ROOT / "TUTORIAL.md").read_text(encoding="utf-8")
    spec = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")

    assert "copy or import the example-owned implementations" not in readme
    assert "copy or vendor the example-owned implementations" in readme
    assert "| `examples/_composite_tasks.py` |" not in readme
    assert "source-only" in readme or "not part of the installed package" in readme

    for doc_text in (readme, tutorial, spec):
        for block in _python_code_blocks(doc_text):
            assert not (_imported_iriai_compose_names(block.code) & REMOVED_COMPOSITES)

    spec_allowed_start, spec_allowed_end = _section_bounds(
        spec,
        "Example Composite Task Patterns (application-owned)",
    )
    for block in _python_code_blocks(spec):
        if _uses_removed_composites(block.code):
            assert spec_allowed_start <= block.start < spec_allowed_end

    tutorial_helper_start, _ = _section_bounds(
        tutorial,
        "Step 3: Define App-Owned Interaction Helpers",
    )
    for block in _python_code_blocks(tutorial):
        if _uses_removed_composites(block.code):
            assert block.start >= tutorial_helper_start

    readme_use_cases_start, readme_use_cases_end = _section_bounds(
        readme,
        "Example Use Cases",
    )
    for block in _python_code_blocks(readme):
        if _uses_removed_composites(block.code):
            assert readme_use_cases_start <= block.start < readme_use_cases_end
            assert _readme_block_is_explicitly_local(readme, block)


def test_wheel_does_not_ship_examples_helpers(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(dist_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_path = next(dist_dir.glob("iriai_compose-*.whl"))
    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()

    assert all(not name.startswith("examples/") for name in names)
