#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

CHANGELOG_PATH = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"

SECTION_HEADER_RE = re.compile(
    r"^## \[(?P<version>\d+\.\d+\.\d+)\] - (?P<date>\d{4}-\d{2}-\d{2})\s*$"
)
NAME_RE = re.compile(r'^\s*name\s*=\s*"(?P<name>[^"]+)"')
VERSION_RE = re.compile(r'^\s*version\s*=\s*"(?P<version>\d+\.\d+\.\d+)"')


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=False)


def get_metadata_from_pyproject(pyproject_path: Path = PYPROJECT_PATH) -> Tuple[Optional[str], Optional[str]]:
    name: Optional[str] = None
    version: Optional[str] = None

    try:
        for raw in pyproject_path.read_text(encoding="utf-8").splitlines():
            if name is None:
                match = NAME_RE.search(raw)
                if match:
                    name = match.group("name")
            if version is None:
                match = VERSION_RE.search(raw)
                if match:
                    version = match.group("version")
            if name and version:
                break
    except FileNotFoundError:
        return None, None

    return name, version


def git_is_clean() -> Tuple[bool, str]:
    try:
        dirty_wc = subprocess.run(["git", "diff", "--quiet", "--exit-code"]).returncode != 0
        dirty_staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--exit-code"]).returncode != 0
        if dirty_wc or dirty_staged:
            status = subprocess.run(["git", "status"], capture_output=True, text=True).stdout
            return False, status
        return True, ""
    except Exception as exc:
        return False, str(exc)


def git_tag_exists(tag: str) -> bool:
    return subprocess.run(["git", "rev-parse", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def extract_latest_changelog(changelog_text: str) -> Tuple[str, str]:
    lines = changelog_text.splitlines()
    sections: List[Tuple[str, int]] = []
    for idx, line in enumerate(lines):
        match = SECTION_HEADER_RE.match(line)
        if match:
            sections.append((match.group("version"), idx))
    if not sections:
        raise ValueError("No versioned sections found in CHANGELOG")

    latest_version, start = sections[0]
    end = len(lines)
    for _, section_start in sections[1:]:
        if section_start > start:
            end = section_start
            break

    section_text = "\n".join(lines[start:end]).strip()
    return latest_version, section_text


def extract_latest_changelog_from_file(path: Path = CHANGELOG_PATH) -> Tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    return extract_latest_changelog(text)


def ensure_gh_cli() -> None:
    if subprocess.run(["which", "gh"], stdout=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("GitHub CLI (gh) is not installed. See https://cli.github.com/")


def main() -> int:
    print("🚀 Starting publish process...")

    print("🔍 Checking for uncommitted changes...")
    clean, status = git_is_clean()
    if not clean:
        print("❌ Error: Git working directory is dirty. Please commit or stash your changes before publishing.")
        if status:
            print(status)
        return 1
    print("✅ Git working directory is clean.")

    name, version = get_metadata_from_pyproject()
    if not version:
        print("❌ Error: Could not extract version from pyproject.toml")
        return 1
    package_label = name or "package"
    print(f"🔖 Detected version: v{version} for {package_label}")

    tag = f"v{version}"
    print(f"🏷️ Checking for git tag {tag}...")
    if not git_tag_exists(tag):
        print(f"❌ Error: Git tag {tag} does not exist. Please create it before publishing.")
        print(f"Example: git tag {tag} && git push origin tag {tag}")
        return 1
    print(f"✅ Git tag {tag} exists.")

    print(f"🚀 Creating GitHub Release {tag}...")
    ensure_gh_cli()

    try:
        latest_version, section = extract_latest_changelog_from_file()
    except Exception as exc:
        print(f"⚠️ Warning: Failed to extract latest changelog section: {exc}")
        latest_version, section = version, ""

    if latest_version != version:
        print(
            f"⚠️ Warning: Latest changelog version {latest_version} does not match pyproject version {version}"
        )

    import tempfile

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
        tf.write(section)
        notes_path = tf.name

    release_title = f"{package_label} v{version}"
    run(["gh", "release", "create", tag, "--title", release_title, "--notes-file", notes_path])

    print("🎉 Publish process completed successfully!")
    print(f"✨ GitHub Release {tag} created successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
