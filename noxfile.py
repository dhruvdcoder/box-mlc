"""Nox sessions."""
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import nox
from nox import Session, session

package = "box_mlc"
python_versions = ['3.7', '3.8', '3.9']
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = (
    "pre-commit",
    "mypy",
    "tests", "xdoctest",  "docs-build", )


def activate_virtualenv_in_precommit_hooks(session: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        session: The Session object.
    """

    if session.bin is None:
        return

    virtualenv = session.env.get("VIRTUAL_ENV")

    if virtualenv is None:
        return

    hookdir = Path(".git") / "hooks"

    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        text = hook.read_text()
        bindir = repr(session.bin)[1:-1]  # strip quotes

        if not (
            Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text
        ):
            continue

        lines = text.splitlines()

        if not (lines[0].startswith("#!") and "python" in lines[0].lower()):
            continue

        header = dedent(
            f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """
        )

        lines.insert(1, header)
        hook.write_text("\n".join(lines))


@session(name="pre-commit", python="3.8")
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    session.install('-r', 'lint_requirements.txt')
    session.run("pre-commit", *args)

    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)

# DP: Unmodified
@session(python="3.9")
def safety(session: Session) -> None:
    """Scan dependencies for insecure packages."""
    requirements = session.poetry.export_requirements()
    session.install("safety")
    session.run("safety", "check", "--full-report", f"--file={requirements}")


@session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests", "docs/*.py"]
    session.install(".")
    # read the mypy version from lint_requirements.txt
    with open('lint_requirements.txt') as f:
        for line in f:
            if "mypy" in line:
                mypy_version = line.strip()
    session.install(mypy_version)
    session.run("mypy", *args)

    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install(".")
    session.install('-r', 'test_requirements.txt')
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@session
def coverage(session: Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]

    session.install("coverage[toml]")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@session(python=python_versions)
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    session.install(".")
    session.install("pytest", "typeguard", "pygments")
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@session(python=python_versions)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.install(".")
    session.install("xdoctest[colors]")
    session.run("python", "-m", "xdoctest", package, *args)

def _clear_docs_build_dir(session: Session, d: str)->None:
    build_dir = Path(d)

    if build_dir.exists():
        session.log(f"Clearing {build_dir}")
        shutil.rmtree(build_dir)


@session(name="docs-build", python="3.8")
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["-b", "html" ,"docs", "docs/_build"]
    session.install('-r', 'doc_requirements.txt')
    _clear_docs_build_dir(session, args[-1])
    session.run('sphinx-build', *args)
