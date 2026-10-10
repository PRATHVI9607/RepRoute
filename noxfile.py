"""Quality gates (PRD §12.1). Sessions reuse the project's uv environment."""

import nox

nox.options.default_venv_backend = "none"
nox.options.sessions = ["lint", "types", "tests"]
STRICT = ["src/kai/contracts", "src/kai/certify", "src/kai/select"]


@nox.session
def lint(session: nox.Session) -> None:
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")
    session.run("lint-imports")
    session.run("python", "scripts/check_no_l0_splits.py")


@nox.session
def types(session: nox.Session) -> None:
    session.run("mypy", *STRICT)
    session.run("mypy", "src/kai")


@nox.session
def tests(session: nox.Session) -> None:
    session.run("pytest", *session.posargs)
    # the certificate and the selectors are the correctness core: hold them to a higher bar
    session.run(
        "coverage", "report", "--include=src/kai/certify/*,src/kai/select/*", "--fail-under=85"
    )
    session.run("coverage", "report", "--fail-under=70")


@nox.session
def edge_smoke(session: nox.Session) -> None:
    """The edge plane must import and run without torch / sklearn / pandas / scipy / lightgbm."""
    session.run("python", "scripts/verify_env.py", "--edge-only")
