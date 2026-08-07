from __future__ import annotations


class JankiError(Exception):
    """Base class for every error janki reports to the user.

    ``cli.main`` catches this one type, so a new command only needs its error
    class to subclass this — there is no registry to keep in sync. It bases
    ``Exception`` (not ``RuntimeError``) deliberately: broad ``except
    RuntimeError`` blocks in future code must never swallow a user-facing
    janki error before the CLI can format it.
    """
