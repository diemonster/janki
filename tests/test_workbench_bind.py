"""Browser-independent checks for the workbench's listening boundary."""

from __future__ import annotations

from typing import Any, cast

from japanese_anki.workbench import make_server


def test_workbench_server_binds_only_to_ipv4_loopback() -> None:
    server = make_server(cast(Any, object()))
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
