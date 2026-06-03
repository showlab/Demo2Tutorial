from __future__ import annotations

import argparse
from typing import Any

from .service import ShowHowService


def _build_fastmcp_server(service: ShowHowService) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError(
            "MCP SDK not available. Install MCP server dependency to run `showhow serve`."
        ) from exc

    app = FastMCP("showhow")

    @app.tool(name="start_recording")
    def start_recording(topic: str = "") -> dict:
        safe_topic = topic.strip() or None
        return service.start_recording(topic=safe_topic)

    @app.tool(name="stop_recording")
    def stop_recording() -> dict:
        return service.stop_recording()

    @app.tool(name="generate_tutorial")
    def generate_tutorial(session_id: str) -> dict:
        return service.generate_tutorial(session_id=session_id)

    @app.tool(name="stop_and_generate")
    def stop_and_generate() -> dict:
        return service.stop_and_generate()

    return app


def run_stdio() -> None:
    service = ShowHowService()
    service.startup()
    try:
        mcp_app = _build_fastmcp_server(service)
        mcp_app.run()
    finally:
        service.shutdown()


def run_status() -> None:
    service = ShowHowService()
    service.startup()
    try:
        print(service.status())
    finally:
        service.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShowHow runtime server with optional MCP stdio support"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="Run optional MCP stdio server")
    sub.add_parser("status", help="Debug runtime status")

    args = parser.parse_args()
    if args.command == "serve":
        run_stdio()
    elif args.command == "status":
        run_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
