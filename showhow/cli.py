from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="ShowHow CLI")
    sub = parser.add_subparsers(dest="command")

    web = sub.add_parser("web", help="Launch browser-based recording UI (recommended)")
    web.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    web.add_argument(
        "--port", type=int, default=18090, help="Bind port (default: 18090)"
    )
    web.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )
    sub.add_parser("serve", help="Run optional ShowHow MCP stdio server")
    sub.add_parser("status", help="Debug runtime status")
    sub.add_parser("recorder", help="Run recorder service directly")
    sub.add_parser(
        "doctor", help="Check local runtime requirements (permissions, ffmpeg, paths)"
    )
    rec = sub.add_parser(
        "record",
        help="Single-terminal interactive record flow (start -> stop -> optional generate)",
    )
    rec.add_argument("--topic", default=None, help="Optional recording topic")
    rec.add_argument("--folder", default=None, help="Optional output root folder")
    rec.add_argument("--base-name", default=None, help="Optional session base name")
    rec.add_argument(
        "--generate",
        action="store_true",
        help="Generate tutorial immediately after stopping",
    )
    rec.add_argument(
        "--model", default=None, help="OpenAI model for caption, planner, and critic"
    )
    start = sub.add_parser(
        "start", help="Start recorder session via local recorder API"
    )
    start.add_argument("--topic", default=None, help="Optional recording topic")
    start.add_argument("--folder", default=None, help="Optional output root folder")
    start.add_argument("--base-name", default=None, help="Optional session base name")
    sub.add_parser("stop", help="Stop recorder session via local recorder API")
    sub.add_parser("rec-status", help="Get recorder status via local recorder API")
    sag = sub.add_parser(
        "stop-and-generate", help="Stop recorder and generate tutorial in one command"
    )
    sag.add_argument(
        "--session-id", default=None, help="Optional override session id for generation"
    )
    sag.add_argument(
        "--data-path", default=None, help="Optional explicit session data path"
    )
    sag.add_argument(
        "--model", default=None, help="OpenAI model for caption, planner, and critic"
    )
    gen = sub.add_parser("generate", help="Generate tutorial for an existing session")
    gen.add_argument(
        "--session-id", required=True, help="Session id (folder name under record root)"
    )
    gen.add_argument(
        "--data-path", default=None, help="Optional explicit session data path"
    )
    gen.add_argument(
        "--model", default=None, help="OpenAI model for caption, planner, and critic"
    )
    args = parser.parse_args()

    if args.command == "web":
        from showhow.web_server.server import run as web_run

        web_run(host=args.host, port=args.port, open_browser=not args.no_browser)
        return

    if args.command in {"serve", "status"}:
        from showhow.runtime.server import main as runtime_main

        # Re-dispatch to runtime subcommand parser.
        import sys

        sys.argv = [sys.argv[0], args.command]
        runtime_main()
        return

    if args.command == "recorder":
        from showhow.recorder_service.main import run as recorder_run

        recorder_run()
        return

    if args.command == "doctor":
        from showhow.recorder_service.diagnostics import run_diagnostics

        print(json.dumps(run_diagnostics(), ensure_ascii=False, indent=2))
        return

    if args.command == "record":
        from showhow.runtime.service import ShowHowService

        service = ShowHowService()
        service.startup()
        try:
            started = service.start_recording(
                topic=args.topic, folder=args.folder, base_name=args.base_name
            )
            session_id = started.get("session_id")
            data_path = started.get("data_path")
            print(
                json.dumps(
                    {
                        "status": "recording",
                        "session_id": session_id,
                        "data_path": data_path,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            input("Recording... press Enter to stop.\n")
            stopped = service.stop_recording()
            if not args.generate:
                print(json.dumps({"stop": stopped}, ensure_ascii=False, indent=2))
                return
            result = service.generate_tutorial_with_options(
                session_id=stopped.get("session_id"),
                data_path=stopped.get("data_path"),
                caption_model=args.model,
                planner_model=args.model,
                critic_model=args.model,
            )
            print(
                json.dumps(
                    {"stop": stopped, "tutorial": result}, ensure_ascii=False, indent=2
                )
            )
            return
        finally:
            service.shutdown()

    if args.command == "generate":
        from showhow.runtime.service import ShowHowService

        service = ShowHowService()
        result = service.generate_tutorial_with_options(
            session_id=args.session_id,
            data_path=args.data_path,
            caption_model=args.model,
            planner_model=args.model,
            critic_model=args.model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command in {"start", "stop", "rec-status"}:
        from showhow.runtime.recorder_client import RecorderHTTPClient

        client = RecorderHTTPClient()
        if args.command == "start":
            result = client.start_recording(
                topic=args.topic, folder=args.folder, base_name=args.base_name
            )
        elif args.command == "stop":
            result = client.stop_recording()
        else:
            result = client.status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "stop-and-generate":
        from showhow.runtime.service import ShowHowService

        service = ShowHowService()
        stopped = service.stop_recording()
        session_id = args.session_id or stopped.get("session_id")
        result = service.generate_tutorial_with_options(
            session_id=session_id,
            data_path=args.data_path or stopped.get("data_path"),
            caption_model=args.model,
            planner_model=args.model,
            critic_model=args.model,
        )
        print(
            json.dumps(
                {"stop": stopped, "tutorial": result}, ensure_ascii=False, indent=2
            )
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
