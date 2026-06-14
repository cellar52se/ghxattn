from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from ghxattn.dressing.loader import load_config
from ghxattn.dressing.schema import RootConfig
from ghxattn.mill import commands

COMMANDS: dict[str, Callable[[RootConfig], object]] = {
    "weave": commands.weave,
    "inspect": commands.inspect,
    "forecast": commands.forecast,
    "export": commands.export,
}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = list(sys.argv[1:] if argv is None else argv)
    command = "weave"
    overrides: list[str] = []
    if args:
        if "=" in args[0]:
            overrides = args
        else:
            command = args[0]
            overrides = args[1:]
    if command not in COMMANDS:
        raise SystemExit(f"unknown command {command}; choose from {sorted(COMMANDS)}")
    cfg = load_config(overrides)
    COMMANDS[command](cfg)


if __name__ == "__main__":
    main()
