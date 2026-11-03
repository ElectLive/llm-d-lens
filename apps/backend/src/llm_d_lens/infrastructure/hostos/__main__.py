"""Command line for shell callers: ``python -m llm_d_lens.infrastructure.hostos``.

Exit codes: ``0`` success or true; ``1`` false, failed, or unsupported on this
host (stderr explains an unsupported operation); ``2`` usage or internal error.
Predicate commands such as ``port-in-use`` and ``check`` use ``1`` for both
false and unsupported, so shell callers must gate on ``check`` first.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict

from . import Host, SupportStatus, UnsupportedHostOperation, check_host, current_host


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llm_d_lens.infrastructure.hostos")
    subcommands = parser.add_subparsers(dest="command", required=True)

    check = subcommands.add_parser("check", help="check host support")
    check.add_argument("--installer", metavar="OS_ID")

    info = subcommands.add_parser("info", help="print host facts and support")
    info.add_argument("--json", action="store_true")

    packages = subcommands.add_parser("packages", help="system package operations")
    package_commands = packages.add_subparsers(dest="packages_command", required=True)
    package_commands.add_parser("missing", help="print missing package names")
    show = package_commands.add_parser("show", help="print install commands")
    show.add_argument("packages", nargs="+")
    install = package_commands.add_parser("install", help="install packages")
    install.add_argument("packages", nargs="+")
    hint = package_commands.add_parser("hint", help="print a manual install command")
    hint.add_argument("packages", nargs="+")

    port_in_use = subcommands.add_parser("port-in-use", help="exit 0 when the port is in use")
    port_in_use.add_argument("port", type=int)
    free_port = subcommands.add_parser("free-port", help="print the first free port")
    free_port.add_argument("port", type=int)
    port_owner = subcommands.add_parser("port-owner", help="describe the process holding a port")
    port_owner.add_argument("port", type=int)
    subcommands.add_parser("lan-ip", help="print the first LAN address")
    tool_platform = subcommands.add_parser("tool-platform", help="print a tool artifact platform")
    tool_platform.add_argument("--style", required=True, choices=("node", "go"))
    return parser


def _info(host: Host) -> dict[str, object]:
    definition = host.definition
    return {
        "facts": asdict(host.facts),
        "definition": (
            {
                "id": definition.id,
                "display_name": definition.display_name,
                "installer": definition.installer,
            }
            if definition is not None
            else None
        ),
        "support": asdict(host.support),
        "capabilities": host.capabilities,
    }


def _print_info(host: Host, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_info(host), indent=2, sort_keys=True))
        return
    facts = host.facts
    definition = host.definition
    status = host.support
    for name in ("family", "arch", "machine", "distro_id", "version", "pretty_name"):
        print(f"{name}: {getattr(facts, name)}")
    print(f"in_container: {facts.in_container}")
    print(f"in_wsl: {facts.in_wsl}")
    if definition is None:
        print("definition: none")
    else:
        print(f"definition: {definition.id}")
        print(f"display_name: {definition.display_name}")
        print(f"installer: {definition.installer}")
    print(f"supported: {status.supported}")
    print(f"message: {status.message}")
    for name, value in host.capabilities.items():
        print(f"capability.{name}: {value}")


def _run(args: argparse.Namespace) -> int:
    if args.command == "check":
        status: SupportStatus = check_host(args.installer)
        print(status.message, file=sys.stdout if status.supported else sys.stderr)
        return 0 if status.supported else 1

    host = current_host()

    if args.command == "info":
        _print_info(host, as_json=args.json)
        return 0

    if args.command == "packages":
        if args.packages_command == "missing":
            for package in host.missing_packages():
                print(package)
            return 0
        if args.packages_command == "show":
            problem = host.package_install_problem(args.packages)
            if problem is not None:
                print(problem, file=sys.stderr)
                return 1
            commands = host.package_commands(args.packages)
            print(" && ".join(shlex.join(command) for command in commands))
            return 0
        if args.packages_command == "install":
            return host.install_packages(args.packages)
        print(host.install_hint(args.packages))
        return 0

    if args.command == "port-in-use":
        return 0 if host.port_in_use(args.port) else 1

    if args.command == "free-port":
        for offset in range(200):
            port = args.port + offset
            if not host.port_in_use(port):
                print(port)
                return 0
        print(f"could not find a free port at or after {args.port} (checked 200 ports)", file=sys.stderr)
        return 1

    if args.command == "port-owner":
        print(host.describe_port_owner(args.port))
        return 0

    if args.command == "lan-ip":
        address = host.first_lan_address()
        if address:
            print(address)
        return 0

    print(host.tool_platform(args.style).joined())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except UnsupportedHostOperation as error:
        print(error, file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - CLI boundary maps unexpected errors to exit 2
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
