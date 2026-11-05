"""``llmd stacks``: the guides of a source, or one guide's variables and what it applies."""

from __future__ import annotations

import argparse

from .. import catalog, guides
from ..errors import NotFoundError
from . import Context, Handler, add_source_arguments, parse_variables


def register(commands: argparse._SubParsersAction) -> dict[str, Handler]:
    parser = commands.add_parser(
        "stacks", help="list the guides of a source, or show one guide's variables (no cluster access)"
    )
    parser.add_argument(
        "prefix",
        nargs="?",
        default="",
        help="a guide (guides/optimized-baseline) to show its variables, or a path to list the guides below",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="with a guide: the variables to follow its steps with (repeatable)",
    )
    add_source_arguments(parser)
    return {"stacks": command_stacks}


def command_stacks(ctx: Context) -> int:
    source = ctx.source()
    prefix = ctx.args.prefix.strip("/")
    stack = None
    if prefix:
        try:
            found = catalog.find_stack(source, prefix)
            stack = found if found.kind == "guide" else None
        except NotFoundError:
            stack = None
    if stack is not None:
        return _show_guide(ctx, stack, parse_variables(ctx.args.var))
    stacks = catalog.list_stacks(source, ctx.runner, prefix)
    if ctx.args.output != "text":
        ctx.emit({"source": source.describe(), "stacks": [stack.id for stack in stacks]})
        return 0
    revision = f" @ {source.revision[:12]}" if source.revision else ""
    print(f"{source.name}{revision}{' (dirty)' if source.dirty else ''}: {len(stacks)} guide(s)")
    for item in stacks:
        print(f"  {item.id:<56} instance name {catalog.default_name(item)}")
    print("\nA guide's variables and steps: llmd stacks GUIDE")
    return 0


def _show_guide(ctx: Context, stack: catalog.Stack, variables: dict[str, str]) -> int:
    guide = guides.load_guide(stack.path, stack.id)
    followed = guides.follow(guide, stack.source.root, namespace="NAMESPACE", overrides=variables)
    root = stack.source.root
    summary = {
        "stack": stack.id,
        "name": guide.name,
        "variables": guide.variables(),
        "support": guide.support(),
        "applies": [guides.describe_unit(unit, root) for unit in followed.units],
        "steps": followed.steps,
        "findings": [finding.as_dict() for finding in followed.findings],
    }
    if ctx.args.output != "text":
        ctx.emit(summary)
        return 0
    print(f"Guide {stack.id} ({guide.name})")
    print("\nVariables (--var NAME=VALUE)")
    for item in summary["variables"]:
        if item["name"] in guides.FIXED or item["name"] == "GUIDE_NAME":
            continue
        values = f"  one of: {', '.join(item['values'])}" if item.get("values") else ""
        default = "(sensitive)" if item.get("sensitive") else item["default"] or '""'
        print(f"  {item['name']:<22} {default}{values}")
    if summary["support"]:
        print("\nSupport (accelerator: model server status)")
        for accelerator, engines in summary["support"].items():
            cells = ", ".join(f"{engine} {status}" for engine, status in engines.items())
            print(f"  {accelerator:<22} {cells}")
    print("\nApplies, with these variables")
    for item in summary["applies"]:
        print(f"  {item}")
    for finding in followed.findings:
        print(f"  {finding.severity.upper():<8} {finding.message}")
    chosen = "".join(f" --var {name}={value}" for name, value in variables.items())
    print(f"\nPlan it: llmd plan {stack.id} -n NAMESPACE{chosen}")
    return 0
