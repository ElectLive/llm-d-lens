"""``llmd render``: the stack's objects as the guide's commands or ``kubectl apply`` would send them."""

from __future__ import annotations

import argparse
import os
import sys

from .. import catalog, yamlio
from ..errors import UsageError
from ..render import render
from . import Context, Handler, add_source_arguments, parse_variables


def register(commands: argparse._SubParsersAction) -> dict[str, Handler]:
    parser = commands.add_parser("render", help="print a stack's rendered objects (no cluster access)")
    parser.add_argument("stack", help="a guide, a kustomize directory, or manifest files (see `llmd stacks`)")
    add_source_arguments(parser)
    parser.add_argument(
        "-n",
        "--namespace",
        default=os.environ.get("LLMD_NAMESPACE", "") or "default",
        help="the namespace a guide's steps render for (default: $LLMD_NAMESPACE, else default)",
    )
    parser.add_argument("--var", action="append", default=[], metavar="NAME=VALUE", help="a guide variable")
    parser.add_argument(
        "--set", action="append", default=[], dest="sets", metavar="KEY=VALUE", help="a value for the guide's charts"
    )
    return {"render": command_render}


def command_render(ctx: Context) -> int:
    args = ctx.args
    stack = catalog.find_stack(ctx.source(), args.stack)
    variables = parse_variables(args.var)
    if stack.kind != "guide" and (variables or args.sets):
        raise UsageError(f"--var and --set are for guides; {stack.id} is a {stack.kind} stack")
    rendered = render(stack, runner=ctx.runner, namespace=args.namespace, variables=variables, sets=args.sets)
    blocking = [finding for finding in rendered.findings if finding.severity == "error"]
    for finding in rendered.findings:
        print(f"[render] {finding.severity}: {finding.message}", file=sys.stderr)
    if blocking:
        raise UsageError(
            f"guide {stack.id} cannot be rendered with these variables",
            hint=f"Run `llmd stacks {stack.id}` for its variables.",
        )
    print(f"[render] {rendered.command}: {len(rendered.documents)} object(s)", file=sys.stderr)
    if args.output == "json":
        ctx.emit(rendered.documents)
    else:
        print(yamlio.dump_documents(rendered.documents), end="")
    return 0
