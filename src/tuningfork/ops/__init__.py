"""The operations layer: every tuningfork function, once.

The CLI, the HTTP API and — in Phase 6 — an MCP server are front ends over these functions,
so each is reachable from all of them and none is built twice. An operation takes typed
parameters, reports progress as events that serialize to JSON, honours cancellation, and
returns a result that knows its own JSON form. Front ends parse and format; nothing a second
front end would need lives in one.

Where an operation needs a person's go-ahead after seeing what it will do — which run and
step, what gets swapped, what is declared as the target — it comes in two parts: `prepare_*`
checks everything and returns a plan without side effects, and `execute_*` carries it out.
`execute_*` either returns its whole result or raises `Cancelled`, whose `partial` is the
result over what finished; what finished stays stored either way.
"""

from tuningfork.ops.common import (
    Cancel,
    Cancelled,
    Event,
    OperationError,
    Progress,
    emit,
    params_schema,
    parse_params,
)

__all__ = [
    "Cancel",
    "Cancelled",
    "Event",
    "OperationError",
    "Progress",
    "emit",
    "params_schema",
    "parse_params",
]
