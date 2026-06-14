# Repository Review Results

Last reviewed: 2026-06-14

Scope: current local `master`. I ran the non-slow test suite with coverage:

```bash
uv run pytest -n auto -m "not slow" --cov=coding_assistant --cov-report=term-missing
```

Result: 328 passed, total coverage 83%. Slow Docker integration tests were not part of that command.

Only items I think are worth fixing are listed below.

## 1. Test Coverage Gaps

- Entry points are under-covered in the non-slow suite. Add tests for remaining `manager/main.py` startup config paths plus `worker/main.py` arg parsing, environment parsing, and worker config construction.
- `manager/docker_worker.py` has low non-slow coverage because the meaningful lifecycle paths are mostly in slow Docker tests. Add unit tests with injected command/client factories for container removal, start failure cleanup, readiness timeout, active-run registration, cancellation, and final cleanup without needing a real Docker daemon.
- Structured ACP prompt conversion is under-covered. `remote/acp.py` is at 53% coverage, with many missing branches around text validation, image blocks, embedded resources, resource links, and invalid content forms.
- Remote protocol and controller error paths need more coverage. `remote/control.py`, `remote/protocol.py`, and `manager/server.py` have untested branches for invalid JSON-RPC envelopes, missing request IDs, unknown methods, malformed prompt blocks, disconnects, and failed prompt result handling.
- The Docker integration smoke test covers happy-path manager-to-worker execution, but not the security topology. Add a slow test that proves a worker cannot call the manager once auth is implemented.
