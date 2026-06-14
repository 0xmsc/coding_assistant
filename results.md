# Repository Review Results

Last reviewed: 2026-06-14

Scope: current local `master`. I ran the non-slow test suite with coverage:

```bash
uv run pytest -n auto -m "not slow" --cov=coding_assistant --cov-report=term-missing
```

Result: 328 passed, total coverage 83%. Slow Docker integration tests were not part of that command.

Only items I think are worth fixing are listed below.

## 1. Security Findings To Improve

- The manager API is unauthenticated while the manager container has the Docker socket. The compose file mounts `/var/run/docker.sock`, binds the manager on `0.0.0.0` inside the Docker network, and starts workers on that same network. A malicious prompt with shell/Python execution in a worker can connect back to `ws://manager:8764`, choose its own `_meta.scopeId`, create sessions, and prompt them. That can make the manager start more Docker containers via the socket. Add manager API authentication with a secret that is never passed to worker containers, and consider separating manager and worker networks.
- There is no global concurrency or resource limit for manager-launched workers. The manager rejects a second prompt for the same session, but tests explicitly allow concurrent prompts for different sessions. `_docker_run_args()` does not set `--memory`, `--cpus`, `--pids-limit`, or similar limits. Add a manager-side global semaphore/quota and Docker resource limits to reduce container-storm, host resource, and provider-spend risk.
- Worker tools can read provider credentials from their environment. The dev compose passes `OPENAI_API_KEY` and `OPENAI_BASE_URL` to workers, and `start_process()` copies `os.environ` into shell/Python subprocesses. A malicious prompt can print or exfiltrate these values. Prefer manager-side model calls, short-lived scoped worker credentials, or an explicit scrubbed tool env.
- The Docker socket mount makes manager compromise equivalent to host-level Docker control. The current prompt path does not expose arbitrary Docker flags or arbitrary host bind mounts, but any manager code execution or manager API bug has high impact. Consider a narrow Docker API proxy, a dedicated least-privileged worker launcher, or moving worker creation out of the network-reachable manager process.
- Risky tools run without a permission round trip. `docs/rpc.md` already notes there is no permission request before tools execute, while shell, Python, filesystem, remote, and MCP tools can perform high-impact actions. For web or multi-tenant use, add capability profiles and approval gates before exposing these tools to model-driven prompts.
- CLI tracing is enabled by default and records full provider messages, tool definitions, and completions through `trace_json("completion.json5", ...)`. That is useful for debugging, but it can persist secrets, source code, and private prompt content. Consider default-off tracing, redaction, or clearer warnings.

## 2. Test Coverage Gaps

- Entry points are effectively untested in the non-slow suite: `manager/main.py` and `worker/main.py` show 0% coverage, and `worker/agent.py` also shows 0%. Add tests for arg parsing, environment parsing, manager/worker config construction, and future auth/resource-limit flags.
- `manager/docker_worker.py` has low non-slow coverage because the meaningful lifecycle paths are mostly in slow Docker tests. Add unit tests with injected command/client factories for container removal, start failure cleanup, readiness timeout, active-run registration, cancellation, and final cleanup without needing a real Docker daemon.
- Security behavior lacks regression tests. Add tests that manager auth is required, worker containers do not receive the manager secret, worker prompts cannot recursively call the manager, worker env is scrubbed for shell/Python tools, and generated `docker run` args include resource limits and no Docker socket mount for workers.
- Structured ACP prompt conversion is under-covered. `remote/acp.py` is at 53% coverage, with many missing branches around text validation, image blocks, embedded resources, resource links, and invalid content forms.
- Remote protocol and controller error paths need more coverage. `remote/control.py`, `remote/protocol.py`, and `manager/server.py` have untested branches for invalid JSON-RPC envelopes, missing request IDs, unknown methods, malformed prompt blocks, disconnects, and failed prompt result handling.
- The Docker integration smoke test covers happy-path manager-to-worker execution, but not the security topology. Add a slow test that proves a worker cannot reach/call the manager once auth/network isolation is implemented.
