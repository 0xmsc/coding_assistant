# Improvement ideas

This is a curated backlog from a repository audit on 2026-07-10. It covers the
terminal CLI, core agent runtime, local tools, managed sessions, workers, RPC,
bundled skills, and developer workflow. These are proposals, not commitments;
each should get a focused design pass before implementation.

The list deliberately preserves current architectural decisions: CLI and worker
tool bundles remain separate runtime definitions, model discovery remains
manager-owned, manager authentication continues to use the fixed manager-only
secret, injected skills remain session-scoped, and worker subprocesses keep
their scrubbed environment.

## Priority scale

| Priority | Meaning |
| --- | --- |
| P0 | Confirmed correctness defect that can break a normal core workflow. |
| P1 | Major reliability or data-integrity improvement with broad impact. |
| P2 | Meaningful UX, capability, operability, or scalability improvement. |
| P3 | Larger optional product or scalability investment. |
| P4 | Low-urgency maintainability, dependency, or test-structure cleanup. |

## Reliability and correctness

- [ ] **P0 — Preserve session metadata when a worker returns a title**

**Why:** `RemoteWorkerRunner` returns the entire finish `_meta` object as worker
metadata. `ManagerService.prompt()` then passes that object to
`SessionStore.apply_prompt_result()`, which replaces the session metadata rather
than merging it. A worker that returns `{"title": "..."}` can therefore remove
the stored `model`; the next prompt then fails with `Session has no model
selected.` This path is especially likely because workers are instructed to set
a title during the first turn.

**Direction:** Treat title as its own field all the way through the worker
result, define which worker metadata keys are actually allowed to update durable
session metadata, and merge those keys without removing manager-owned metadata.
Add a real two-prompt manager/worker test in which the first prompt sets a title
and the second reuses the selected model.

**Likely scope:** `remote/client.py`, `manager/remote_worker.py`,
`manager/service.py`, `manager/store.py`, and manager integration tests. Small.

- [ ] **P0 — Normalize incomplete provider usage data**

**Why:** `Usage.cost` and `Usage.tokens` are typed as required values, but
`llm/openai.py` can construct usage with a missing cost or token count. Tests
already exercise a provider response without cost, but not through
`AgentSession`, where `float + None` can fail the run. The CLI status formatter
also assumes cost is always numeric.

**Direction:** Choose one explicit contract: either normalize missing provider
fields to safe numeric values while retaining an “unknown” marker elsewhere, or
make the fields optional throughout and render/accumulate them accordingly.
Test the complete provider-to-session-to-status path for missing cost and
missing token counts.

**Likely scope:** `llm/types.py`, `llm/openai.py`, `core/agent_session.py`, and
`cli/output.py`. Small.

- [ ] **P1 — Make retries safe after streaming has begun**

**Why:** `stream_completion()` retries every `httpx.HTTPError`. If a connection
fails after content deltas have already been yielded, the retry starts the
completion again and the client can display duplicated partial text. The
current retry tests cover failures before useful output, not interrupted
streams.

**Direction:** Retry automatically only before the first user-visible delta, or
introduce an explicit message reset/replacement contract before retrying a
partially streamed response. Also distinguish retryable transport/status errors
from permanent 4xx configuration errors and honor server retry hints when
available.

**Likely scope:** `llm/openai.py`, session update semantics, and provider/remote
streaming tests. Medium.

- [ ] **P1 — Bound background-task output without silently losing unread data**

**Why:** `OutputBuffer` retains all subprocess output in a growing `bytearray`,
so a noisy or long-lived background command can grow the process indefinitely.
`tasks_get_output` then consumes the entire buffer before truncating the returned
text, which means the part beyond `truncate_at` is discarded even though the
caller has not seen it. Python failures also return `handle.stdout` without the
normal truncation path.

**Direction:** Back task output with a bounded spool file or bounded chunk/ring
buffer and a read cursor. Report how much is available, return one bounded page
at a time, and make truncation non-destructive. Apply the same result formatting
to successful and failed shell/Python processes. Add positive bounds to
`timeout` and `truncate_at` inputs.

**Likely scope:** `tools/process.py`, `tools/tasks.py`, `tools/shell.py`, and
`tools/python.py`. Medium.

- [ ] **P1 — Close all subprocesses when a tool bundle shuts down**

**Why:** `TaskManager` has no close operation. `CliToolBundle.close()` closes
MCP and remote connections but not shell/Python background tasks, so commands
started by an exiting CLI can remain alive. Worker containers eventually remove
their processes, but making lifecycle ownership explicit benefits both runtime
definitions.

**Direction:** Give `TaskManager` an async, idempotent `close()` that terminates
and awaits every live process. Make both tool bundles own and close their task
manager; make worker agent creation use `try/finally` around bundle lifetime.

**Likely scope:** tool-bundle and process lifecycle code plus shutdown tests.
Small.

- [ ] **P1 — Make attachment upload atomic across files and SQLite**

**Why:** `session/upload_file` writes the attachment to disk before committing
its message and attachment rows. A stale commit, database error, or cancellation
after the write leaves an orphaned file. Writes also go directly to the final
path, so a process interruption can leave a partial file that looks complete.

**Direction:** Write to a temporary file, commit the database record, atomically
rename into place, and clean up on every failure path. If a fully atomic
filesystem/database transaction is not practical, add explicit reconciliation
for orphaned or missing attachment files and test injected failures at each
boundary.

**Likely scope:** `manager/service.py`, `manager/store.py`, and attachment tests.
Medium.

## Product and UX

- [ ] **P1 — Bring CLI image handling up to the worker image contract**

**Why:** worker `load_image` has byte and dimension limits, MIME detection, EXIF
orientation, alpha preservation, and clear decode errors. CLI `/image` has a
separate, less defensive path: URL bodies and local files are read completely,
all images are converted to JPEG, EXIF is not normalized, and an exception from
`get_image()` escapes the input loop instead of becoming a recoverable
user-facing error.

**Direction:** Extract a focused shared byte-to-prepared-image helper while
keeping CLI and worker acquisition policies separate. Stream remote downloads
with a maximum size, reject unsupported content clearly, preserve sensible
formats, and let `/image` report an error without ending the session.

**Likely scope:** `cli/image.py`, `tools/load_image.py`, and CLI UI/tests. Medium.

- [ ] **P2 — Make the terminal interaction model discoverable**

**Why:** `/help` lists only four slash commands, while important behavior lives
in undocumented key bindings: Enter steers, Tab queues, Ctrl-J inserts a
newline, Ctrl-U restores the last queued prompt, and Ctrl-C changes meaning
between cancel/pause, resume, and clear-input. New users cannot infer this from
the prompt or footer. Startup also prints the complete system prompt, which can
bury the first useful interaction in a large repository.

**Direction:** Show a compact first-run/key-binding hint and make `/help`
describe the actual state-dependent controls. Consider `/queue`, `/cancel`, and
`/resume` aliases for discoverability, plus a `/system` command or verbose mode
for the full system prompt instead of always rendering it at startup.

**Likely scope:** `cli/ui.py`, `cli/output.py`, README, and UI tests. Small to
medium.

- [ ] **P2 — Make CLI configuration composable and validated**

**Why:** the README says `--mcp-servers` may be repeated, but argparse's current
`nargs="*"` plus default store action keeps only the last occurrence. Long JSON
objects on the command line are also hard to edit, quote, reuse, and inspect.
Duplicate MCP names are silently collapsed in `MCPServerManager`, unlike
duplicate skill names, which fail clearly.

**Direction:** Fix repeated-option semantics (`action="append"` with deliberate
flattening, or one value per occurrence), reject duplicate server names, and add
a documented TOML/JSON config file for reusable models, skill roots, and MCP
servers. Define clear precedence between config, environment, and command-line
overrides. Keep secrets referenced by environment-variable name rather than
stored in the file.

**Likely scope:** CLI argument/config parsing, `tools/mcp_manager.py`, docs, and
parser tests. Medium.

- [ ] **P3 — Offer local session resume and export**

**Why:** the terminal CLI keeps only prompt-entry history; its conversation and
tool transcript disappear when the process exits. Managed sessions are durable,
but that is a much heavier deployment path for someone who only wants to resume
a local coding session or archive what happened.

**Direction:** Add an opt-in local transcript store with `--resume`/session
selection and a simple Markdown or JSON export. Reuse the core message types,
but keep the storage contract independent from manager SQLite so the CLI remains
the simplest runtime path. Handle interrupted tool calls explicitly rather than
pretending they are resumable.

**Likely scope:** CLI/session persistence, commands, and documentation. Large.

- [ ] **P1 — Distinguish model-provider failure from an empty model list**

**Why:** manager model discovery returns `{"models": []}` when the provider is
unavailable and no cache exists. An application cannot tell “no models exist”
from “credentials/network/provider are broken”; when stale cache exists, it
cannot tell that stale data was served.

**Direction:** Keep discovery manager-owned, but return explicit availability
metadata or a typed RPC error with a retryable signal. When serving cached
models after failure, include cache age/staleness and provide a refresh path.

**Likely scope:** `manager/service.py`, RPC documentation, and manager tests.
Small to medium.

- [ ] **P3 — Add incremental history and real session pagination**

**Why:** `session/list` always returns every session with `nextCursor: null`, and
`session/load` replays the entire transcript and every attachment before
`history_complete`. Long-lived accounts and tool-heavy sessions will make both
startup latency and WebSocket traffic grow without bound.

**Direction:** Define stable cursor pagination for session lists and version- or
message-ID-based incremental history loading. Keep a full-reset mode for simple
clients, but let reconnecting clients request only changes after a known
version. Document ordering and deletion/reset semantics before implementation.

**Likely scope:** store queries, manager service, RPC protocol/docs, and app
integration. Large.

- [ ] **P2 — Preserve structured MCP results**

**Why:** `MCPServerManager.call()` returns text for a single text result and
`str(result.content)` for everything else. Images, embedded resources,
annotations, and multiple content blocks therefore lose their protocol shape
before they reach the agent. This limits MCP-backed visual and document
workflows even though the core runtime already supports structured tool-message
content.

**Direction:** Convert supported MCP content blocks into `ToolMessageResult`
content, preserve multiple blocks, and fail clearly on unsupported variants.
Expose the selected MCP tool's input schema in `mcp_list_tools` so the model does
not have to guess arguments.

**Likely scope:** `tools/mcp_manager.py`, `tools/mcp_tools.py`, result types, and
MCP tests. Medium.

- [ ] **P2 — Improve service-binary diagnostics**

**Why:** `coding-assistant-manager --help` currently raises a traceback because
all CLI arguments are rejected. Startup configuration errors are similarly
useful to operators but are presented as raw exceptions, and the reference
Compose service has no readiness/health signal.

**Direction:** Provide a small help page that documents the environment
contract, format startup failures as concise actionable messages, and expose a
health/readiness check suitable for Compose and deployment automation. Do not
put model or Docker calls on the liveness path.

**Likely scope:** `manager/main.py`, server/deployment plumbing, Compose, and
operator docs. Small to medium.

## Skills, instructions, and maintainability

- [ ] **P2 — Make skill loading strict, diagnosable, and resilient**

**Why:** skill frontmatter parsing can raise and abort startup for one malformed
skill, while missing fields merely log and silently omit it. Names and
descriptions from filesystem skills are not type/format validated to the same
standard as injected skills. The system prompt lists skills, but users do not
get a concise startup summary of what loaded, what was skipped, or why.

**Direction:** Use one validation contract for bundled, configured, and injected
skills; collect per-skill diagnostics; fail on bundled/configured errors by
default with an opt-in lenient mode for user directories. Document precedence
and collision behavior, and add a CLI command that reports active skill roots
and validation failures. Extend instruction discovery to a documented,
deterministic parent/scoped `AGENTS.md` precedence so starting in a repository
subdirectory does not silently miss root guidance.

**Likely scope:** `tools/skills.py`, `core/instructions.py`, CLI UX, and docs.
Medium.

- [ ] **P2 — Refresh the bundled instruction and skill content**

**Why:** `global.md` still tells the model it has built-in TODO tools even
though TODO continuity moved to a skill. The advanced-tool skill uses a rigid
“more than 50 lines” redirect threshold and describes redirection as context
economy even though the redirected tool result is still materialized in memory
before it is written. Its guidance overlaps with tool descriptions and can push
the model toward unnecessary temporary files. The PDF skill stops at “OCR is
required” without offering an optional next workflow.

**Direction:** Remove stale tool claims, make redirection guidance size- and
capability-based, state the memory limitation honestly, and prefer a tool's
native pagination/output controls before redirection. Consider a separate,
triggered OCR skill only if the worker image provides a supported OCR toolchain;
otherwise keep the current clear failure boundary. Add lightweight assertions
that instruction text names only tools present in each runtime bundle.

**Likely scope:** `builtin/instructions`, bundled skills, tool descriptions, and
tool-bundle tests. Small.

- [ ] **P4 — Split manager code and tests by capability, not by technical layer**

**Why:** `manager/service.py` is responsible for RPC payload validation, skill
materialization, attachment I/O, model caching, session mutations, prompt
orchestration, and update conversion. Its main server test file is about 1,770
lines. The breadth makes correctness fixes such as metadata ownership and
attachment cleanup harder to reason about and encourages large integration
fixtures for small behavior.

**Direction:** Keep `ManagerService` as the orchestration boundary, but extract
cohesive attachment storage/validation and session-capability payload parsing.
Split tests into model, session lifecycle, attachments, capabilities, prompt
runs, and authorization modules while retaining a small end-to-end server
suite. Avoid generic repository/factory layers that only move calls around.

**Likely scope:** manager package and tests. Medium.

- [ ] **P4 — Tighten dependency and static-quality hygiene incrementally**

**Why:** `requests` and `types-requests` appear unused, while `debugpy` is a
runtime dependency imported on every CLI startup for an opt-in debug flag.
Ruff currently checks only `E` and `F`; an exploratory run of import, upgrade,
bugbear, simplify, async, performance, and Ruff-specific rules found many
mechanical issues, including inconsistent imports and blocking filesystem calls
inside async functions. Enabling every rule at once would create noisy churn.

**Direction:** Remove confirmed unused dependencies, lazy-import or move
debug-only packages, and enable additional Ruff families in small reviewed
batches (`I` and `UP` first, then selected `B`, `SIM`, and async rules). Use the
new checks to catch real boundary mistakes, not to force clever rewrites. Add a
small coverage report or threshold only after identifying the few critical
paths it should protect, such as two-turn managed sessions and interrupted
streams.

**Likely scope:** `pyproject.toml`, imports, CI, and targeted tests. Small in
stages.
