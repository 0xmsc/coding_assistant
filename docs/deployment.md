# Deployment

This document defines the container interface for deploying the Docker manager
service. It does not require a specific orchestrator. `docker/compose.manager.yml`
is only a reference implementation of this contract.

## Manager Container

Run the Coding Assistant image with its default command. The image starts
`coding-assistant-manager`, and the manager reads deployment configuration only
from environment variables. The manager fails during startup when a required
environment variable is missing or empty.

Expose manager port `8764`.

## Required Environment

Set these environment variables on the manager container:

```dotenv
CODING_ASSISTANT_MANAGER_AUTH_SECRET=<long-random-secret>
CODING_ASSISTANT_HOST_DATA_DIR=/absolute/host/path/backing/manager-data
CODING_ASSISTANT_WORKER_IMAGE=<worker-image>
CODING_ASSISTANT_WORKER_NETWORK=<worker-network>
OPENAI_API_KEY=sk-... # or OPENROUTER_API_KEY=sk-or-...
```

`CODING_ASSISTANT_MANAGER_AUTH_SECRET` is the bearer token clients use when
connecting to the manager.

The manager always stores state under `/data` inside the manager container.
Mount persistent storage at `/data`, and set `CODING_ASSISTANT_HOST_DATA_DIR`
to the absolute Docker-host path backing that mount.

`CODING_ASSISTANT_WORKER_IMAGE` is the image used for temporary worker
containers. `CODING_ASSISTANT_WORKER_NETWORK` is the container network where
workers are reachable by container name from the manager.

Set `OPENAI_API_KEY` for OpenAI or OpenAI-compatible providers. Set
`OPENROUTER_API_KEY` instead when using OpenRouter. Provider and tool variables
keep their standard names; project-owned variables use the
`CODING_ASSISTANT_` prefix. Provider keys are forwarded to worker containers.

The manager does not choose a default model. Clients must call `model/list`,
let the user or application select a model, and persist it with
`session/set_model` before prompting a session.

## Optional Environment

Set this variable only when using an OpenAI-compatible provider with a custom
base URL:

```dotenv
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Leave it unset to use the default OpenAI API base URL, or OpenRouter's default
base URL when `OPENROUTER_API_KEY` is set.

## Runtime Requirements

The manager container must be able to run Docker commands against the host
Docker daemon. The reference deployment does this by mounting
`/var/run/docker.sock` into the manager container and adding the host Docker
socket group as a supplemental group.

The manager container also needs a persistent data mount for:

- SQLite session state.
- Session directories.

Mount the host path from `CODING_ASSISTANT_HOST_DATA_DIR` at `/data` inside the
manager container. The manager stores `sessions.sqlite` at `/data/sessions.sqlite`
and session directories below `/data/sessions`.

The manager creates worker containers through the host Docker socket, so worker
bind mounts must use host paths. The manager maps
`/data/sessions/<session>/workspace` to
`$CODING_ASSISTANT_HOST_DATA_DIR/sessions/<session>/workspace` as `/workspace`
and maps `/data/sessions/<session>/attachments` to
`$CODING_ASSISTANT_HOST_DATA_DIR/sessions/<session>/attachments` as read-only
`/attachments` when starting a worker.

In the reference Compose configuration, `CODING_ASSISTANT_DOCKER_SOCKET_GID`
is only a Compose interpolation value for the Docker socket group. It is not a
manager environment variable.

## Client Authentication

Clients connect to the manager WebSocket endpoint and send:

```text
Authorization: Bearer <value of CODING_ASSISTANT_MANAGER_AUTH_SECRET>
```

Do not expose `CODING_ASSISTANT_MANAGER_AUTH_SECRET` to browser clients or
worker containers. Browser traffic should terminate at a trusted backend that
connects to the manager with this secret.

## Worker Containers

The manager starts one temporary worker container for each active prompt.
Workers run from `CODING_ASSISTANT_WORKER_IMAGE` and join
`CODING_ASSISTANT_WORKER_NETWORK`. Workers listen on port `8765`, and their
session workspace is mounted at `/workspace`. Session attachments are mounted
read-only at `/attachments`.

Worker processes are one-shot. After sending `_session/run_finished` for the
prompt, the worker server exits normally and Docker removes the temporary
container.

Worker containers do not inherit the full manager environment. The manager
forwards only the provider variables the built-in worker needs:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
OPENROUTER_API_KEY
```

This allowlist keeps manager-only secrets out of workers and makes the worker
runtime environment predictable. There is no generic worker environment
pass-through option.

Workers do not receive the Docker socket. They receive only their generated
session workspace mount, read-only attachments mount, and the provider
environment allowlist.

## Docker Socket Trust Boundary

Access to `/var/run/docker.sock` is effectively access to the host Docker
daemon. Treat the manager as trusted infrastructure.

The manager secret protects the manager API, but it does not sandbox the
manager container itself. If an attacker gets code execution inside the manager
container, they can usually control Docker on the host.
