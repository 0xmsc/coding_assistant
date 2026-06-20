# Deployment

This document defines the container interface for deploying the Docker manager
service. It does not require a specific orchestrator. `docker/compose.manager.yml`
is only a reference implementation of this contract.

## Manager Container

Run the Coding Assistant image as a manager service with the
`coding-assistant-manager` command.

The manager process needs:

- `--model`: model used by worker agents.
- `--host`: bind host inside the container. Use `0.0.0.0` when exposing the
  service through a container port.
- `--port`: manager WebSocket port. The reference Compose configuration uses
  `8764`.
- `--database`: SQLite database path inside the persistent data mount.
- `--workspace-root`: session workspace root inside the persistent data mount.
- `--worker-image`: image used for worker containers.
- `--worker-network`: container network where workers are reachable by
  container name from the manager.

## Environment Variables

Set these environment variables on the manager container:

```dotenv
CODING_ASSISTANT_MANAGER_AUTH_SECRET=<long-random-secret>
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

`CODING_ASSISTANT_MANAGER_AUTH_SECRET` is required. Clients use it as a bearer
token when connecting to the manager.

`OPENAI_API_KEY` is required for normal worker operation. `OPENAI_BASE_URL` is
optional; set it for OpenAI-compatible providers such as OpenRouter, or leave
it unset to use the default OpenAI API base URL.

Project-owned variables use the `CODING_ASSISTANT_` prefix. Provider and tool
variables keep their standard names, such as `OPENAI_API_KEY` and
`OPENAI_BASE_URL`.

`docker/.env.example` is a copyable helper for the reference Compose
configuration. It includes the manager container environment variables above
plus Compose interpolation values for the host data path and Docker socket
group.

## Runtime Requirements

The manager container must be able to run Docker commands against the host
Docker daemon. The reference deployment does this by mounting
`/var/run/docker.sock` into the manager container and adding the host Docker
socket group as a supplemental group.

The manager container also needs a persistent data mount for:

- SQLite session state.
- Session workspaces.

Use the same absolute path inside the manager container and on the Docker host
when the manager creates worker containers through the host Docker socket. This
lets host-created worker containers mount the same session workspace path.

In the reference Compose configuration, `CODING_ASSISTANT_DATA_DIR` supplies
this host path and `CODING_ASSISTANT_DOCKER_SOCKET_GID` supplies the Docker
socket group id.

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
Workers run from `--worker-image` and join `--worker-network`.

Worker containers do not inherit the full manager environment. The manager
forwards only the provider variables the built-in worker needs:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
```

This allowlist keeps manager-only secrets out of workers and makes the worker
runtime environment predictable. There is no generic worker environment
pass-through option.

Workers do not receive the Docker socket. They receive only their generated
session workspace mount and the provider environment allowlist.

## Docker Socket Trust Boundary

Access to `/var/run/docker.sock` is effectively access to the host Docker
daemon. Treat the manager as trusted infrastructure.

The manager secret protects the manager API, but it does not sandbox the
manager container itself. If an attacker gets code execution inside the manager
container, they can usually control Docker on the host.
