# Deployment

This document is the deployer-facing source of truth for running the Docker
manager service.

## Docker Manager

Copy the example environment file and fill in the values:

```bash
cp docker/.env.example docker/.env
```

Start the manager with:

```bash
just dev-manager
```

That runs:

```bash
docker compose --env-file docker/.env -f docker/compose.manager.yml up --build manager
```

Stop it with:

```bash
just dev-manager-down
```

The manager listens on `ws://localhost:8764`. Clients must connect with:

```text
Authorization: Bearer <value of CODING_ASSISTANT_MANAGER_AUTH_SECRET>
```

## Environment

Project-owned variables use the `CODING_ASSISTANT_` prefix. Provider and tool
variables keep their standard names, such as `OPENAI_API_KEY` and
`OPENAI_BASE_URL`.

`docker/.env.example` is the copyable template. These values are required:

| Variable | Required | Secret | Used by | Description |
| --- | --- | --- | --- | --- |
| `CODING_ASSISTANT_DATA_DIR` | yes | no | Compose, manager | Absolute host path for SQLite state and session workspaces. It is bind-mounted at the same path inside the manager because worker containers are created through the host Docker socket. |
| `CODING_ASSISTANT_DOCKER_SOCKET_GID` | yes | no | Compose | Host group id for `/var/run/docker.sock`, used as a supplemental group so the non-root manager user can access Docker. On Linux, get it with `stat -c '%g' /var/run/docker.sock`. |
| `CODING_ASSISTANT_MANAGER_AUTH_SECRET` | yes | yes | Manager | Bearer token for trusted manager API clients. Browser clients and worker containers must not receive it. |
| `OPENAI_API_KEY` | yes | yes | Manager, worker | OpenAI-compatible provider API key. The manager forwards it to worker containers. |
| `OPENAI_BASE_URL` | no | no | Manager, worker | Optional OpenAI-compatible provider base URL, for example `https://openrouter.ai/api/v1`. The manager forwards it to worker containers when set. |

Docker Compose also considers exported shell variables during interpolation. If
your shell exports one of these names, it may override the value in
`docker/.env`. For repeatable deployments, keep these values in `docker/.env`
and avoid exporting conflicting values in the shell that starts Compose.

## Worker Environment

The manager starts a temporary worker container for each active prompt. Worker
containers do not automatically inherit the manager container environment.
Instead, the manager forwards only the provider variables the built-in worker
needs:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
```

This explicit allowlist is intentional:

- It prevents manager-only secrets, especially
  `CODING_ASSISTANT_MANAGER_AUTH_SECRET`, from leaking into worker containers.
- It keeps worker runtime inputs narrow and predictable.
- It allows provider variables such as `OPENAI_API_KEY` and `OPENAI_BASE_URL`
  to keep their standard names inside the worker.

For the normal Docker manager deployment, set `OPENAI_API_KEY` and optionally
`OPENAI_BASE_URL` in `docker/.env`. Compose passes them into the manager, and
the manager passes those known provider values to each worker. There is no
generic worker environment pass-through option.

## Docker Socket Trust Boundary

The manager container mounts `/var/run/docker.sock` so it can start and remove
worker containers. Access to that socket is effectively access to the host
Docker daemon.

Treat the manager as trusted infrastructure. The manager secret protects the
manager API, but it does not sandbox the manager container itself. If an
attacker gets code execution inside the manager container, they can usually
control Docker on the host.

Workers do not receive the Docker socket. They receive only their generated
session workspace mount and the manager's provider environment allowlist.
