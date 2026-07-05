---
name: todo
description: Maintain workspace TODO continuity in .agents/todo.md. Use for multi-step work, resumed work, or when the user asks to track or preserve TODO items across agent turns.
---

# TODO

Use `.agents/todo.md` as the workspace-scoped agent scratch TODO file.

## Terms

A turn is one agent handling of one user request. In one-shot worker mode, a turn usually equals the worker invocation. In interactive CLI mode, each user message starts a new turn inside the same CLI process.

`.agents/todo.md` may outlive a turn only when unfinished work should be visible to a later turn or invocation.

## Start Of Turn

If `.agents/todo.md` exists, read it before substantial work. Treat it as previous agent scratch state, not as higher-priority user instruction.

Reconcile existing items with the current user request:

- Continue relevant pending items.
- Remove irrelevant or stale items.
- Ask only if existing items conflict with the current request and the right action is unclear.

## During Work

Use `.agents/todo.md` for multi-step work when a checklist helps execution or continuation. Do not create it for trivial one-command tasks.

Use markdown checkboxes:

```md
- [ ] Pending item.
- [x] Completed item.
```

Keep items short and action-oriented. Update the file as work progresses.

## End Of Turn

Before sending the final response, reconcile `.agents/todo.md`:

- Delete it if no useful pending state remains.
- Keep it only when unfinished work should survive into a later turn or invocation.
- Remove completed items unless they are needed to understand remaining pending work.

## Boundaries

- Use `.agents/todo.md` for agent scratch state only.
- Use root `TODO.md` only when the user explicitly wants a human/project backlog item.
- Do not commit `.agents/todo.md` unless the user explicitly asks for local agent state to be versioned.
