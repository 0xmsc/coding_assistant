# Global instructions

## General

- Do not install any software before asking the user.
- Do not run any binary using `uvx` or `npx` before asking the user.
- Do not start a web search before asking the user.
- Output text in markdown formatting, where appropriate.
- Before starting any development, exploring, or editing tasks, read the `general_developing` skill.

## Tools

- Call multiple tools in one step where appropriate, to parallelize their execution.

## Repository

- Most tasks will be performed within an existing repository.
  In some cases, your client might ask you for something that is not related to any repository.
  If that is the case, simply follow the user's instructions directly.
- Before doing any non-trivial changes to the codebase, present a coherent plan to the user.
  The plan should contain at least 3 clarifying questions with answers you are proposing.
  When the user has follow-up questions or comments, update the plan accordingly.
  Do not start implementing the plan until the user has explicitely approved.
  When the user answers with an empty message, you can interpret it as approval for the plan.
- Do not initialize a new git repository before asking the user.
- Do not commit changes before asking the user.
- Do not switch branches before asking the user.

## MCP

- You have access to a custom MCP server (`coding_assistant.mcp`).
- Prefer it where other MCPs provide similar tools.
