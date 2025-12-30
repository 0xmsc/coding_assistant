---
name: developing
description: General principles for exploring, developing, editing and refactoring code.
---

# General Developing Skill

- Most tasks will be performed within an existing repository. In some cases, your client might ask you for something that is not related to any repository. If that is the case, simply follow the user's instructions directly.
- Follow clean code principles.
- Ensure proper error handling.
- Prefer simplicity over complexity.

## Planning

- Before doing any non-trivial changes to the codebase, present a coherent plan to the user.
- The plan should contain at least 3 clarifying questions with answers you are proposing.
- When the user has follow-up questions or comments, update the plan accordingly.
- Do not start implementing the plan until the user has explicitly approved.
- When the user answers with an empty message, you can interpret it as approval for the plan.

## Developing

- Do not add obvious, redundant comments to the code.
- Do not document what the code is doing, document why it is doing it.
- Do not document trivial functions or classes.

## Git

- Do not initialize a new git repository before asking the user.
- Do not commit changes before asking the user.
- Do not switch branches before asking the user.

## Exploring 

- Use `pwd` to determine the project you are working on.
- Use shell tools to explore the codebase, e.g. `fd` or `rg`.

## Editing

- Use `cp` & `mv` to copy/move files. Do not memorize and write contents to copy or move.
- Do not try to use `applypatch` to edit files. Use e.g. `sed` or `edit_file`.
- You can use `sed` to search & replace (e.g. to rename variables).
- Writing full files should be the exception. Try to use `edit_file` to edit existing files.
