---
name: general_developing
description: General principles for exploring, developing, editing and refactoring.
---
# General Developing Skill

- Follow clean code principles.
- Ensure proper error handling.
- Prefer simplicity over complexity.

## Developing

- Do not add obvious, redundant comments to the code.
- Do not document what the code is doing, document why it is doing it.
- Do not document trivial functions or classes.

## Exploring 

- Use `pwd` to determine the project you are working on.
- Use shell tools to explore the codebase, e.g. `fd` or `rg`.

## Editing

- Use `cp` & `mv` to copy/move files. Do not memorize and write contents to copy or move.
- Do not try to use `applypatch` to edit files. Use e.g. `sed` or `edit_file`.
- You can use `sed` to search & replace (e.g. to rename variables).
- Writing full files should be the exception. Try to use `edit_file` to edit existing files.
