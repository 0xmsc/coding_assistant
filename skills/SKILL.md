---
name: skills
description: Guide for creating and editing Agent Skills. Use this skill when you need to create or edit a skill.
---

# Skills

## Skill Structure
```
skill-name/
├── SKILL.md          # instructions + metadata
├── scripts/          # executable code (optional)
├── references/       # documentation (optional)
└── assets/           # templates/resources (optional)
```

## Requirements
- **Frontmatter**: Must include `name` (lowercase-alphanumeric) and `description`.
- **Description**: Must include **what** it does and **when** to use it.
- **Body**: Keep under 500 lines. Use imperative language. 
- **Resources**: Move detailed docs to `references/` and code to `scripts/`.

## Process
1. **Directory**: `mkdir <name>`. Create `scripts/`, `references/`, or `assets/` only if you actually have content for them.
2. **Metadata**: Write the `SKILL.md` frontmatter first.
3. **Resources**: Implement scripts and test them.
4. **Instructions**: Write concise steps in `SKILL.md` body.
5. **Verify**: Ensure frontmatter name matches directory name.

## Editing Skills
- Follow the same structure and requirements as creating a skill.
- Ensure that the skill's functionality is preserved or enhanced.
- Update the `SKILL.md` to reflect any changes made.

## References
- [Full Specification](references/specification.md)
- [Design Patterns](references/what-are-skills.md)