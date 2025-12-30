---
name: create_skill
description: Guide for creating new Agent Skills. Use this skill when you need to extend the agent's capabilities with specialized knowledge or workflows.
---

# Create Skill

This skill provides the instructions necessary to create new Agent Skills following the [Agent Skills specification](https://github.com/agentskills/agentskills).

## Skill Structure

A skill is a directory containing at minimum a `SKILL.md` file:

```
skill-name/
├── SKILL.md          # Required: instructions + metadata
├── scripts/          # Optional: executable code
├── references/       # Optional: documentation
└── assets/           # Optional: templates, resources
```

## Creating a new Skill

### 1. Naming
- The skill name must be 1-64 characters.
- Use only lowercase alphanumeric characters and hyphens (`a-z`, `0-9`, `-`).
- Must not start or end with a hyphen.
- Must not contain consecutive hyphens (`--`).
- The directory name must match the `name` field in `SKILL.md`.

### 2. SKILL.md Frontmatter
Every `SKILL.md` must start with YAML frontmatter:

```yaml
---
name: skill-name
description: A clear description of what the skill does and when to use it.
---
```

**Description Guidelines:**
- Describe both what the skill does and when the agent should use it.
- Include keywords to help with discovery.

### 3. SKILL.md Content
The body of `SKILL.md` should contain clear instructions for the agent. Use Markdown formatting.
Recommended sections:
- **Prerequisites**: Any tools or setup needed.
- **Workflow**: Step-by-step instructions.
- **Examples**: How to perform specific tasks.
- **Troubleshooting**: Common issues.

### 4. Progressive Disclosure
- Keep `SKILL.md` focused and under 500 lines.
- Move detailed technical references or large datasets to the `references/` directory.
- Move executable logic to the `scripts/` directory.

## Documentation

For more detailed information, see the original documents:
- [What are Skills?](references/what-are-skills.md)
- [Full Specification](references/specification.md)
