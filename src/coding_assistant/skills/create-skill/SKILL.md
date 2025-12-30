---
name: create-skill
description: Guide for creating new Agent Skills. Use this skill when you need to create a new skill.
---

# Create Skill

This skill provides comprehensive guidance for creating effective Agent Skills that extend capabilities with specialized knowledge, workflows, and tool integrations.

## What Are Skills

Skills are modular packages that transform a general-purpose agent into a specialized one by providing:
- **Specialized workflows** - Multi-step procedures for specific domains
- **Tool integrations** - Instructions for working with file formats, APIs, or tools
- **Domain expertise** - Company-specific knowledge, schemas, business logic
- **Bundled resources** - Scripts, references, and assets for complex tasks

## Core Design Principles

### Concise is Key
The context window is a shared resource. Only add what the agent truly needs. **Challenge each piece**: "Does the agent really need this explanation?"

**Default assumption: The agent is already smart.** Add context only when necessary.

### Progressive Disclosure
Skills use a three-level loading system:

1. **Metadata** (name + description) - Always loaded (~100 words)
2. **SKILL.md body** - Loaded when skill triggers (<500 lines)
3. **Bundled resources** - Loaded as needed (scripts/references/assets)

Keep SKILL.md lean. Split detailed content into references. Move executable code to scripts.

### Match Freedom to Task
- **High freedom** (text instructions): Multiple valid approaches, context-dependent decisions
- **Medium freedom** (pseudocode/scripts with parameters): Preferred patterns, some variation acceptable
- **Low freedom** (specific scripts): Fragile operations, consistency critical

## Skill Structure

A skill is a directory containing:

```
skill-name/
├── SKILL.md          # Required: instructions + metadata
├── scripts/          # Optional: executable code
├── references/       # Optional: documentation
└── assets/           # Optional: templates/resources
```

### SKILL.md Requirements

#### Frontmatter (Required)
Every SKILL.md must start with YAML frontmatter:

```yaml
---
name: skill-name
description: Clear description of what the skill does AND when to use it
---
```

**Name rules:**
- 1-64 characters
- Lowercase alphanumeric + hyphens only
- Cannot start/end with hyphen
- No consecutive hyphens

**Description guidelines:**
- Describe both **what** it does and **when** to use it
- Include specific triggers/contexts
- Include keywords for discovery
- **All "when to use" information belongs in description** - body loads after triggering

#### Body (Required)
Clear instructions for using the skill. Use imperative form. Recommended sections:
- Prerequisites
- Workflow steps
- Examples
- Troubleshooting

**Length limit**: Keep under 500 lines to avoid context bloat.

## Bundled Resources

### Scripts (`scripts/`)
Executable code for tasks requiring deterministic reliability.

**When to include:**
- Same code rewritten repeatedly
- Deterministic reliability needed
- Complex operations prone to errors

**Examples:**
- `rotate_pdf.py` - PDF rotation
- `validate_schema.py` - Schema validation
- `generate_report.py` - Report generation

**Benefits**: Token efficient, executable without loading into context, testable.

### References (`references/`)
Documentation to be loaded as needed.

**When to include:**
- Database schemas
- API documentation
- Domain knowledge
- Company policies
- Detailed workflow guides

**Examples:**
- `schema.md` - Table schemas
- `api_docs.md` - API specifications
- `policies.md` - Company policies
- `workflow.md` - Detailed procedures

**Best practices:**
- Keep SKILL.md lean by moving details here
- For files >10k words, include grep search patterns in SKILL.md
- Avoid duplication between SKILL.md and references
- If >100 lines, add table of contents

### Assets (`assets/`)
Files used in output, not loaded into context.

**When to include:**
- Templates
- Brand assets
- Boilerplate code
- Sample documents

**Examples:**
- `logo.png` - Brand assets
- `slides.pptx` - PowerPoint templates
- `frontend-template/` - HTML/React boilerplate
- `font.ttf` - Typography files

## What NOT to Include

Do NOT create extraneous documentation:
- README.md
- INSTALLATION_GUIDE.md
- QUICK_REFERENCE.md
- CHANGELOG.md
- etc.

Only include files that directly support the skill's functionality.

## Creation Process

### Step 1: Understand with Concrete Examples
Skip if usage patterns are already clear. Otherwise, ask:
- "What functionality should this skill support?"
- "Can you give examples of how it would be used?"
- "What would a user say to trigger this skill?"

### Step 2: Plan Reusable Contents
Analyze each example to identify what resources would help:

**Example**: `pdf-editor` skill for "Help me rotate this PDF"
- Analysis: Rotating requires rewriting same code
- Resource: `scripts/rotate_pdf.py`

**Example**: `big-query` skill for "How many users logged in today?"
- Analysis: Requires re-discovering schemas each time
- Resource: `references/schema.md`

**Example**: `frontend-webapp-builder` for "Build me a todo app"
- Analysis: Same boilerplate each time
- Resource: `assets/hello-world/` template

### Step 3: Initialize the Skill
Create a new directory for the skill with the required `SKILL.md` file and optional resource directories:

```bash
mkdir -p <skill-name>/{scripts,references,assets}
touch <skill-name>/SKILL.md
```

### Step 4: Edit the Skill

#### Learn Design Patterns
Consult the [Design Patterns](#design-patterns) section below for guidance on structuring complex workflows and maintaining output quality.

#### Start with Resources
Implement reusable resources first:
1. Add scripts and test them
2. Add reference documentation
3. Add assets/templates

#### Update SKILL.md
**Frontmatter**: Write clear name and description.

**Body**: Use imperative form. Include:
- Prerequisites
- Workflow
- Examples
- Troubleshooting

### Step 5: Verify the Skill
Ensure the skill follows the requirements:
1. Valid frontmatter with `name` and `description`.
2. The `name` matches the directory name.
3. All referenced resources exist within the skill directory.

### Step 6: Iterate
Test the skill on real tasks, then:
1. Notice struggles or inefficiencies
2. Identify needed improvements
3. Update SKILL.md or resources
4. Re-test and iterate again

## Design Patterns

### Multi-Step Workflows
For complex processes:
1. Break into clear sequential steps
2. Include decision points
3. Provide fallback options
4. Document error handling

### Output Quality Standards
For specific formats:
1. Define quality criteria upfront
2. Include validation checks
3. Provide templates or examples
4. Document common pitfalls

### Domain Organization
For skills with multiple domains:
```
bigquery-skill/
├── SKILL.md (overview + navigation)
└── reference/
    ├── finance.md
    ├── sales.md
    ├── product.md
    └── marketing.md
```

### Variant Organization
For skills supporting multiple options:
```
cloud-deploy/
├── SKILL.md (workflow + selection)
└── references/
    ├── aws.md
    ├── gcp.md
    └── azure.md
```

## Quick Reference

### Do's
✅ Be concise - challenge every sentence
✅ Use progressive disclosure
✅ Match freedom to task fragility
✅ Test scripts before packaging
✅ Move detailed docs to references
✅ Use imperative language in SKILL.md body
✅ Include "when to use" in description

### Don'ts
❌ Don't include extraneous documentation files
❌ Don't duplicate information between SKILL.md and references
❌ Don't create deeply nested references (keep one level deep)
❌ Don't exceed 500 lines in SKILL.md body
❌ Don't put "when to use" info in the body

## Documentation

For detailed specifications, see:
- [What are Skills?](references/what_are_skills.md)
- [Full Specification](references/specification.md)