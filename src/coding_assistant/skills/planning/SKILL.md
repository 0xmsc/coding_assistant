---
name: planning
description: Guidelines for iteratively planning tasks and changes before implementation. Use this when the user requests a non-trivial task or when you need to align on a complex implementation strategy.
---

# Task Planning

This skill provides a structured approach for creating, refining, and getting approval for implementation plans. Planning ensures alignment with the user's expectations and minimizes wasted effort.

## Core Principles

- **Iterative refinement**: Start with a high-level approach and add detail as needed.
- **Explicit approval**: Never start non-trivial implementation without the user's "go ahead".
- **Zero-impact planning**: Do not modify any codebase files (filesystem or git) until the plan is approved.
- **Clarification first**: Use the planning phase to resolve ambiguities.

## Planning Workflow

### 1. Information Gathering
Before drafting a plan, explore the context:
- Search the codebase to understand the current state.
- List relevant files and dependencies.
- Identify potential constraints or edge cases.

### 2. Drafting the Plan
Structure your plan clearly. It must include:
- **Objective**: What is the goal of the task?
- **Proposed Approach**: High-level technical strategy.
- **Step-by-step Implementation**: Sequential list of actions (e.g., "Step 1: Create X", "Step 2: Update Y").
- **Verification Strategy**: How will you prove the changes work?

### 3. Clarifying Questions
Every draft plan **must** include at least **3 clarifying questions**. For each question, provide your **proposed answer/assumption**. 

Example:
> 1. **Question**: Should the new API maintain backward compatibility?
>    **Proposed Answer**: Yes, I will keep the existing method and mark it as deprecated.

### 4. Review and Refinement
Present the plan to the user.
- If the user has feedback, update the plan and resubmit for approval.
- An **empty response** from the user can be interpreted as approval.

### 5. Implementation
Once approved, proceed with the implementation according to the finalized plan.

## Planning Best Practices

### Do's
✅ Use a TODO list to track progress during the planning process.
✅ Keep the plan concise and readable.
✅ Explicitly mention if you will be using specific tools or libraries.
✅ Anticipate potential side effects.

### Don'ts
❌ Do not perform any `edit_file`, `write_file`, or `git` commands during the planning phase.
❌ Do not create overly complex plans for trivial tasks.
❌ Do not ignore user preferences expressed in previous messages.

## Documentation

For more details on planning philosophy, see:
- [Planning Principles](references/principles.md)
