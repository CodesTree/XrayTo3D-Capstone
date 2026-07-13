# Reusable Foundation Agents

This tracked library stores persistent role specifications. A role file is a durable contract: it describes what an agent may do, what evidence it needs, and what it must hand downstream. It is not a continuously running worker and does not contain hidden state.

A runtime worker is temporary. The Project Orchestrator creates a concrete task brief, loads common_contract.md plus one role file, and assigns that bounded brief to a worker. Runtime context, commands, results and verdicts are written under TestProject/reports/agent_runs/<stage>/<task_id>/; they are never stored only in the worker conversation.

## Runtime sequence

1. Read common_contract.md.
2. Read the selected role under roles/.
3. Read or create a concrete task brief from templates/task_brief.md.
4. Check predecessor gates in registry.yaml.
5. Perform only the permitted notebook/report work.
6. For an HPC task, issue the result request and pause for the user bundle.
7. Write a handoff using templates/handoff.md.
8. Agent N performs an independent gate review.

Read-only audits may run concurrently. Any workers that could edit the same notebook run sequentially. Agent M is intentionally absent. Agents G, I, J and K are registered but inactive until a later-stage brief explicitly activates them.
