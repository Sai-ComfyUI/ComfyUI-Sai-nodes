# ComfyUI-Sai-nodes development instructions

## ComfyUI custom-node skills

- Use the project-local skills under `.agents/skills/` whenever the task matches their trigger.
- Prefer ComfyUI V3 APIs for new nodes unless compatibility requirements explicitly call for V1.
- Keep implementation, tests, packaging metadata, and frontend assets consistent with the relevant installed skills.

## Development records

- When local collaboration guidance maps this repository to a Vault Project, read the Vault root `AGENTS.md`, the mapped Project Home, the current Task or Plan, and directly relevant canonical records before meaningful work.
- Keep personal Vault paths and machine-specific routing in local guidance, outside this repository.
- Scoped canonical records are the development-record SSOT. Base definitions and rendered views are the read-only presentation layer. Product behavior is established by code, tests, builds, and execution results.
- Read and edit Vault records directly through the filesystem; Obsidian and Obsidian CLI are not required for record maintenance.
- Store durable specifications, decisions, plans, and checklists in `Docs/`.
- Store execution records and verification evidence in `Tasks/`.
- Follow the mapped Vault's canonical metadata, record lifecycle, verification, handoff, and agent workflow rules rather than duplicating their schemas here.
- Use `Docs/D001 - ComfyUI Custom Node 通用開發規範.md` as the project development baseline.
- Update `Project Home.md` when milestones, architecture, or the active focus changes.
- Each Task should capture the goal, decisions, changed files, developer verification, user acceptance when needed, and follow-up items.
- Use Obsidian wikilinks for links between notes in this vault.
- Never create fallback copies of vault records in the repository, including `.tmp*.md` files. If a vault write fails, preserve the existing SSOT, report the failed target, and retry or stop instead of writing a repository-side substitute.
- Before completing meaningful work, synchronize the Task and any affected Plan or Project Home, then run the centralized collaboration audit specified by the mapped Project using its documented parameters and fix policy.
- Use the Vault's central-tool routing decision for ownership and chat handoff: existing audit execution and project-record corrections stay with the source project; shared tooling and Vault-wide rule changes belong to the central maintenance project. Do not create repository-local audit copies.
- Keep `Project Home.md` as a compact navigation and current-focus page. Detailed execution history belongs in Tasks; phase details belong in Plans.

## Environment boundaries

- The ComfyUI installation's `start_venv.bat` is user-managed and used to update ComfyUI. Do not modify it unless the user explicitly requests that exact file.
- If this project needs a launcher, create a separate project-owned BAT or PowerShell script.
- Keep machine-specific ComfyUI paths out of committed configuration; accept them through environment variables or ignored local configuration.
