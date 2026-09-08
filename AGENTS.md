# ComfyUI-Sai-nodes development instructions

## ComfyUI custom-node skills

- Use the project-local skills under `.agents/skills/` whenever the task matches their trigger.
- Prefer ComfyUI V3 APIs for new nodes unless compatibility requirements explicitly call for V1.
- Keep implementation, tests, packaging metadata, and frontend assets consistent with the relevant installed skills.

## Development records

- Record meaningful development work in the Obsidian vault folder `I:\_ObsidianBase\dev_keeps\Dev_ComfyUI-Sai-nodes`.
- Store durable specifications, decisions, plans, and checklists in `Docs/`.
- Store execution records and verification evidence in `Tasks/`.
- Name tasks `TNNN - <natural-language title>.md` and documents `DNNN - <natural-language title>.md`; Task and Doc IDs use independent sequences.
- Follow `_Habits/Attribute Items.md` and `_Habits/Collaboration settings.md` for properties, status, verification, and handoff conventions.
- Use `Docs/D001 - ComfyUI Custom Node 通用開發規範.md` as the project development baseline.
- Update `Project Home.md` when milestones, architecture, or the active focus changes.
- Each Task should capture the goal, decisions, changed files, developer verification, user acceptance when needed, and follow-up items.
- Use Obsidian wikilinks for links between notes in this vault.

## Environment boundaries

- `I:\Repository\_ComfyUI\start_venv.bat` is user-managed and used to update ComfyUI. Do not modify it unless the user explicitly requests that exact file.
- If this project needs a launcher, create a separate project-owned BAT or PowerShell script.
- Keep machine-specific ComfyUI paths out of committed configuration; accept them through environment variables or ignored local configuration.
