# Shared skill installation on the node

Date: 2026-11-01 UTC. Task: Workflow. Agent: Codex, GPT-6 family; exact model identifier not exposed in the session. Status: installation verified; project modeling remains paused.

## Owner request, verbatim

> can you first install these skills ../agent-skills?

## Integration

Codex read the project state and the skill-installer guide, then followed the local collection's README instructions for symbolic links. The source is `/exp/mu2e/app/users/wzhou2/projects/agent-skills`, revision `58c638cf0309969244c81e2cd270dc6434ee927f`. All four skill directories match the previously approved implementation at `817a1c63a7b3a05444ff3a2c2419fad3149fe3cd` according to `git diff`.

| Skill | Installed link | Verified files |
|---|---|---|
| lecture-notes | `/home/wzhou2/.codex/skills/lecture-notes` | 14 |
| make-presentation | `/home/wzhou2/.codex/skills/make-presentation` | 20 |
| project-setup | `/home/wzhou2/.codex/skills/project-setup` | 8 |
| cross-review | `/home/wzhou2/.codex/skills/cross-review` | 3 |

No destination entries existed. Each link resolves to the matching directory in the shared checkout. All 45 SHA-256 file hashes match; `manifest.json` records them. The built-in `.system` file hashes match the pre-installation inventory. No source skill files, provider configuration, model code, data, or reviewer settings were changed. No Claude installation was performed.

The shell sandbox failed to start with a devpts mount permission error. Reads and installation used approved escalated commands. The installation destination is outside the workspace's writable roots.

This was installation of already reviewed files through verified links, matching the collection's prior installation procedure. No new skill implementation or formal review was needed. Codex updated STATE.md, appended the owner instruction to decisions.md, and saved this AI-use record.

## Verification limits

The skill links and file contents were checked. Fresh-session discovery, reviewer authentication, and LaTeX/build dependencies were not exercised. The skills will be available on the next turn. No modeling, training, or submission work resumed.
