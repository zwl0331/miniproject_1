# Fast exploration mode: how the prompt was made

2026-11-01, about 04:00 to 04:12 UTC.

| Item | Fact |
|---|---|
| LLM | Claude Opus 5.5 (`claude-opus-5-5`) in Claude Code |
| Interaction | Sam asked for a read-only status report, then said the checks were slowing progress ("I'm just trying to get a quick and dirty test of the different methods and see if I can improve my AUC results a bit more"). Claude estimated that P2.9 at its planned pace would use the whole window to the Kaggle close, and that new model families were unlikely to close a 0.006 to 0.016 CV gap. Sam then asked: "Can you write a prompt that changes the current rule such that the development is faster and more faithful to the goal I stated in this session with you?" |
| Output | `prompt.md`, drafted by Claude for a new Codex session. |
| Integration | Sam pasted the prompt into the Claude Code session instead. The new Codex session had received only "Can you read in the project to get started?" and was idle, so Claude Code carries out the prompt as the primary agent. |
| Effect | `AGENTS.md` section "Fast exploration mode (Sam, 2026-11-01)"; one row in `decisions.md`; `STATE.md` rewritten short. |
