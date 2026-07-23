# writing — Catalog Context

Rules, notes, and references that apply only to skills in this catalog.
(Repo-wide standards live in `.agents/knowledge/skill-quality.md`.)

## Requirements

- Human readers only: skills here produce or improve text meant to be read
  by people. Code, commit messages, and agent-facing documents (AGENTS.md,
  knowledge bases, skills) are out of scope for this catalog.
- Authorial presence: a skill must never instruct flattening an author's
  explicit stance into mechanical neutrality, nor padding text with
  defensive qualifiers to reduce risk. Readers must be able to sense a
  specific author behind the text.
- Framework-agnostic: no dependency on a specific agent framework, tool
  name, subagent API, or platform layout. Express optional capabilities
  conditionally — "if the environment supports isolated-context subagents,
  use them; otherwise <fallback or skip>".
- Trilingual: skills are written in English and must support English,
  Chinese, and Japanese output. Shared writing logic lives in the SKILL.md
  body; language-level idiom, AI tells, and genre norms are handled per
  language (typically one reference file per language).
- Citation integrity: never fabricate a source. Every cited book, paper,
  case, or dataset must exist and meet the credibility bar the genre
  demands; a claim whose source cannot be verified is dropped.
- Restraint first: when input text is already natural and idiomatic, the
  skill must say what is good and change nothing, rather than editing to
  justify its own invocation.
- Genre skills pair with `human-writing` (in the `core` catalog) via the
  standard install instruction and state that the specialized skill leads
  when both are loaded: its requirements override the general baseline
  where they conflict.

## References

- https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing — community
  catalog of AI-writing failure patterns (content, language and grammar,
  style, citations).
