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
- Genre skills build on `human-writing` (in the `core` catalog): they state
  that both load together, that the specialized skill leads when they
  conflict, and hand off a missing `human-writing` through
  `ryan-minato-skills-installing`.

## Dependencies

- Default range: `core` only (`human-writing`). No grant between writing
  skills — a medium skill names "a genre writing skill" by role, never a
  specific one — and no dependency on or recommendation of skills from
  other repositories; no exemptions.

## Naming

Suffix by kind: genre skills end in `-writing` (`academic-writing`,
`blog-writing`), source-medium skills in `-authoring` (`latex-authoring`,
`markdown-authoring`). `copywriting` is one word and stands as is.

## References

- https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing — community
  catalog of AI-writing failure patterns (content, language and grammar,
  style, citations).
