# Colab Markdown Rendering

Read when writing markdown cells. Bundled because the official reference
(markdown_guide.ipynb) sits behind a Colab login that agents cannot pass.
Colab renders text cells with marked.js — close to GitHub Flavored Markdown
but not identical. When a rendering question is not answered here, test in a
real Colab cell instead of assuming GitHub behavior.

## Works in Colab

- Headings `#`–`######`. Colab builds its table-of-contents sidebar from
  them, so keep the hierarchy real: one `#` document title, `##` sections.
- Emphasis, strikethrough, inline code, fenced code blocks with syntax
  highlighting, blockquotes, ordered/unordered/nested lists, horizontal
  rules, links, images.
- GFM pipe tables.
- LaTeX through MathJax: `$...$` inline, `$$...$$` display. Math is a Colab
  strength — prefer real formulas over ASCII approximations.
- A sanitized HTML subset: text-level tags, `<img>`, `<table>`, `<br>`, and
  `<details>`/`<summary>` for collapsible sections.

## Stripped or unreliable

- `<script>`, `<style>`, `<iframe>`, event-handler attributes, and remote
  embeds are sanitized away — interactive HTML in a markdown cell dies
  silently. Interactivity belongs in code cells (form controls, rendered
  outputs).
- GitHub extensions beyond core GFM — task-list checkboxes, footnotes,
  alerts (`> [!NOTE]`), `:emoji:` shortcodes — generally do not render;
  write the character or the prose directly, and verify in a real Colab
  cell before relying on any extension.
- Hand-written `<a name>` anchors are unreliable across Colab versions;
  link between sections through headings and the table of contents.
