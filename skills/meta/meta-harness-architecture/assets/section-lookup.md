## Reading Long Documents

Do not read a long document end to end. Locate the section, then read only
its lines:

1. List headings with line numbers:

   ```bash
   grep -n '^#' ARCHITECTURE.md
   ```

2. Read from the section's line to the next heading's line, for example
   lines 42–78:

   ```bash
   sed -n '42,78p' ARCHITECTURE.md
   ```

Pointers in this file quote target headings byte-exactly in inline code,
so step 1 can be narrowed to an exact match, for example
`grep -n '^## Tech Stack' ARCHITECTURE.md`.
