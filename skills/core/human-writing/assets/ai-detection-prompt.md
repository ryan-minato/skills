# AI-writing detection prompt template

Send to a subagent with a separate, clean context. Provide only the text —
no authorship information, no drafting history, no hint of what answer is
expected. Without clean-context subagents, skip this pass; the taxonomy
below still serves as a drafting and self-review checklist.

---

Assess whether the following text reads as AI-generated. Judge by the
accumulation of patterns and your own overall impression — a single match
means little; several stacking up in one passage means a lot. Polished or
formal writing is not, by itself, AI writing: do not flag text merely for
being correct and well organized.

The text:

{{full text}}

Check against these pattern families (drawn from Wikipedia's community
catalog "Signs of AI writing" and general experience), plus anything else
you notice:

**Content patterns**
- Inflated significance: "stands as a testament", "pivotal moment",
  "underscores its importance", "reflects broader trends", "evolving
  landscape", subjects framed as historically important without evidence.
- Superficial analysis bolted on with "-ing" clauses: "..., highlighting
  the need for...", "..., emphasizing the role of...".
- Promotional gloss on neutral subjects: "boasts", "vibrant", "rich
  heritage", "seamless experience".
- Views attributed to no one: "experts say", "many consider", "is widely
  regarded", with no named source.
- Formulaic closings about challenges and future prospects: "Despite these
  challenges, ... continues to ...", "The future of ... remains ...".

**Language patterns**
- AI-flavored vocabulary density: delve, underscore, pivotal, crucial,
  robust, landscape, tapestry, realm, testament, meticulous, intricate,
  showcase, foster, leverage, garner (English); 赋能, 抓手, 闭环, 里程碑式,
  值得注意的是 (Chinese); ソリューション/インサイト density, 「〜と言える
  でしょう」 endings (Japanese).
- Copula avoidance: "serves as / functions as / represents" where "is"
  would do.
- Negative parallelism as a habit: "not just X, but Y"; "it's not about X,
  it's about Y"; 「XだけでなくYも」; “不仅...更...”.
- Rule-of-three chains in every paragraph.
- Synonym rotation for one recurring term; uniform sentence length; every
  claim hedged ("arguably", "can potentially", 「〜かもしれません」).

**Style patterns**
- Mechanical bold emphasis; bolded lead-in bullet lists replacing prose;
  emoji as section markers; em-dash overuse; title-case headings in
  English; uniform paragraph shape and length throughout.
- Mixed punctuation width in Chinese/Japanese; です・ます / だ・である
  mixing in Japanese.

**Citation patterns**
- References that do not exist or do not support the claim; dead or
  tracking-parameter URLs; book citations with no page numbers; a polished
  reference list with nothing actually cited in the body.

Signals of human writing (weigh these against the above): varied rhythm,
concrete first-hand detail, a stance the author commits to, tolerated
repetition of the right word, small idiosyncrasies that a style-averaging
process would smooth away.

Report:

1. Verdict: reads as AI-written / leaning AI / leaning human / reads as
   human-written.
2. For each finding: the quoted passage, the pattern family, and why it
   contributes to the verdict.
3. The three passages that most drive your verdict, whichever way.
4. Passages that read distinctly human, if any.

Do not rewrite the text; report findings only.
