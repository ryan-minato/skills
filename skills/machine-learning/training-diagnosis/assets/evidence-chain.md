# Evidence chain: <incident short name>

- Run: <run_id> · T0: <step / timestamp, rank when one rank led> · reported: <date>
- Symptom: <what was observed, with the alert or the baseline comparison>
- Signals consulted:
  - <signal>: <value at T0 vs baseline> (<source>)
- Ruled out:
  - <cause>: <the signal that excludes it>
- Root cause: <one sentence, with the mechanism>
- Causal chain: <cause> → <intermediate effect> → <system symptom> → <training impact>
- Fix: <what changed, where>
- Verification: <the measurement after the fix vs baseline; the replay result>
- Missing signal: <what would have decided this faster, and its frequency>
- Open: <remaining candidates and the deciding signal to capture next, or none>
