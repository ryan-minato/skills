# Ruby Script Specification

## Inline Dependencies with `bundler/inline`

```ruby
require 'bundler/inline'

gemfile do
  source 'https://rubygems.org'
  gem 'nokogiri', '~> 1.16'
  gem 'httparty', '= 0.22.0'
end
```

Place at the top of the script, before other requires. Always pin versions — there is
no `Gemfile.lock` for inline gemfiles.

| Specifier | Meaning |
|---|---|
| `'~> X.Y'` | `>= X.Y` and `< X+1` (pessimistic constraint) |
| `'~> X.Y.Z'` | `>= X.Y.Z` and `< X.Y+1` |
| `'= X.Y.Z'` | Exact pin |
| `'>= X.Y'` | Minimum only — avoid without an upper bound |

## Gemfile Interference

`bundler/inline` is **disabled** when a `Gemfile` exists in the current working
directory or any parent, or `BUNDLE_GEMFILE` is set. Mitigation:

```bash
BUNDLE_GEMFILE=/dev/null ruby scripts/extract.rb
```

Document this in `SKILL.md` if the skill is likely to be used in environments with
existing `Gemfile`s.

## Shebang

```ruby
#!/usr/bin/env ruby
```

## Running

```bash
ruby scripts/extract.rb
```

## Design Rules

- **No interactive prompts** — use `OptionParser` or `ARGV` for all input
- **`--help` required** — `OptionParser` generates it automatically
- **Data to stdout, diagnostics to stderr** — `puts` vs `warn`
- **Structured output** (JSON preferred; `require 'json'`)
- **Exit codes**: `exit(0)` success, `exit(2)` bad arguments, `exit(1)` general error
