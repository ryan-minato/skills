# Release automation recipes

Load condition: the user wants automation beyond the tag check —
auto-drafting releases or attaching build artifacts. First-party-only
rule applies: everything here uses `actions/*` steps and plain `run:`
commands with gh (gh is preinstalled on GitHub-hosted runners and
authenticates with the workflow's `GITHUB_TOKEN`).

## Auto-draft a release when a tag is pushed

Appended as a second job in `.github/workflows/tag-check.yml` (so it
runs only on tags that passed validation), or as its own workflow:

```yaml
  draft-release:
    needs: validate-tag
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Draft the release with generated notes
        env:
          GH_TOKEN: ${{ github.token }}
          TAG: ${{ github.ref_name }}
        run: >
          gh release create "$TAG" --repo "$GITHUB_REPOSITORY"
          --draft --generate-notes --verify-tag
```

The release stays a **draft**: a human (or the generated release skill,
gate included) curates the notes and publishes. Auto-publishing skips
the review gate — do not wire `--draft=false` into CI unless the user
explicitly accepts unreviewed notes going live.

## Attach build artifacts to the draft

Add build steps before the create, then upload:

```yaml
      - uses: actions/checkout@v4
      # ... build steps producing dist/ ...
      - name: Upload assets to the draft
        env:
          GH_TOKEN: ${{ github.token }}
          TAG: ${{ github.ref_name }}
        run: |
          (cd dist && sha256sum *) > checksums.txt
          gh release upload "$TAG" --repo "$GITHUB_REPOSITORY" dist/* checksums.txt --clobber
```

## Why not release-drafter or semantic-release

Both are third-party automation (a marketplace action; an npm toolchain
with plugins) that take write access to releases. They are reasonable
choices a user may explicitly make — record the opt-in and pin versions
— but they are not this skill's default, which stays first-party and
keeps the publish step behind the review gate.
