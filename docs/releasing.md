# Releasing

Nothing is published until a maintainer pushes a version tag. `release.yml` builds, checks and publishes `dod-lineage` to PyPI on a `v*` tag, with trusted publishing, so no token is stored in the repository.

## One-time setup (a maintainer, on pypi.org)

1. Sign in to PyPI and open **Your projects → Publishing → Add a new pending publisher**.
2. Fill in:
   - PyPI project name: `dod-lineage`
   - Owner: `vitorizz`
   - Repository: `dod`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. In GitHub, under **Settings → Environments**, create `pypi`. Add required reviewers if a human should approve each upload.

## Each release

1. Make sure `main` is green.
2. Set `version` in `pyproject.toml`, and move the `CHANGELOG.md` entry from "unreleased" to the version and date.
3. Merge that, then tag the merge commit:
   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```
4. The workflow checks that the tag matches `pyproject.toml`'s version, builds the sdist and wheel, runs `twine check`, and publishes.

Check locally before tagging:
```bash
pip install build twine && python -m build && twine check dist/*
```
- The wheel holds only `dod`.
- The sdist holds the Python package, its tests, the demo they run, the examples, the scripts and the docs.

## TypeScript

`@dod-lineage/capture` (in `ts/`) isn't published by this workflow. To publish it by hand: `cd ts && npm ci && npm test && npm publish --access public`. You need an npm account with access to the `@dod-lineage` scope.
