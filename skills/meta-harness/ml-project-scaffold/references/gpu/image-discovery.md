# Image Discovery Without the Scripts

Read when a bundled script fails, the registry is not covered by them, or a
result needs manual confirmation.

## NGC catalog

- Web: <https://catalog.ngc.nvidia.com/> — filter by "Containers"; each
  image page lists tags, sizes, and release notes.
- NGC CLI: `ngc registry image list` and `ngc registry image info`
  enumerate the same inventory from a terminal; guest access covers public
  images. Install and current syntax: <https://docs.ngc.nvidia.com/cli/>.
- The search endpoint the bundled script calls is the one the catalog web
  UI uses, not a formally documented API — when it changes, the web catalog
  and the CLI are the stable paths.

## Docker Hub

- Web: each repository's Tags page (e.g.
  <https://hub.docker.com/r/pytorch/pytorch>) supports name filtering.
- API: the documented Docker Hub API covers repository and tag listing;
  reference: <https://docs.docker.com/reference/api/hub/latest/>.

## Any other registry (private mirrors, self-hosted)

Registries speak the OCI/Docker Registry v2 protocol — token handshake,
then `GET /v2/<name>/tags/list`; spec and reference implementation:
<https://distribution.github.io/distribution/>. `docker manifest inspect
<image>:<tag>` confirms a specific tag exists and shows its platforms
without pulling.

## Reading tag inventories

- NGC framework images tag monthly as `yy.mm-py3`; the CUDA and driver
  versions inside each monthly tag are published in the support matrix, not
  in the tag name.
- Docker Hub framework images commonly encode their stack as
  `<version>-cuda<x>-cudnn<y>-<runtime|devel>`; treat the pattern as a
  reading aid, and confirm against the listing rather than constructing
  tags from it.
- Registry listings include non-runnable artifacts (attestations like
  `sha256-*.sig`, cache manifests); ignore anything that does not look like
  a release tag.
