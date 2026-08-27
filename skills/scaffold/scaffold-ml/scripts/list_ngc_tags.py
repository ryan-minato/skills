"""List and filter the currently available tags of an NGC container image.

Read-only and credential-free for public images: obtains an anonymous pull
token from nvcr.io, lists tags through the standard Docker Registry v2 API,
and prints one tag per line, newest-looking first (reverse lexicographic).
Attestation artifacts (sha256-*.sig/.sbom/.vex) are filtered out.

Usage:
    python list_ngc_tags.py nvidia/pytorch --limit 10
    python list_ngc_tags.py nvidia/cuda --filter 'devel'
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

REGISTRY = "https://nvcr.io"
BROWSE_URL = "https://catalog.ngc.nvidia.com/containers"
ATTESTATION = re.compile(r"^sha256-[0-9a-f]{64}\.")


def fetch_json(url: str, headers: dict | None = None) -> dict:
    """GET a URL and decode the JSON body, exiting with guidance on failure."""
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except json.JSONDecodeError:
        sys.exit(
            f"nvcr.io returned a non-JSON response for {url}\n"
            f"The registry may have changed or a proxy intercepted the "
            f"request; browse {BROWSE_URL} instead."
        )
    except urllib.error.HTTPError as error:
        sys.exit(
            f"nvcr.io returned HTTP {error.code} for {url}\n"
            f"Check the image name (an <org>/<repo> from the catalog), or "
            f"browse {BROWSE_URL} instead. Non-public images need credentials."
        )
    except urllib.error.URLError as error:
        sys.exit(
            f"Could not reach nvcr.io ({error.reason}).\n"
            f"Check network access, or browse {BROWSE_URL} instead."
        )


def require_field(body: dict, field: str, url: str):
    """Return body[field], exiting with guidance when the response lacks it.

    A 200 response carrying JSON without the field reaches here: a changed
    endpoint, or a proxy answering with its own body. Without this the caller
    raises a bare KeyError, the one failure of this script that would print a
    traceback instead of saying where to go next.
    """
    if not isinstance(body, dict) or field not in body:
        sys.exit(
            f"nvcr.io returned a response without a {field!r} field for {url}\n"
            f"The registry API may have changed, or a proxy answered instead; "
            f"browse {BROWSE_URL} instead."
        )
    return body[field]


def list_tags(image: str) -> list[str]:
    """Return all tags of a public nvcr.io image via an anonymous pull token."""
    auth_url = f"{REGISTRY}/proxy_auth?scope=repository:{image}:pull"
    token = require_field(  # pragma: allowlist secret
        fetch_json(auth_url), "token", auth_url
    )
    body = fetch_json(
        f"{REGISTRY}/v2/{image}/tags/list",
        headers={"Authorization": f"Bearer {token}"},
    )
    return body.get("tags") or []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="image as <org>/<repo>, e.g. nvidia/pytorch")
    parser.add_argument("--filter", help="regex a tag must match")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="stop after this many matching tags (0 = no limit, default 50)",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    pattern = None
    if args.filter:
        try:
            pattern = re.compile(args.filter)
        except re.error as error:
            parser.error(f"invalid --filter regex: {error}")

    tags = [tag for tag in list_tags(args.image) if not ATTESTATION.match(tag)]
    if pattern:
        tags = [tag for tag in tags if pattern.search(tag)]
    tags.sort(reverse=True)
    if args.limit:
        tags = tags[: args.limit]

    if not tags:
        sys.exit(f"No tags matched. Browse {BROWSE_URL}")
    print("\n".join(tags))


if __name__ == "__main__":
    main()
