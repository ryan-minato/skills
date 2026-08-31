"""List and filter the currently available tags of a Docker Hub repository.

Read-only and credential-free: queries the public Docker Hub API and prints
one tag per line (name, last push date, platforms), newest first.

Usage:
    python list_dockerhub_tags.py pytorch/pytorch --filter 'devel$' --limit 10
    python list_dockerhub_tags.py python            # official image (library/)
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

API_ROOT = "https://hub.docker.com/v2/repositories"
BROWSE_URL = "https://hub.docker.com/search?q="


def fetch_json(url: str) -> dict:
    """GET a URL and decode the JSON body, exiting with guidance on failure."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except json.JSONDecodeError:
        sys.exit(
            f"Docker Hub returned a non-JSON response for {url}\n"
            f"The API may have changed or a proxy intercepted the request; "
            f"browse {BROWSE_URL}<name> instead."
        )
    except urllib.error.HTTPError as error:
        sys.exit(
            f"Docker Hub returned HTTP {error.code} for {url}\n"
            f"Check the repository name, or browse {BROWSE_URL}<name> instead."
        )
    except urllib.error.URLError as error:
        sys.exit(
            f"Could not reach Docker Hub ({error.reason}).\n"
            f"Check network access, or browse {BROWSE_URL}<name> instead."
        )
    except TimeoutError:
        sys.exit(
            f"Docker Hub timed out answering {url}\n"
            f"Retry, or browse {BROWSE_URL}<name> instead."
        )


def iter_tags(repository: str):
    """Yield tag entries for the repository, following API pagination.

    A page without `results` is an unexpected body — a changed API or a proxy
    answering instead — and exits with guidance rather than a bare KeyError,
    the one failure here that would otherwise print a traceback.
    """
    url = f"{API_ROOT}/{repository}/tags/?page_size=100"
    while url:
        page = fetch_json(url)
        if not isinstance(page, dict) or "results" not in page:
            sys.exit(
                f"Docker Hub returned a response without a 'results' field "
                f"for {url}\nThe API may have changed, or a proxy answered "
                f"instead; browse {BROWSE_URL}<name> instead."
            )
        yield from page["results"]
        url = page.get("next")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "repository",
        help="repository as <namespace>/<name>; a bare <name> means library/<name>",
    )
    parser.add_argument("--filter", help="regex a tag name must match")
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

    repository = (
        args.repository if "/" in args.repository else f"library/{args.repository}"
    )

    shown = 0
    for tag in iter_tags(repository):
        name = tag.get("name") if isinstance(tag, dict) else None
        if not name:
            continue
        if pattern and not pattern.search(name):
            continue
        pushed = (tag.get("tag_last_pushed") or "")[:10]
        platforms = ",".join(
            f"{image.get('os')}/{image.get('architecture')}"
            for image in tag.get("images", [])
        )
        print(f"{name}\t{pushed}\t{platforms}")
        shown += 1
        if args.limit and shown >= args.limit:
            break

    if shown == 0:
        sys.exit(f"No tags matched. Browse https://hub.docker.com/r/{repository}/tags")


if __name__ == "__main__":
    main()
