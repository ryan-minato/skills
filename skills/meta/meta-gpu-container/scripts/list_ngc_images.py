"""List and filter the container images currently listed in the NVIDIA NGC catalog.

Read-only and credential-free: queries the public NGC catalog search API and
prints one image per line (repository, display name, description).

Usage:
    python list_ngc_images.py --query pytorch
    python list_ngc_images.py --query jax --org nvidia --limit 10
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

SEARCH_URL = "https://api.ngc.nvidia.com/v2/search/catalog/resources/CONTAINER"
BROWSE_URL = "https://catalog.ngc.nvidia.com/containers"
PAGE_SIZE = 100


def search_page(query: str, page: int) -> dict:
    """Fetch one page of catalog search results, exiting with guidance on failure."""
    params = urllib.parse.quote(
        json.dumps({"query": query, "page": page, "pageSize": PAGE_SIZE})
    )
    request = urllib.request.Request(
        f"{SEARCH_URL}?q={params}", headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except json.JSONDecodeError:
        sys.exit(
            f"The NGC catalog returned a non-JSON response.\n"
            f"The unofficial search endpoint may have changed; "
            f"browse {BROWSE_URL} instead."
        )
    except urllib.error.HTTPError as error:
        sys.exit(
            f"NGC catalog search returned HTTP {error.code}.\n"
            f"The unofficial search endpoint may have changed; "
            f"browse {BROWSE_URL} instead."
        )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        sys.exit(
            f"Could not reach the NGC catalog ({error}).\n"
            f"Check network access, retry, or browse {BROWSE_URL} instead."
        )


def iter_images(query: str):
    """Yield container resources matching the query, following pagination."""
    page = 0
    while True:
        body = search_page(query, page)
        for group in body.get("results", []):
            yield from group.get("resources", [])
        page += 1
        if page >= body.get("resultPageTotal", 0):
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", required=True, help="search term, e.g. pytorch")
    parser.add_argument("--org", help="only images from this NGC org, e.g. nvidia")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="stop after this many images (0 = no limit, default 50)",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    shown = 0
    for image in iter_images(args.query):
        if args.org and image.get("orgName") != args.org:
            continue
        repository = image.get("resourceId", "?")
        display = image.get("displayName") or ""
        description = " ".join((image.get("description") or "").split())[:120]
        print(f"nvcr.io/{repository}\t{display}\t{description}")
        shown += 1
        if args.limit and shown >= args.limit:
            break

    if shown == 0:
        sys.exit(f"No images matched. Browse {BROWSE_URL}")


if __name__ == "__main__":
    main()
