#!/usr/bin/env python3
"""OKLCH color calculator for DESIGN.md authoring.

Compatible extra for the DESIGN.md format: the spec accepts oklch() but
recommends hex and converts every color to sRGB for WCAG contrast checks,
so wide-gamut values need a gamut check before they silently clip.

Subcommands (all non-interactive):
  to-hex   "oklch(62% 0.18 250)"          -> #rrggbb (errors if out of gamut)
  from-hex "#1a1c1e"                       -> oklch(...)
  gamut    "oklch(62% 0.29 27)"            -> in|out of sRGB + nearest in-gamut hex
  contrast "#1a1c1e" "oklch(97% 0.01 90)"  -> WCAG ratio + AA/AAA verdicts
"""

from __future__ import annotations

import argparse
import math
import re
import sys

_NUM = r"(?:\d+(?:\.\d+)?|\.\d+)"
OKLCH_RE = re.compile(
    r"^\s*(?:oklch\(\s*)?"
    rf"({_NUM}%?)\s+({_NUM})\s+({_NUM}(?:deg)?)"
    r"\s*(?:\))?\s*$",
    re.IGNORECASE,
)


def parse_oklch(text: str) -> tuple[float, float, float]:
    m = OKLCH_RE.match(text)
    if not m:
        sys.exit(
            f'error: cannot parse OKLCH value: {text!r} (expected like "oklch(62% 0.18 250)")'
        )
    lit, c_lit, h_lit = m.groups()
    if lit.endswith("%"):
        percent = float(lit[:-1])
        if percent > 100:
            sys.exit(f"error: lightness {lit} is out of range (0%-100%)")
        lightness = percent / 100
    else:
        lightness = float(lit)
        if lightness > 1:
            sys.exit(
                f"error: bare lightness {lit} is out of range (0-1) — did you mean {lit}%?"
            )
    hue = float(h_lit[:-3]) if h_lit.lower().endswith("deg") else float(h_lit)
    return lightness, float(c_lit), hue % 360


def parse_hex(text: str) -> tuple[float, float, float]:
    t = text.strip().lstrip("#")
    if len(t) in (3, 4):
        t = "".join(ch * 2 for ch in t)
    if len(t) not in (6, 8) or any(ch not in "0123456789abcdefABCDEF" for ch in t):
        sys.exit(f"error: cannot parse hex color: {text!r}")
    if len(t) == 8:
        if t[6:8].lower() != "ff":
            sys.exit(
                f"error: {text!r} is translucent (alpha {t[6:8]}) — "
                "composite it over its background first; this calculator "
                "only handles opaque colors"
            )
        t = t[:6]
    return tuple(int(t[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


_HEX_BODY_RE = re.compile(
    r"[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8}"
)


def parse_color(text: str) -> tuple[float, float, float]:
    """Return linear-light sRGB (possibly out of [0,1]) from hex or oklch."""
    t = text.strip()
    if t.startswith("#") or _HEX_BODY_RE.fullmatch(t):
        return tuple(srgb_to_linear(v) for v in parse_hex(t))  # type: ignore[return-value]
    if not OKLCH_RE.match(t):
        sys.exit(
            f"error: cannot parse color {text!r} — expected a hex color "
            'like "#1a2b3c" or an OKLCH value like "oklch(62% 0.18 250)"'
        )
    return oklch_to_linear_srgb(*parse_oklch(t))


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def oklch_to_linear_srgb(lightness: float, chroma: float, hue: float):
    h = math.radians(hue)
    a, b = chroma * math.cos(h), chroma * math.sin(h)
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    lm, mm, sm = l_**3, m_**3, s_**3
    return (
        +4.0767416621 * lm - 3.3077115913 * mm + 0.2309699292 * sm,
        -1.2684380046 * lm + 2.6097574011 * mm - 0.3413193965 * sm,
        -0.0041960863 * lm - 0.7034186147 * mm + 1.7076147010 * sm,
    )


def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1 / 3), x)


def linear_srgb_to_oklch(r: float, g: float, b: float):
    lm = _cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    mm = _cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    sm = _cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    okl = 0.2104542553 * lm + 0.7936177850 * mm - 0.0040720468 * sm
    oka = 1.9779984951 * lm - 2.4285922050 * mm + 0.4505937099 * sm
    okb = 0.0259040371 * lm + 0.7827717662 * mm - 0.8086757660 * sm
    chroma = math.hypot(oka, okb)
    hue = math.degrees(math.atan2(okb, oka)) % 360
    return okl, chroma, hue


def in_gamut(rgb, eps: float = 3e-3) -> bool:
    # eps absorbs the wobble of decimal-rounded boundary colors (a color
    # printed with 4-decimal chroma can sit ~1e-3 outside in linear light)
    # while still catching every visibly clipping value.
    return all(-eps <= v <= 1 + eps for v in rgb)


def to_hex(rgb) -> str:
    return "#" + "".join(
        f"{round(255 * min(1.0, max(0.0, linear_to_srgb(v)))):02x}" for v in rgb
    )


def clamp_chroma(lightness: float, chroma: float, hue: float):
    """Binary-search the largest in-gamut chroma at this L and H."""
    lo, hi = 0.0, chroma
    for _ in range(40):
        mid = (lo + hi) / 2
        if in_gamut(oklch_to_linear_srgb(lightness, mid, hue)):
            lo = mid
        else:
            hi = mid
    return lo


def relative_luminance(rgb) -> float:
    r, g, b = (min(1.0, max(0.0, v)) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def fmt_oklch(lightness: float, chroma: float, hue: float) -> str:
    return f"oklch({lightness * 100:.2f}% {chroma:.4f} {hue:.2f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("to-hex").add_argument("value")
    sub.add_parser("from-hex").add_argument("value")
    sub.add_parser("gamut").add_argument("value")
    contrast = sub.add_parser("contrast")
    contrast.add_argument("color_a")
    contrast.add_argument("color_b")
    args = parser.parse_args()

    if args.cmd == "to-hex":
        lch = parse_oklch(args.value)
        rgb = oklch_to_linear_srgb(*lch)
        if not in_gamut(rgb):
            sys.exit(
                f"error: {fmt_oklch(*lch)} is outside sRGB and would clip; "
                f"nearest in-gamut: {fmt_oklch(lch[0], clamp_chroma(*lch), lch[2])} "
                f"= {to_hex(oklch_to_linear_srgb(lch[0], clamp_chroma(*lch), lch[2]))}"
            )
        print(to_hex(rgb))
    elif args.cmd == "from-hex":
        rgb = tuple(srgb_to_linear(v) for v in parse_hex(args.value))
        print(fmt_oklch(*linear_srgb_to_oklch(*rgb)))
    elif args.cmd == "gamut":
        lch = parse_oklch(args.value)
        rgb = oklch_to_linear_srgb(*lch)
        if in_gamut(rgb):
            print(f"in sRGB gamut: {fmt_oklch(*lch)} = {to_hex(rgb)}")
        else:
            c = clamp_chroma(*lch)
            print(
                f"OUT of sRGB gamut: {fmt_oklch(*lch)} — the linter's sRGB "
                f"conversion will clip it.\nnearest in-gamut: "
                f"{fmt_oklch(lch[0], c, lch[2])} = "
                f"{to_hex(oklch_to_linear_srgb(lch[0], c, lch[2]))}"
            )
            sys.exit(1)
    elif args.cmd == "contrast":
        rgb_a = parse_color(args.color_a)
        rgb_b = parse_color(args.color_b)
        for label, rgb in (("first", rgb_a), ("second", rgb_b)):
            if not in_gamut(rgb):
                print(
                    f"warning: the {label} color is outside sRGB — the "
                    "ratio below is computed on the clipped color",
                    file=sys.stderr,
                )
        la = relative_luminance(rgb_a)
        lb = relative_luminance(rgb_b)
        ratio = (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
        verdict = (
            f"{ratio:.2f}:1 — "
            f"AA normal {'pass' if ratio >= 4.5 else 'FAIL'}, "
            f"AA large {'pass' if ratio >= 3.0 else 'FAIL'}, "
            f"AAA normal {'pass' if ratio >= 7.0 else 'FAIL'}"
        )
        print(verdict)


if __name__ == "__main__":
    main()
