---
version: alpha
name: <design system name>
colors:
  primary: "<#hex>"
  secondary: "<#hex>"
  neutral: "<#hex>"
typography:
  headline-lg:
    fontFamily: <family>
    fontSize: <px>
    fontWeight: <number>
    lineHeight: <multiplier>
  body-md:
    fontFamily: <family>
    fontSize: <px>
    fontWeight: <number>
    lineHeight: <multiplier>
spacing:
  sm: <px>
  md: <px>
  lg: <px>
rounded:
  sm: <px>
  md: <px>
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.neutral}"
    rounded: "{rounded.md}"
    padding: <px>
---

# <Design System Name>

## Overview

<The brand personality, target audience, and the feel the UI should
evoke — the context that guides decisions no token covers.>

## Colors

<What each palette means and where it is used; descriptive names may
appear here as long as they correspond to the tokens above.>

## Typography

<The families, their roles, and the voice each weight carries.>

## Layout

<The grid or spacing strategy and its rhythm.>

## Elevation & Depth

<How hierarchy is conveyed — shadows, tonal layers, or the flat-design
alternative.>

## Shapes

<The corner language and what it signals.>

## Components

<Per-component guidance: buttons, inputs, and whatever the domain adds.>

## Do's and Don'ts

- Do <the rule that protects the design>
- Don't <the pitfall that keeps recurring>
