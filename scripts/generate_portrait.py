#!/usr/bin/env python3
"""Turn a portrait photo into a self-typing ASCII SVG for a GitHub README.

The generated SVG is self-contained: the font is inlined, the background is
transparent, and the animation uses SMIL (which GitHub permits in SVG images).
"""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import cv2
import numpy as np


RAMP = "@%#sc*+=-:.` "
FONT_SIZE = 12.9
CHAR_WIDTH = 7.74  # JetBrains Mono advances exactly 0.600 em.
LINE_HEIGHT = 15
PADDING = 14


def detect_crop(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find the largest face and return a head-and-shoulders crop."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=6, minSize=(100, 100)
    )
    if len(faces) == 0:
        raise SystemExit("No face found. Pass --crop x,y,width,height explicitly.")

    fx, fy, fw, fh = max(faces, key=lambda box: box[2] * box[3])
    crop_width = int(fw * 5.0)
    crop_height = int(fh * 5.4)
    center_x = fx + fw // 2
    x = max(0, min(image.shape[1] - crop_width, center_x - crop_width // 2))
    y = max(0, min(image.shape[0] - crop_height, fy - int(fh * 1.05)))
    return x, y, crop_width, crop_height


def parse_crop(value: str) -> tuple[int, int, int, int]:
    try:
        crop = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height") from error
    if len(crop) != 4 or crop[2] <= 0 or crop[3] <= 0:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height")
    return crop  # type: ignore[return-value]


def isolate_subject(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use face/torso seeds with GrabCut, then soften the resulting edge."""
    target_width = 720
    target_height = max(1, round(crop.shape[0] * target_width / crop.shape[1]))
    image = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]

    mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    silhouette = np.array(
        [
            [int(width * 0.38), int(height * 0.04)],
            [int(width * 0.62), int(height * 0.04)],
            [int(width * 0.66), int(height * 0.20)],
            [int(width * 0.82), int(height * 0.30)],
            [int(width * 0.88), height - 1],
            [int(width * 0.12), height - 1],
            [int(width * 0.18), int(height * 0.30)],
            [int(width * 0.34), int(height * 0.20)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [silhouette], cv2.GC_PR_FGD)

    # Certain-foreground seeds: face center and central torso, never background.
    cv2.ellipse(
        mask,
        (width // 2, int(height * 0.25)),
        (int(width * 0.075), int(height * 0.09)),
        0,
        0,
        360,
        cv2.GC_FGD,
        -1,
    )
    cv2.rectangle(
        mask,
        (int(width * 0.39), int(height * 0.45)),
        (int(width * 0.61), int(height * 0.82)),
        cv2.GC_FGD,
        -1,
    )

    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        image,
        mask,
        None,
        background,
        foreground,
        8,
        cv2.GC_INIT_WITH_MASK,
    )
    alpha = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Keep only the connected subject containing the center of the torso.
    _, labels, _, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8))
    subject_label = labels[min(height - 1, int(height * 0.60)), width // 2]
    if subject_label:
        alpha = np.where(labels == subject_label, 255, 0).astype(np.uint8)

    # GrabCut can cling to background objects that touch dark clothing. Limit
    # the result to a conservative head-and-shoulders envelope before feathering.
    envelope = np.zeros_like(alpha)
    outline = np.array(
        [
            [int(width * 0.42), int(height * 0.05)],
            [int(width * 0.58), int(height * 0.05)],
            [int(width * 0.63), int(height * 0.12)],
            [int(width * 0.63), int(height * 0.24)],
            [int(width * 0.58), int(height * 0.34)],
            [int(width * 0.68), int(height * 0.37)],
            [int(width * 0.78), int(height * 0.40)],
            [int(width * 0.88), int(height * 0.50)],
            [int(width * 0.88), height - 1],
            [int(width * 0.12), height - 1],
            [int(width * 0.12), int(height * 0.50)],
            [int(width * 0.22), int(height * 0.40)],
            [int(width * 0.32), int(height * 0.37)],
            [int(width * 0.42), int(height * 0.34)],
            [int(width * 0.37), int(height * 0.24)],
            [int(width * 0.37), int(height * 0.12)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(envelope, [outline], 255)
    alpha = cv2.bitwise_and(alpha, envelope)
    alpha[int(height * 0.28) : int(height * 0.42), : int(width * 0.45)] = 0
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (9, 9), 0)
    return image, alpha


def ascii_rows(image: np.ndarray, alpha: np.ndarray, columns: int) -> list[str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 55, 55)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    gray = np.uint8(np.power(gray / 255.0, 1.90) * 255)
    gray = np.where(alpha > 18, gray, 255).astype(np.uint8)

    rows = max(1, round(columns * image.shape[0] / image.shape[1] * 0.48))
    small = cv2.resize(gray, (columns, rows), interpolation=cv2.INTER_AREA)
    indexes = np.clip(
        np.rint(small.astype(np.float32) / 255 * (len(RAMP) - 1)),
        0,
        len(RAMP) - 1,
    ).astype(np.uint8)
    return ["".join(RAMP[index] for index in row).rstrip() for row in indexes]


def svg_for(rows: list[str], font_path: Path) -> str:
    font = base64.b64encode(font_path.read_bytes()).decode("ascii")
    width = round(max((len(row) for row in rows), default=1) * CHAR_WIDTH + PADDING * 2)
    height = len(rows) * LINE_HEIGHT + PADDING * 2
    family = (
        "JBMono,ui-monospace,SFMono-Regular,Menlo,Consolas,"
        "'Liberation Mono',monospace"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{family}" role="img" '
        'aria-labelledby="title desc">',
        "<title id=\"title\">Vivaan Shahani, drawn as a typing ASCII portrait</title>",
        "<desc id=\"desc\">An animated character portrait that types itself row by row.</desc>",
        "<style>",
        "@font-face{font-family:JBMono;font-style:normal;font-weight:400;"
        f"font-display:block;src:url(data:font/woff2;base64,{font}) format('woff2')}}",
        ".ink{fill:#24292f}@media(prefers-color-scheme:dark){.ink{fill:#c9d1d9}}",
        "</style>",
    ]

    for index, row in enumerate(rows):
        y = PADDING + index * LINE_HEIGHT
        baseline = y + 11.2
        reveal_width = max(CHAR_WIDTH, len(row) * CHAR_WIDTH)
        start = index * 0.09
        end = start + 0.09
        escaped = html.escape(row, quote=False)
        parts.extend(
            [
                f'<clipPath id="row{index}"><rect x="{PADDING}" y="{y}" '
                f'height="{LINE_HEIGHT}" width="0"><animate attributeName="width" '
                f'from="0" to="{reveal_width:.1f}" begin="{start:.2f}s" dur="0.09s" '
                'fill="freeze"/></rect></clipPath>',
                f'<g clip-path="url(#row{index})"><text xml:space="preserve" '
                f'x="{PADDING}" y="{baseline:.1f}" class="ink" font-size="{FONT_SIZE}">'
                f"{escaped}</text></g>",
                f'<rect y="{y + 1}" width="6" height="12" class="ink" opacity="0">'
                f'<animate attributeName="x" from="{PADDING}" '
                f'to="{PADDING + reveal_width:.1f}" begin="{start:.2f}s" dur="0.09s" '
                f'fill="freeze"/><set attributeName="opacity" to="0.8" '
                f'begin="{start:.2f}s"/><set attributeName="opacity" to="0" '
                f'begin="{end:.2f}s"/></rect>',
            ]
        )
    parts.append("</svg>\n")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="source portrait (PNG or JPEG)")
    parser.add_argument("--output", type=Path, default=Path("assets/ascii-portrait.svg"))
    parser.add_argument("--columns", type=int, default=90)
    parser.add_argument("--crop", type=parse_crop, help="x,y,width,height; otherwise face-detected")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not read {args.image}")
    x, y, width, height = args.crop or detect_crop(image)
    crop = image[y : y + height, x : x + width]
    subject, alpha = isolate_subject(crop)
    rows = ascii_rows(subject, alpha, args.columns)

    font = Path(__file__).parent / "fonts" / "jbmono-ramp.woff2"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg_for(rows, font), encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} rows, {args.columns} columns)")


if __name__ == "__main__":
    main()
