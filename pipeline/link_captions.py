from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scipy.optimize import linear_sum_assignment

# OPTIONAL:
# pip install sentence-transformers
USE_EMBEDDINGS = True

if USE_EMBEDDINGS:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    EMBED_MODEL = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

REGIONS_PATH = Path(
    "extracted_regions/regions.json"
)

OUTPUT_DIR = REGIONS_PATH.parent

MAX_LINK_DISTANCE = 1200
CONFIDENCE_THRESHOLD = 0.22

# weights
W_VERTICAL = 0.30
W_HORIZONTAL = 0.12
W_OVERLAP = 0.20
W_SEMANTIC = 0.28
W_PLACEMENT = 0.10


# ─────────────────────────────────────────────────────────────
# Regex
# ─────────────────────────────────────────────────────────────

FIGURE_REGEX = re.compile(
    r"^(figure|fig\.?|scheme|chart)\s*([a-z0-9\-]+)",
    re.I
)

TABLE_REGEX = re.compile(
    r"^(table)\s*([a-z0-9\-]+)",
    re.I
)


# ─────────────────────────────────────────────────────────────
# Geometry Helpers
# ─────────────────────────────────────────────────────────────

def bbox_width(b):
    return b[2] - b[0]


def bbox_height(b):
    return b[3] - b[1]


def bbox_center_x(b):
    return (b[0] + b[2]) / 2


def bbox_center_y(b):
    return (b[1] + b[3]) / 2


def bbox_area(b):
    return bbox_width(b) * bbox_height(b)


def is_above(a, b):
    return a[3] <= b[1]


def is_below(a, b):
    return a[1] >= b[3]


def vertical_distance(a, b):

    if is_above(a, b):
        return b[1] - a[3]

    if is_below(a, b):
        return a[1] - b[3]

    return 0


def horizontal_overlap_ratio(a, b):

    overlap = max(
        0,
        min(a[2], b[2]) - max(a[0], b[0])
    )

    denom = min(
        bbox_width(a),
        bbox_width(b)
    )

    if denom <= 0:
        return 0

    return overlap / denom


# ─────────────────────────────────────────────────────────────
# Caption Classification
# ─────────────────────────────────────────────────────────────

def classify_caption(text: str):

    text = text.lower().strip()

    if TABLE_REGEX.match(text):
        return "table"

    if FIGURE_REGEX.match(text):
        return "figure"

    return None


def extract_caption_id(text: str):

    text = text.strip()

    m = FIGURE_REGEX.match(text)

    if m:
        return (
            "figure",
            m.group(2).lower()
        )

    m = TABLE_REGEX.match(text)

    if m:
        return (
            "table",
            m.group(2).lower()
        )

    return (None, None)


# ─────────────────────────────────────────────────────────────
# Text Helpers
# ─────────────────────────────────────────────────────────────

def normalize_text(text: str):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def keyword_overlap_score(a, b):

    a_words = set(
        normalize_text(a).split()
    )

    b_words = set(
        normalize_text(b).split()
    )

    if not a_words or not b_words:
        return 0

    overlap = len(a_words & b_words)

    return overlap / max(
        1,
        len(a_words)
    )


def semantic_similarity(a, b):

    if not a or not b:
        return 0

    if USE_EMBEDDINGS:

        emb_a = EMBED_MODEL.encode(a)
        emb_b = EMBED_MODEL.encode(b)

        return cosine_similarity(
            [emb_a],
            [emb_b]
        )[0][0]

    return keyword_overlap_score(a, b)


# ─────────────────────────────────────────────────────────────
# Column Detection
# ─────────────────────────────────────────────────────────────

def assign_columns(regions):

    x_centers = [
        bbox_center_x(r["bbox_pt"])
        for r in regions
    ]

    if not x_centers:
        return

    median_x = sorted(x_centers)[
        len(x_centers) // 2
    ]

    for r in regions:

        cx = bbox_center_x(
            r["bbox_pt"]
        )

        r["column"] = (
            0 if cx < median_x else 1
        )


# ─────────────────────────────────────────────────────────────
# Pairwise Score
# ─────────────────────────────────────────────────────────────

def compute_pair_score(
    caption,
    target,
):

    cb = caption["bbox_pt"]
    tb = target["bbox_pt"]

    vdist = vertical_distance(cb, tb)

    hdist = abs(
        bbox_center_x(cb)
        - bbox_center_x(tb)
    )

    overlap = horizontal_overlap_ratio(
        cb,
        tb
    )

    # normalize
    norm_vdist = min(
        1.0,
        vdist / MAX_LINK_DISTANCE
    )

    norm_hdist = min(
        1.0,
        hdist / 1500
    )

    # semantic
    cap_text = caption.get(
        "caption_text",
        ""
    )

    target_text = (
        target.get("ocr_text", "")
        + " "
        + target.get("text", "")
    )

    sem_score = semantic_similarity(
        cap_text,
        target_text
    )

    # placement priors
    placement_penalty = 0

    cap_kind = classify_caption(
        cap_text
    )

    if cap_kind == "figure":

        if is_above(cb, tb):
            placement_penalty += 1

    elif cap_kind == "table":

        if is_below(cb, tb):
            placement_penalty += 1

    # column mismatch
    column_penalty = 0

    if caption.get("column") != target.get("column"):
        column_penalty += 0.7

    # final cost
    cost = (
        norm_vdist * W_VERTICAL
        + norm_hdist * W_HORIZONTAL
        - overlap * W_OVERLAP
        - sem_score * W_SEMANTIC
        + placement_penalty * W_PLACEMENT
        + column_penalty
    )

    confidence = (
        overlap * 0.35
        + sem_score * 0.45
        + (1 - norm_vdist) * 0.20
    )

    return cost, confidence


# ─────────────────────────────────────────────────────────────
# Main Linking
# ─────────────────────────────────────────────────────────────

def link_regions(data):

    pages = defaultdict(list)

    for item in data:
        pages[item["page"]].append(item)

    for page_num, regions in pages.items():

        assign_columns(regions)

        figures = [
            r for r in regions
            if r["type"] == "figure"
        ]

        tables = [
            r for r in regions
            if r["type"] == "table"
        ]

        captions = [
            r for r in regions
            if r["type"] == "caption"
        ]

        if not captions:
            continue

        # build candidate lists
        caption_candidates = []

        for caption in captions:

            cap_text = caption.get(
                "caption_text",
                ""
            )

            kind = classify_caption(
                cap_text
            )

            if kind == "figure":
                candidates = figures

            elif kind == "table":
                candidates = tables

            else:
                candidates = (
                    figures + tables
                )

            caption_candidates.append(
                candidates
            )

        # create cost matrix
        max_candidates = max(
            len(c)
            for c in caption_candidates
        )

        cost_matrix = []

        flat_targets = (
            figures + tables
        )

        for caption in captions:

            row = []

            for target in flat_targets:

                cost, conf = compute_pair_score(
                    caption,
                    target
                )

                row.append(cost)

            cost_matrix.append(row)

        if not cost_matrix:
            continue

        # Hungarian optimization
        rows, cols = linear_sum_assignment(
            cost_matrix
        )

        for r, c in zip(rows, cols):

            caption = captions[r]
            target = flat_targets[c]

            cost, confidence = (
                compute_pair_score(
                    caption,
                    target
                )
            )

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            caption["linked_to"] = {
                "page": target["page"],
                "index": target["index"],
                "type": target["type"],
                "confidence": round(
                    confidence,
                    3
                ),
            }

            target["linked_caption"] = {
                "page": caption["page"],
                "index": caption["index"],
                "confidence": round(
                    confidence,
                    3
                ),
            }

    return data


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():

    if not REGIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {REGIONS_PATH}"
        )

    with open(
        REGIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    linked = link_regions(data)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    out_path = (
        OUTPUT_DIR
        / f"regions_linked_smart_{timestamp}.json"
    )

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            linked,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n✓ Smart linked file written:")
    print(out_path)


if __name__ == "__main__":
    main()