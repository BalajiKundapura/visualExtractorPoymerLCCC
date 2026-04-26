from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pymupdf as fitz
from pdf2image import convert_from_path
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QColor, QFont, QImage, QPainter, QPen,
                          QPixmap, QBrush, QShortcut, QKeySequence)
from PyQt6.QtWidgets import (QApplication, QGraphicsPixmapItem,
                              QGraphicsRectItem, QGraphicsScene,
                              QGraphicsView, QHBoxLayout, QLabel,
                              QMainWindow, QMessageBox, QPushButton,
                              QSizePolicy, QStatusBar, QToolBar, QVBoxLayout,
                              QWidget)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH = (
    "/Users/balaji/Desktop/Polymer LCC Project /"
    "figure-extraction-pipeline/Inputs/polymerPaper1.pdf"
)
OUTPUT_DIR = (
    "/Users/balaji/Desktop/Polymer LCC Project /"
    "figure-extraction-pipeline/extracted_regions"
)

RENDER_DPI       = 150
MIN_AREA_PT2     = 2_500   # ~50×50 pt minimum
MERGE_GAP_PT     = 14      # vector-path merge radius
TABLE_MIN_COLS   = 3       # distinct x-buckets to qualify a row
ROW_BAND_SIZE    = 8       # pt — y-bucket granularity for rows
COL_BUCKET_SIZE  = 20      # pt — x-bucket granularity for columns
TABLE_GAP_PT     = 6       # max vertical gap between rows of the same table
CAPTION_SEARCH   = 80      # pt above/below figure for caption search
TABLE_CAP_SEARCH = 60      # pt above/below table for "Table N" caption

_SCALE = RENDER_DPI / 72.0

PALETTE: dict[str, QColor] = {
    "figure":  QColor("#00e676"),
    "drawing": QColor("#40c4ff"),
    "table":   QColor("#ffd740"),
    "manual":  QColor("#ff6d00"),
}

_FIG_CAPTION_RE = re.compile(
    r"(fig(ure|\.)?|scheme|plate|chart)\s*[\d]+",
    re.IGNORECASE,
)
_TBL_CAPTION_RE = re.compile(r"table\s*[\d]+", re.IGNORECASE)


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Region:
    rtype:         str
    bbox:          List[float]       # [x0, y0, x1, y1] PDF points
    source:        str   = "auto"
    caption:       str   = ""        # figure/drawing caption text
    table_caption: str   = ""        # "Table N. …" line

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, (x1 - x0) * (y1 - y0))

    def scaled_rect(self) -> QRectF:
        x0, y0, x1, y1 = self.bbox
        return QRectF(x0 * _SCALE, y0 * _SCALE,
                      (x1 - x0) * _SCALE, (y1 - y0) * _SCALE)

    def colour(self) -> QColor:
        return PALETTE["manual"] if self.source == "manual" \
               else PALETTE.get(self.rtype, QColor("white"))


# ── Detection helpers ─────────────────────────────────────────────────────────
def _merge_rects(rects: list, gap: float = MERGE_GAP_PT) -> list:
    if not rects:
        return []
    changed = True
    while changed:
        changed = False
        out, used = [], [False] * len(rects)
        for i, r1 in enumerate(rects):
            if used[i]:
                continue
            cur = fitz.Rect(r1)
            for j, r2 in enumerate(rects[i + 1:], start=i + 1):
                if used[j]:
                    continue
                if (cur + (-gap, -gap, gap, gap)).intersects(r2):
                    cur |= r2
                    used[j] = True
                    changed = True
            out.append(cur)
            used[i] = True
        rects = out
    return rects


def _is_dup(bbox: list, seen: list, tol: float = 10.0) -> bool:
    r = fitz.Rect(bbox)
    return any(
        abs(s.x0 - r.x0) < tol and abs(s.y0 - r.y0) < tol
        and abs(s.x1 - r.x1) < tol and abs(s.y1 - r.y1) < tol
        for s in seen
    )


def _block_text(block: dict) -> str:
    lines = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            lines.append(span.get("text", "").strip())
    return " ".join(lines).strip()


def _find_caption(text_blocks: list, ref_y0: float, ref_y1: float,
                  search_pt: float, pattern: re.Pattern) -> str:
    """
    Search text blocks within search_pt above ref_y0 OR below ref_y1.
    Returns the closest block whose text matches pattern; falls back to
    the nearest text block if nothing matches the pattern.
    """
    above, below = [], []
    for blk in text_blocks:
        if blk.get("type") != 0:
            continue
        by0, by1 = blk["bbox"][1], blk["bbox"][3]
        txt = _block_text(blk)
        if not txt:
            continue
        # above the region
        if ref_y0 - search_pt <= by1 <= ref_y0 + 4:
            above.append((abs(ref_y0 - by1), txt))
        # below the region
        elif ref_y1 - 4 <= by0 <= ref_y1 + search_pt:
            below.append((abs(by0 - ref_y1), txt))

    candidates = sorted(above, key=lambda x: x[0]) + \
                 sorted(below, key=lambda x: x[0])

    # prefer a pattern match
    for _, txt in candidates:
        if pattern.search(txt):
            return txt
    # fallback: nearest text
    return candidates[0][1] if candidates else ""


# ── Full-table detection ───────────────────────────────────────────────────────
def _detect_tables(page: fitz.Page, text_blocks: list,
                   seen: list) -> list[Region]:
    """
    1. Bucket every text block by its y-coordinate (ROW_BAND_SIZE granularity).
    2. Keep only bands with >= TABLE_MIN_COLS distinct x-column buckets.
    3. Sort qualifying rows top-to-bottom and merge those within TABLE_GAP_PT
       of each other — this captures ALL rows, not just the header.
    4. Search ±TABLE_CAP_SEARCH pt for a "Table N" caption.
    """
    bands: dict[int, list] = {}
    for blk in text_blocks:
        if blk.get("type") != 0:
            continue
        key = round(blk["bbox"][1] / ROW_BAND_SIZE)
        bands.setdefault(key, []).append(blk)

    qualifying_rows: list[fitz.Rect] = []
    for row_blocks in bands.values():
        distinct_cols = len({round(b["bbox"][0] / COL_BUCKET_SIZE)
                             for b in row_blocks})
        if distinct_cols < TABLE_MIN_COLS:
            continue
        row_bbox = [
            min(b["bbox"][0] for b in row_blocks),
            min(b["bbox"][1] for b in row_blocks),
            max(b["bbox"][2] for b in row_blocks),
            max(b["bbox"][3] for b in row_blocks),
        ]
        qualifying_rows.append(fitz.Rect(row_bbox))

    if not qualifying_rows:
        return []

    qualifying_rows.sort(key=lambda r: r.y0)

    # Merge contiguous rows into table groups
    groups: list[list[fitz.Rect]] = []
    cur_group = [qualifying_rows[0]]
    for row in qualifying_rows[1:]:
        if row.y0 - cur_group[-1].y1 <= TABLE_GAP_PT:
            cur_group.append(row)
        else:
            groups.append(cur_group)
            cur_group = [row]
    groups.append(cur_group)

    regions: list[Region] = []
    for group in groups:
        bbox = [
            min(r.x0 for r in group), min(r.y0 for r in group),
            max(r.x1 for r in group), max(r.y1 for r in group),
        ]
        r = fitz.Rect(bbox)
        if r.width * r.height < MIN_AREA_PT2 or _is_dup(bbox, seen):
            continue
        cap = _find_caption(text_blocks, bbox[1], bbox[3],
                            TABLE_CAP_SEARCH, _TBL_CAPTION_RE)
        regions.append(Region("table", bbox, table_caption=cap))
        seen.append(r)
        log.debug("  table rows=%d  caption=%r", len(group), cap[:60])

    return regions


# ── Main detection entry point ────────────────────────────────────────────────
def detect_regions(pdf_path: str) -> list[list[Region]]:
    doc   = fitz.open(pdf_path)
    pages = []

    for pi, page in enumerate(doc):
        regions: list[Region] = []
        seen:    list         = []

        all_blocks  = page.get_text("dict",
                                    flags=fitz.TEXT_PRESERVE_IMAGES)["blocks"]
        text_blocks = [b for b in all_blocks if b.get("type") == 0]

        # Pass 1 — raster images
        for blk in all_blocks:
            if blk.get("type") != 1:
                continue
            bb = blk["bbox"]
            r  = fitz.Rect(bb)
            if r.width * r.height < MIN_AREA_PT2 or _is_dup(bb, seen):
                continue
            cap = _find_caption(text_blocks, bb[1], bb[3],
                                CAPTION_SEARCH, _FIG_CAPTION_RE)
            regions.append(Region("figure", list(bb), caption=cap))
            seen.append(r)

        # Pass 2 — vector drawings
        raw = [
            fitz.Rect(d["rect"]) for d in page.get_drawings()
            if fitz.Rect(d["rect"]).width
               * fitz.Rect(d["rect"]).height >= MIN_AREA_PT2
        ]
        for r in _merge_rects(raw):
            bb = [r.x0, r.y0, r.x1, r.y1]
            if r.width * r.height < MIN_AREA_PT2 or _is_dup(bb, seen):
                continue
            cap = _find_caption(text_blocks, bb[1], bb[3],
                                CAPTION_SEARCH, _FIG_CAPTION_RE)
            regions.append(Region("drawing", bb, caption=cap))
            seen.append(r)

        # Pass 3 — full tables
        table_regs = _detect_tables(page, text_blocks, seen)
        regions.extend(table_regs)

        log.info("Page %d — %d region(s)  (%d table(s))",
                 pi + 1, len(regions), len(table_regs))
        pages.append(regions)

    doc.close()
    return pages


# ── Page rendering ────────────────────────────────────────────────────────────
def render_page(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> QPixmap:
    pil_images = convert_from_path(
        pdf_path, dpi=dpi,
        first_page=page_num + 1, last_page=page_num + 1,
    )
    pil = pil_images[0].convert("RGB")
    w, h = pil.size
    qimg = QImage(pil.tobytes("raw", "RGB"), w, h,
                  w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ── RegionItem ────────────────────────────────────────────────────────────────
class RegionItem(QGraphicsRectItem):
    def __init__(self, region: Region, idx: int):
        super().__init__(region.scaled_rect())
        self.region = region
        self.idx    = idx
        self._apply_style(False)
        self.setAcceptHoverEvents(True)

        tip_parts = []
        if region.caption:
            tip_parts.append(f"Caption: {region.caption}")
        if region.table_caption:
            tip_parts.append(f"Table caption: {region.table_caption}")
        if tip_parts:
            self.setToolTip("\n\n".join(tip_parts))

    def _apply_style(self, selected: bool):
        c    = self.region.colour()
        pen  = QPen(c, 3 if selected else 2,
                    Qt.PenStyle.SolidLine if selected else Qt.PenStyle.DashLine)
        fill = QColor(c)
        fill.setAlpha(40 if selected else 18)
        self.setPen(pen)
        self.setBrush(QBrush(fill))

    def set_selected_style(self, sel: bool):
        self._apply_style(sel)

    def paint(self, painter: QPainter, option, widget=None):
        super().paint(painter, option, widget)
        r   = self.rect()
        c   = self.region.colour()
        cap = self.region.caption or self.region.table_caption
        lbl = f"[{self.idx + 1}] {self.region.rtype}"
        if cap:
            short = cap[:45].rstrip()
            lbl  += f"  —  {short}{'…' if len(cap) > 45 else ''}"
        fm    = painter.fontMetrics()
        tw    = fm.horizontalAdvance(lbl) + 10
        th    = fm.height() + 4
        badge = QRectF(r.x(), r.y(), tw, th)
        painter.fillRect(badge, c)
        painter.setPen(QPen(QColor("black")))
        painter.setFont(QFont("Helvetica", 8, QFont.Weight.Bold))
        painter.drawText(badge.adjusted(5, 2, -5, -2),
                         Qt.AlignmentFlag.AlignLeft, lbl)


# ── PDFScene ──────────────────────────────────────────────────────────────────
class PDFScene(QGraphicsScene):
    def __init__(self, editor: "RegionEditor"):
        super().__init__()
        self._editor  = editor
        self._drawing = False
        self._origin  = QPointF()
        self._rubber: Optional[QGraphicsRectItem] = None

    def mousePressEvent(self, ev):
        pos = ev.scenePos()
        for item in self.items(pos):
            if isinstance(item, RegionItem):
                self._editor.select_region(item)
                return
        self._editor.deselect()
        self._drawing = True
        self._origin  = pos
        self._rubber  = self.addRect(
            QRectF(pos, pos),
            QPen(PALETTE["manual"], 2, Qt.PenStyle.DashLine))

    def mouseMoveEvent(self, ev):
        if self._drawing and self._rubber:
            self._rubber.setRect(
                QRectF(self._origin, ev.scenePos()).normalized())

    def mouseReleaseEvent(self, ev):
        if not self._drawing:
            return
        self._drawing = False
        if self._rubber:
            self.removeItem(self._rubber)
            self._rubber = None
        rect = QRectF(self._origin, ev.scenePos()).normalized()
        reg  = Region("figure",
                      [rect.x()      / _SCALE, rect.y()        / _SCALE,
                       rect.right()  / _SCALE, rect.bottom()   / _SCALE],
                      source="manual")
        if reg.area >= MIN_AREA_PT2:
            self._editor.add_region(reg)


# ── Main window ───────────────────────────────────────────────────────────────
class RegionEditor(QMainWindow):
    def __init__(self, pdf_path: str, all_regions: list[list[Region]]):
        super().__init__()
        self.pdf_path    = pdf_path
        self.all_regions = all_regions
        self.num_pages   = len(all_regions)
        self.cur_page    = 0
        self._selected:  Optional[RegionItem] = None
        self._items:     list[RegionItem]     = []

        self.setWindowTitle("PDF Region Extractor  v3")
        self.resize(1400, 960)
        self._build_ui()
        self._goto_page(0)

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#1e1e1e; color:#e0e0e0; }
            QPushButton {
                background:#3c3c3c; color:white; border:none;
                padding:6px 14px; border-radius:4px; font-weight:bold;
            }
            QPushButton:hover   { background:#505050; }
            QPushButton:pressed { background:#222; }
            QLabel      { color:#e0e0e0; }
            QStatusBar  { background:#007acc; color:white; font-size:11px; }
            QGraphicsView { border:none; background:#111; }
            QToolBar    { background:#252526; border:none; spacing:4px; padding:4px; }
            QToolTip    { background:#2d2d2d; color:#e0e0e0;
                          border:1px solid #555; font-size:11px; padding:6px; }
        """)

        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        for text, slot in [("◀  Prev", self._prev_page),
                           ("Next  ▶", self._next_page)]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            tb.addWidget(btn)
            if text == "◀  Prev":
                self.page_label = QLabel("Page 1 / 1")
                self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.page_label.setMinimumWidth(150)
                tb.addWidget(self.page_label)

        tb.addSeparator()
        hint = QLabel("  ✏ Drag=add  |  Click=select  |  ← →  |  Del  |  "
                      "Ctrl+Wheel=zoom  |  Hover=caption")
        hint.setStyleSheet("color:#888; font-size:11px;")
        tb.addWidget(hint)

        sp = QWidget()
        sp.setSizePolicy(QSizePolicy.Policy.Expanding,
                         QSizePolicy.Policy.Preferred)
        tb.addWidget(sp)

        btn_del = QPushButton("🗑  Delete Selected")
        btn_del.setStyleSheet("background:#8b1a1a;")
        btn_del.clicked.connect(self._delete_selected)
        tb.addWidget(btn_del)

        btn_save = QPushButton("💾  Save & Export")
        btn_save.setStyleSheet("background:#1a6b3a;")
        btn_save.clicked.connect(self._save_all)
        tb.addWidget(btn_save)

        # Legend
        legend = QWidget()
        legend.setStyleSheet("background:#1e1e1e; padding:4px;")
        lh = QHBoxLayout(legend)
        lh.setContentsMargins(8, 2, 8, 2)
        lh.setSpacing(16)
        for name, color in PALETTE.items():
            sw = QLabel()
            sw.setFixedSize(14, 14)
            sw.setStyleSheet(f"background:{color.name()}; border-radius:3px;")
            lb = QLabel(name)
            lb.setStyleSheet(f"color:{color.name()}; font-size:11px;")
            lh.addWidget(sw)
            lh.addWidget(lb)
        lh.addStretch()

        self.scene = PDFScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.wheelEvent = self._wheel_zoom

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(legend)
        cl.addWidget(self.view)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        QShortcut(QKeySequence(Qt.Key.Key_Left),   self, self._prev_page)
        QShortcut(QKeySequence(Qt.Key.Key_Right),  self, self._next_page)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self._delete_selected)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.deselect)

    def _wheel_zoom(self, ev):
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.view.scale(f, f)

    def _goto_page(self, n: int):
        self.cur_page  = max(0, min(n, self.num_pages - 1))
        self._selected = None
        self._items    = []
        self.scene.clear()

        pixmap   = render_page(self.pdf_path, self.cur_page)
        pix_item = QGraphicsPixmapItem(pixmap)
        pix_item.setZValue(0)
        self.scene.addItem(pix_item)
        self.scene.setSceneRect(pix_item.boundingRect())

        for i, reg in enumerate(self.all_regions[self.cur_page]):
            self._add_region_item(reg, i)

        self.page_label.setText(
            f"Page {self.cur_page + 1} / {self.num_pages}")
        self._update_status()
        self.view.fitInView(self.scene.sceneRect(),
                            Qt.AspectRatioMode.KeepAspectRatio)

    def _add_region_item(self, reg: Region, idx: int) -> RegionItem:
        item = RegionItem(reg, idx)
        item.setZValue(1)
        self.scene.addItem(item)
        self._items.append(item)
        return item

    def _refresh_items(self):
        for it in self._items:
            self.scene.removeItem(it)
        self._items = []
        for i, reg in enumerate(self.all_regions[self.cur_page]):
            self._add_region_item(reg, i)
        self._update_status()

    def select_region(self, item: RegionItem):
        if self._selected and self._selected is not item:
            self._selected.set_selected_style(False)
            self._selected.update()
        self._selected = item
        item.set_selected_style(True)
        item.update()
        self._update_status()

    def deselect(self):
        if self._selected:
            self._selected.set_selected_style(False)
            self._selected.update()
            self._selected = None
        self._update_status()

    def add_region(self, reg: Region):
        self.all_regions[self.cur_page].append(reg)
        item = self._add_region_item(reg,
                                     len(self.all_regions[self.cur_page]) - 1)
        self.select_region(item)
        self._update_status()

    def _delete_selected(self):
        if not self._selected:
            return
        self.all_regions[self.cur_page].remove(self._selected.region)
        self._selected = None
        self._refresh_items()

    def _prev_page(self):
        if self.cur_page > 0:
            self._goto_page(self.cur_page - 1)

    def _next_page(self):
        if self.cur_page < self.num_pages - 1:
            self._goto_page(self.cur_page + 1)

    def _update_status(self):
        n   = len(self.all_regions[self.cur_page])
        sel = (f"selected [{self._selected.region.rtype}]"
               if self._selected else "none selected")
        self.status.showMessage(
            f"  Page {self.cur_page + 1}/{self.num_pages}  ·  "
            f"{n} region(s)  ·  {sel}   "
            "[← → pages | Del remove | Esc deselect | Ctrl+Wheel zoom | Hover=caption]"
        )

    def _save_all(self):
        out = Path(OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        manifest  = []
        img_count = 0

        for pg_num, regions in enumerate(self.all_regions):
            if not regions:
                continue
            pixmap = render_page(self.pdf_path, pg_num)
            pw, ph = pixmap.width(), pixmap.height()

            for idx, reg in enumerate(regions):
                r  = reg.scaled_rect()
                x0 = max(0,  int(r.x()))
                y0 = max(0,  int(r.y()))
                x1 = min(pw, int(r.right()))
                y1 = min(ph, int(r.bottom()))
                crop  = pixmap.copy(x0, y0, x1 - x0, y1 - y0)
                fname = (f"page{pg_num+1:03d}_reg{idx+1:03d}"
                         f"_{reg.rtype}_{reg.source}.png")
                crop.save(str(out / fname), "PNG")
                img_count += 1
                manifest.append({
                    "page":          pg_num + 1,
                    "region":        idx + 1,
                    "type":          reg.rtype,
                    "source":        reg.source,
                    "bbox_pdf_pts":  [round(c, 2) for c in reg.bbox],
                    "image_file":    fname,
                    "caption":       reg.caption,
                    "table_caption": reg.table_caption,
                })
                log.info("Saved %s", fname)

        json_path = out / "regions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        log.info("regions.json — %d entries", len(manifest))

        QMessageBox.information(
            self, "Export complete ✓",
            f"Saved {img_count} image(s) and regions.json\n\n📁 {out}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    pdf = Path(PDF_PATH)
    if not pdf.exists():
        log.error("PDF not found: %s", pdf)
        sys.exit(1)

    log.info("Detecting regions in %s …", pdf.name)
    try:
        all_regions = detect_regions(str(pdf))
    except Exception as exc:
        log.exception("Detection failed: %s", exc)
        sys.exit(1)

    total = sum(len(p) for p in all_regions)
    log.info("%d region(s) across %d page(s).", total, len(all_regions))

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RegionEditor(str(pdf), all_regions)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()