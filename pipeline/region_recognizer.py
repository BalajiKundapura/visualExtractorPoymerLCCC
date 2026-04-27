"""
PDF Figure + Table Extractor
Figures  — raster image blocks embedded in the PDF
Tables   — detected via camelot 
Qt UI    — review, add, resize, delete, re-categorise regions
Export   — figures/ and tables/ sub-folders + regions.json
"""

from __future__ import annotations
import json, logging, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf as fitz
from pdf2image import convert_from_path
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui  import QBrush, QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSizePolicy, QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

# ── Paths (edit these) ────────────────────────────────────────────────────────
PDF_PATH   = "/Users/balaji/Desktop/Polymer LCC Project /figure-extraction-pipeline/Inputs/polymerPaper1.pdf"
OUTPUT_DIR = "/Users/balaji/Desktop/Polymer LCC Project /figure-extraction-pipeline/extracted_regions"

# ── Constants ─────────────────────────────────────────────────────────────────
DPI        = 150
SCALE      = DPI / 72.0
MIN_AREA   = 2_500          # pt² — ignore tiny blobs
TYPES      = ["figure", "table"]
PALETTE    = {"figure": QColor("#00e676"), "table": QColor("#ffd740"), "manual": QColor("#ff6d00")}


# ── Data ──────────────────────────────────────────────────────────────────────
@dataclass
class Region:
    rtype:  str
    bbox:   list[float]     # PDF points [x0,y0,x1,y1]
    source: str = "auto"

    @property
    def area(self):
        x0,y0,x1,y1 = self.bbox
        return max(0,(x1-x0)*(y1-y0))

    def qrect(self) -> QRectF:
        x0,y0,x1,y1 = self.bbox
        return QRectF(x0*SCALE, y0*SCALE, (x1-x0)*SCALE, (y1-y0)*SCALE)

    def color(self) -> QColor:
        return PALETTE.get(self.rtype, PALETTE["manual"])


# ── Detection ─────────────────────────────────────────────────────────────────
def detect(pdf_path: str) -> list[list[Region]]:
    doc, pages = fitz.open(pdf_path), []
    for pno, page in enumerate(doc):
        regions = []

        # Figures — embedded raster image blocks
        for blk in page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)["blocks"]:
            if blk.get("type") != 1:
                continue
            bb = list(blk["bbox"])
            if (bb[2]-bb[0])*(bb[3]-bb[1]) >= MIN_AREA:
                regions.append(Region("figure", bb))

        # Tables — PyMuPDF's built-in finder
        try:
            for tbl in page.find_tables().tables:
                bb = list(tbl.bbox)
                if (bb[2]-bb[0])*(bb[3]-bb[1]) >= MIN_AREA:
                    regions.append(Region("table", bb))
        except Exception as e:
            log.warning("Page %d find_tables error: %s", pno+1, e)

        log.info("Page %d: %d figure(s), %d table(s)",
                 pno+1,
                 sum(1 for r in regions if r.rtype=="figure"),
                 sum(1 for r in regions if r.rtype=="table"))
        pages.append(regions)
    doc.close()
    return pages


# ── Render page to QPixmap ────────────────────────────────────────────────────
def render(pdf_path: str, page_num: int) -> QPixmap:
    pil = convert_from_path(pdf_path, dpi=DPI, first_page=page_num+1, last_page=page_num+1)[0].convert("RGB")
    w,h = pil.size
    return QPixmap.fromImage(QImage(pil.tobytes("raw","RGB"), w, h, w*3, QImage.Format.Format_RGB888))


# ── Region graphics item ──────────────────────────────────────────────────────
class RItem(QGraphicsRectItem):
    HS = 9  # handle size px

    def __init__(self, region: Region, idx: int):
        super().__init__(region.qrect())
        self.region, self.idx = region, idx
        self._sel = False
        self._drag_handle = None
        self._drag_origin_rect = None
        self._drag_origin_pos  = None
        self.setAcceptHoverEvents(True)
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self._style()

    def _style(self):
        c = self.region.color()
        pen = QPen(c, 3 if self._sel else 2,
                   Qt.PenStyle.SolidLine if self._sel else Qt.PenStyle.DashLine)
        fill = QColor(c); fill.setAlpha(50 if self._sel else 20)
        self.setPen(pen); self.setBrush(QBrush(fill))

    def set_sel(self, v: bool):
        self._sel = v; self._style(); self.update()

    def _handles(self) -> dict[str, QRectF]:
        r, s = self.rect(), self.HS
        return {
            "tl": QRectF(r.left()-s/2,  r.top()-s/2,    s, s),
            "tr": QRectF(r.right()-s/2, r.top()-s/2,    s, s),
            "bl": QRectF(r.left()-s/2,  r.bottom()-s/2, s, s),
            "br": QRectF(r.right()-s/2, r.bottom()-s/2, s, s),
        }

    def paint(self, p: QPainter, opt, w=None):
        super().paint(p, opt, w)
        r, c = self.rect(), self.region.color()
        lbl = f"[{self.idx+1}] {self.region.rtype}"
        fm  = p.fontMetrics()
        badge = QRectF(r.x(), r.y(), fm.horizontalAdvance(lbl)+10, fm.height()+4)
        p.fillRect(badge, c)
        p.setPen(QPen(QColor("black")))
        p.setFont(QFont("Helvetica", 8, QFont.Weight.Bold))
        p.drawText(badge.adjusted(5,2,-5,-2), Qt.AlignmentFlag.AlignLeft, lbl)
        if self._sel:
            p.setPen(QPen(QColor("white"), 1))
            p.setBrush(QBrush(c))
            for hr in self._handles().values():
                p.drawRect(hr)

    def mousePressEvent(self, ev):
        if self._sel:
            for name, hr in self._handles().items():
                if hr.contains(ev.pos()):
                    self._drag_handle      = name
                    self._drag_origin_rect = QRectF(self.rect())
                    self._drag_origin_pos  = ev.scenePos()
                    ev.accept(); return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_handle:
            d, r = ev.scenePos()-self._drag_origin_pos, QRectF(self._drag_origin_rect)
            if "l" in self._drag_handle: r.setLeft(r.left()+d.x())
            if "r" in self._drag_handle: r.setRight(r.right()+d.x())
            if "t" in self._drag_handle: r.setTop(r.top()+d.y())
            if "b" in self._drag_handle: r.setBottom(r.bottom()+d.y())
            self.setRect(r.normalized()); self._sync(); ev.accept(); return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag_handle:
            self._drag_handle = None; ev.accept(); return
        super().mouseReleaseEvent(ev)

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            self._sync()
        return super().itemChange(change, value)

    def _sync(self):
        sr = self.mapToScene(self.rect()).boundingRect()
        self.region.bbox = [sr.x()/SCALE, sr.y()/SCALE, sr.right()/SCALE, sr.bottom()/SCALE]

    def hoverMoveEvent(self, ev):
        in_h = self._sel and any(hr.contains(ev.pos()) for hr in self._handles().values())
        self.setCursor(Qt.CursorShape.SizeFDiagCursor if in_h else Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.unsetCursor(); super().hoverLeaveEvent(ev)


# ── Type picker dialog ────────────────────────────────────────────────────────
class TypeDialog(QDialog):
    def __init__(self, parent=None, current="figure"):
        super().__init__(parent)
        self.setWindowTitle("Region type")
        self.setFixedSize(240, 110)
        self.setStyleSheet("QDialog{background:#2a2a2a;color:#e0e0e0}"
                           "QComboBox{background:#3c3c3c;color:white;padding:4px;border-radius:4px}"
                           "QPushButton{background:#007acc;color:white;padding:5px 16px;border-radius:4px;font-weight:bold}")
        lay = QVBoxLayout(self)
        self.combo = QComboBox()
        self.combo.addItems(TYPES + ["manual"])
        self.combo.setCurrentText(current)
        lay.addWidget(QLabel("Type:"))
        lay.addWidget(self.combo)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    @property
    def value(self): return self.combo.currentText()


# ── Scene ─────────────────────────────────────────────────────────────────────
class Scene(QGraphicsScene):
    def __init__(self, editor: "Editor"):
        super().__init__()
        self._ed = editor
        self._drawing = False
        self._origin  = QPointF()
        self._rubber  = None

    def mousePressEvent(self, ev):
        for item in self.items(ev.scenePos()):
            if isinstance(item, RItem):
                self._ed.select(item); return
        self._ed.deselect()
        self._drawing = True
        self._origin  = ev.scenePos()
        self._rubber  = self.addRect(QRectF(self._origin, self._origin),
                                     QPen(PALETTE["manual"], 2, Qt.PenStyle.DashLine))

    def mouseMoveEvent(self, ev):
        if self._drawing and self._rubber:
            self._rubber.setRect(QRectF(self._origin, ev.scenePos()).normalized())

    def mouseReleaseEvent(self, ev):
        if not self._drawing: return
        self._drawing = False
        if self._rubber: self.removeItem(self._rubber); self._rubber = None
        rect = QRectF(self._origin, ev.scenePos()).normalized()
        dlg  = TypeDialog(self._ed.centralWidget())
        if dlg.exec() != QDialog.DialogCode.Accepted: return
        reg = Region(dlg.value,
                     [rect.x()/SCALE, rect.y()/SCALE,
                      rect.right()/SCALE, rect.bottom()/SCALE],
                     source="manual")
        if reg.area >= MIN_AREA:
            self._ed.add_region(reg)


# ── Main window ───────────────────────────────────────────────────────────────
class Editor(QMainWindow):
    def __init__(self, pdf_path: str, all_regions: list[list[Region]]):
        super().__init__()
        self.pdf_path    = pdf_path
        self.all_regions = all_regions
        self.cur         = 0
        self._sel: Optional[RItem] = None
        self._items: list[RItem]   = []
        self.setWindowTitle("PDF Extractor")
        self.resize(1300, 900)
        self._ui()
        self._goto(0)

    def _ui(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#1e1e1e;color:#e0e0e0}
            QPushButton{background:#3c3c3c;color:white;border:none;padding:6px 14px;border-radius:4px;font-weight:bold}
            QPushButton:hover{background:#505050}
            QStatusBar{background:#007acc;color:white;font-size:11px}
            QGraphicsView{border:none;background:#111}
            QToolBar{background:#252526;border:none;spacing:4px;padding:4px}
            QToolTip{background:#2d2d2d;color:#e0e0e0;border:1px solid #555;padding:6px}
        """)
        tb = QToolBar(); tb.setMovable(False); self.addToolBar(tb)

        def btn(label, slot, style=""):
            b = QPushButton(label); b.clicked.connect(slot)
            if style: b.setStyleSheet(style)
            tb.addWidget(b); return b

        btn("◀ Prev", self._prev)
        self.pg_lbl = QLabel("1/1")
        self.pg_lbl.setMinimumWidth(90)
        self.pg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tb.addWidget(self.pg_lbl)
        btn("Next ▶", self._next)
        tb.addSeparator()
        btn("🏷 Retype", self._retype)
        sp = QWidget(); sp.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(sp)
        btn("🗑 Delete", self._delete, "background:#8b1a1a;")
        btn("💾 Export", self._export, "background:#1a6b3a;")

        # Legend
        leg = QWidget(); leg.setStyleSheet("background:#1e1e1e")
        lh  = QHBoxLayout(leg); lh.setContentsMargins(8,2,8,2); lh.setSpacing(12)
        for name, col in PALETTE.items():
            sw = QLabel(); sw.setFixedSize(12,12)
            sw.setStyleSheet(f"background:{col.name()};border-radius:2px")
            lh.addWidget(sw); lh.addWidget(QLabel(name))
        lh.addStretch()

        self.scene = Scene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.wheelEvent = lambda ev: self.view.scale(*(2*(1.15 if ev.angleDelta().y()>0 else 1/1.15,)))

        c = QWidget(); cl = QVBoxLayout(c); cl.setContentsMargins(0,0,0,0); cl.setSpacing(0)
        cl.addWidget(leg); cl.addWidget(self.view)
        self.setCentralWidget(c)
        self.status = QStatusBar(); self.setStatusBar(self.status)

        for key, fn in [(Qt.Key.Key_Left, self._prev), (Qt.Key.Key_Right, self._next),
                        (Qt.Key.Key_Delete, self._delete), (Qt.Key.Key_Backspace, self._delete),
                        (Qt.Key.Key_Escape, self.deselect)]:
            QShortcut(QKeySequence(key), self, fn)

    def _goto(self, n: int):
        self.cur = max(0, min(n, len(self.all_regions)-1))
        self._sel = None; self._items = []; self.scene.clear()
        px = render(self.pdf_path, self.cur)
        pi = QGraphicsPixmapItem(px); pi.setZValue(0); self.scene.addItem(pi)
        self.scene.setSceneRect(pi.boundingRect())
        for i, r in enumerate(self.all_regions[self.cur]): self._add(r, i)
        self.pg_lbl.setText(f"{self.cur+1}/{len(self.all_regions)}")
        self._stat()
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _add(self, reg: Region, idx: int) -> RItem:
        it = RItem(reg, idx); it.setZValue(1); self.scene.addItem(it); self._items.append(it); return it

    def _refresh(self):
        for it in self._items: self.scene.removeItem(it)
        self._items = []
        for i, r in enumerate(self.all_regions[self.cur]): self._add(r, i)
        self._stat()

    def select(self, item: RItem):
        if self._sel and self._sel is not item: self._sel.set_sel(False)
        self._sel = item; item.set_sel(True); self._stat()

    def deselect(self):
        if self._sel: self._sel.set_sel(False); self._sel = None
        self._stat()

    def add_region(self, reg: Region):
        self.all_regions[self.cur].append(reg)
        self.select(self._add(reg, len(self.all_regions[self.cur])-1))

    def _delete(self):
        if not self._sel: return
        self.all_regions[self.cur].remove(self._sel.region)
        self._sel = None; self._refresh()

    def _retype(self):
        if not self._sel:
            QMessageBox.information(self, "No selection", "Click a region first."); return
        dlg = TypeDialog(self.centralWidget(), self._sel.region.rtype)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._sel.region.rtype = dlg.value; self._refresh()

    def _prev(self):
        if self.cur > 0: self._goto(self.cur-1)

    def _next(self):
        if self.cur < len(self.all_regions)-1: self._goto(self.cur+1)

    def _stat(self):
        n = len(self.all_regions[self.cur])
        sel = f"selected [{self._sel.region.rtype}]" if self._sel else "none"
        self.status.showMessage(
            f"  Page {self.cur+1}/{len(self.all_regions)}  ·  {n} region(s)  ·  {sel}"
            "   [← → pages | Del/⌫ delete | Esc deselect | drag=new box | corner-drag=resize]")

    def _export(self):
        out = Path(OUTPUT_DIR)
        dirs = {t: (out/t) for t in TYPES + ["manual"]}
        for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

        manifest, count = [], 0
        for pg, regions in enumerate(self.all_regions):
            if not regions: continue
            px = render(self.pdf_path, pg)
            pw, ph = px.width(), px.height()
            for idx, reg in enumerate(regions):
                r  = reg.qrect()
                x0,y0 = max(0,int(r.x())), max(0,int(r.y()))
                x1,y1 = min(pw,int(r.right())), min(ph,int(r.bottom()))
                folder = dirs.get(reg.rtype, out/"other")
                folder.mkdir(exist_ok=True)
                fname  = f"page{pg+1:03d}_{idx+1:03d}_{reg.source}.png"
                px.copy(x0,y0,x1-x0,y1-y0).save(str(folder/fname), "PNG")
                manifest.append({"page": pg+1, "index": idx+1, "type": reg.rtype,
                                  "source": reg.source, "file": f"{reg.rtype}/{fname}",
                                  "bbox_pt": [round(v,2) for v in reg.bbox]})
                count += 1

        (out/"regions.json").write_text(json.dumps(manifest, indent=2))
        summary = "\n".join(f"  {t}/  {sum(1 for m in manifest if m['type']==t)}" for t in TYPES if any(m['type']==t for m in manifest))
        QMessageBox.information(self, "Done ✓", f"{count} image(s) exported:\n\n{summary}\n\n📁 {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf = Path(PDF_PATH)
    if not pdf.exists():
        print(f"PDF not found: {pdf}"); sys.exit(1)

    log.info("Scanning %s …", pdf.name)
    all_regions = detect(str(pdf))
    log.info("Total: %d region(s)", sum(len(p) for p in all_regions))

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    Editor(str(pdf), all_regions).show()
    sys.exit(app.exec())