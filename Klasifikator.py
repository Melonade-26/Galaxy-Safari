import sys
import os
import random
import csv as csvmod
from datetime import datetime

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QCheckBox, QTextEdit, QScrollArea, QMessageBox,
    QFileDialog, QSizePolicy, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QEvent
import warnings

warnings.filterwarnings("ignore")

CSV_PATH = "Galaxy_stats_compact.csv"
IMAGE_DIR = "obrazky"


RA_COL = "ra"
DEC_COL = "dec"
NAME = "dr7objid"

def image_filename(row):
    id = row[NAME]
    return f"galaxy_{id}.jpg"

MAIN_PARAM_COLUMNS = [
    "base_type",
    "min_confidence",
    "GZ2_type",
    "GZ1_type",
]


PARAM_COLUMNS = None
QUESTION_DISPLAY_NAMES = {
    "Bars":                     "Bars – priečne pásy",
    "Bulge":                    "Bulge – centrálne zhrubnutie",
    "Dust_lane":                "Dust lane – prašný pás",
    "Dwarf_companions":         "Dwarf companions – trpasličí spoločníci",
    "Extraplanar_features":     "Extraplanar features – mimoplošné štruktúry",
    "Flocculent_arms":          "Flocculent arms – chumáčovité ramená",
    "Grand_design_spiral_arms": "Grand design spiral arms – veľkolepé špirálovité ramená",
    "Jellyfish":                "Jellyfish – medúzovité galaxie",
    "Nuclear_ring":             "Nuclear ring – jadrový prstenec",
    "One_armed":                "One-armed – jednoramienkové",
    "Ongoing_merger":           "Ongoing merger – prebiehajúce splývanie",
    "Pitch_angle":              "Pitch angle – uhol špirálovitých ramien",
    "Polar_rings":              "Polar rings – polárne prstence",
    "Ringed":                   "Ringed – prstencové",
    "Superthin disk":           "Superthin disk – super-tenký disk",
    "Tidal_features":           "Tidal features – slapové štruktúry",
    "Warp":                     "Warp – prehnutý disk",
}

CLASS_COL = "base_type"
ELLIPTICAL_VALUES = {"E"}

QUESTIONS = [
    "Bars",
    "Bulge",
    "Dust_lane",
    "Dwarf_companions",
    "Extraplanar_features",
    "Flocculent_arms",
    "Grand_design_spiral_arms",
    "Jellyfish",
    "Nuclear_ring",
    "One_armed",
    "Ongoing_merger",
    "Pitch_angle",
    "Polar_rings",
    "Ringed",
    "Superthin disk",
    "Tidal_features",
    "Warp",
    "Neviem",
]

CATEGORY_IMAGES_DIR = "Obrazky_kategorii"
OUTPUT_PATH = "ratings_output.csv"


def qpixmap_to_numpy(pixmap: QPixmap) -> np.ndarray:
    """Skonvertuje QPixmap na numpy pole tvaru (H, W, 4) v RGBA poradí."""
    image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4)).copy()
    return arr


def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Skonvertuje numpy pole (H, W, 4) RGBA uint8 späť na QPixmap."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    height, width, channels = arr.shape
    if channels != 4:
        raise ValueError("Pole musí mať 4 kanály (RGBA). Uprav si výstup svojej funkcie.")
    image = QImage(arr.data, width, height, width * 4, QImage.Format_RGBA8888)
    # .copy() je dôležité - QImage si inak drží referenciu na `arr`,
    # ktoré môže Python medzičasom uvoľniť z pamäte
    return QPixmap.fromImage(image.copy())


# ---------------------------------------------------------------------
# VLASTNÉ TRANSFORMAČNÉ FUNKCIE (pracujú na jednom 2D kanáli naraz)
# ---------------------------------------------------------------------
def linear_transform(wcs_w4, f_range=(0, 255)):
    maxi = np.nanmax(wcs_w4)
    mini = np.nanmin(wcs_w4)
    wcs_w4 = np.nan_to_num(wcs_w4, nan=maxi)
    nieco = (f_range[1] - f_range[0]) / (maxi - mini)
    mins = f_range[0] - mini * nieco
    obr = wcs_w4 * nieco
    obr = obr + mins
    obr = np.clip(obr, f_range[0], f_range[1])
    return obr


def asinh_transform(wcs, a=0.1):
    wcs = np.true_divide(wcs, a)
    wcs = np.arcsinh(wcs)
    wcs = np.true_divide(wcs, np.arcsinh(1.0 / a))
    return wcs


def square_two_transform(wcs):
    wcs = np.power(wcs, 2)
    wcs = linear_transform(wcs)
    return wcs


def square_three_transform(wcs):
    wcs = np.power(wcs, 3)
    wcs = linear_transform(wcs)
    return wcs


def custom_transform(wcs):
    wcs = linear_transform(wcs)
    wcs = np.power(wcs, 2)
    wcs = linear_transform(wcs)
    return wcs


# Priraď, ktorá transformácia patrí ku ktorému z 4 tlačidiel.
# Pokojne si premenuj/popresúvaj podľa toho, čo chceš mať na ktorom tlačidle.
SCALE_MODE_FUNCTIONS = {
    "scale_1": linear_transform,
    "scale_2": square_two_transform,
    "scale_3": square_three_transform,
    "scale_4": asinh_transform,
    # "scale_5": custom_transform,  # ak by si pridal aj piate tlačidlo
}


def my_custom_scaling_function(arr: np.ndarray, mode: str) -> np.ndarray:
    """
    Aplikuje zvolenú transformáciu samostatne na R, G a B kanál
    (alfa kanál sa ponechá nezmenený, aby obrázok ostal nepriehľadný).

    `arr` je RGBA uint8 pole (H, W, 4).
    Vracia RGBA uint8 pole rovnakého tvaru.
    """
    transform_fn = SCALE_MODE_FUNCTIONS.get(mode, linear_transform)

    # Pracujeme vo float, aby transformácie (mocniny, asinh, ...) fungovali správne
    arr_float = arr.astype(np.float64)

    out = np.empty_like(arr_float)

    # R, G, B - každý kanál samostatne
    for ch in range(3):
        channel = arr_float[:, :, ch]
        transformed = transform_fn(channel)
        # nech je výstup akýkoľvek rozsah, vždy ho normalizujeme do 0-255
        # (linear_transform sa dá zavolať aj druhýkrát bez problému)
        transformed = linear_transform(transformed, f_range=(0, 255))
        out[:, :, ch] = transformed

    # Alfa kanál necháme tak, ako bol (plná nepriehľadnosť)
    out[:, :, 3] = arr_float[:, :, 3]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def scale_pixmap(pixmap: QPixmap, mode: str) -> QPixmap:
    """
    Prevedie QPixmap na numpy pole, zavolá tvoju vlastnú škálovaciu
    funkciu a výsledok prevedie späť na QPixmap.
    """
    if pixmap is None or pixmap.isNull():
        return pixmap

    arr = qpixmap_to_numpy(pixmap)
    new_arr = my_custom_scaling_function(arr, mode)
    return numpy_to_qpixmap(new_arr)


SUPPORTED_IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"}
HOVER_THUMB_SIZE = 220  # px - veľkosť náhľadu pri hover


def first_image_in_folder(category: str) -> str | None:
    """Vráti cestu k prvému obrázku v priečinku kategórie, alebo None."""
    folder = os.path.join(CATEGORY_IMAGES_DIR, category)
    if not os.path.isdir(folder):
        return None
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in SUPPORTED_IMG_EXT:
            return os.path.join(folder, f)
    return None


class HoverPreviewCheckBox(QCheckBox):
    """
    QCheckBox ktorý pri nabehnutí myšou zobrazí malé vyskakovacie okno
    s prvým obrázkom z priečinka danej kategórie.
    Ak priečinok neexistuje alebo je prázdny, správa sa ako normálny checkbox.
    """
    def __init__(self, label: str, category: str, parent=None):
        super().__init__(label, parent)
        self._category = category
        self._popup: QWidget | None = None
        self._thumb_path = first_image_in_folder(category)
        if self._thumb_path:
            self.setMouseTracking(True)

    def enterEvent(self, event):
        if not self._thumb_path:
            return
        pix = QPixmap(self._thumb_path)
        if pix.isNull():
            return
        pix = pix.scaled(HOVER_THUMB_SIZE, HOVER_THUMB_SIZE,
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)

        popup = QWidget(None, Qt.ToolTip | Qt.FramelessWindowHint)
        popup.setAttribute(Qt.WA_ShowWithoutActivating)
        layout = QVBoxLayout(popup)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        img_lbl = QLabel()
        img_lbl.setPixmap(pix)
        layout.addWidget(img_lbl)

        display_name = QUESTION_DISPLAY_NAMES.get(self._category, self._category)
        name_lbl = QLabel(display_name)
        name_lbl.setStyleSheet("font-size: 11px; color: #ccc; background: transparent;")
        name_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_lbl)

        popup.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        popup.adjustSize()

        # Umiestni popup napravo od checkboxu
        global_pos = self.mapToGlobal(self.rect().topRight())
        popup.move(global_pos.x() + 6, global_pos.y())
        popup.show()
        self._popup = popup

    def leaveEvent(self, event):
        if self._popup:
            self._popup.hide()
            self._popup.deleteLater()
            self._popup = None


class CategoryPreviewWindow(QWidget):
    """
    Samostatné okno zobrazujúce ukážkové obrázky jednej kategórie.
    Hľadá všetky obrázky v CATEGORY_IMAGES_DIR/{category}/ a zobrazí
    ich v mriežke (max IMAGE_COLS stĺpcov), každý s rozmerom THUMB_SIZE.
    Okno sa dá zatvoriť, znovu otvoriť - vždy jedno okno na kategóriu.
    """
    THUMB_SIZE = 200   # px - veľkosť strany miniatúry
    IMAGE_COLS = 3     # počet stĺpcov v mriežke
    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"}

    def __init__(self, category: str, parent=None):
        super().__init__(parent, Qt.Window)
        self.category = category
        display_name = QUESTION_DISPLAY_NAMES.get(category, category)
        self.setWindowTitle(f"Ukážky: {display_name}")
        self.resize(700, 600)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)

        display_name = QUESTION_DISPLAY_NAMES.get(self.category, self.category)
        title = QLabel(f"<b>{display_name}</b>")
        title.setStyleSheet("font-size: 18px; padding: 6px;")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(8)
        scroll.setWidget(container)

        folder = os.path.join(CATEGORY_IMAGES_DIR, self.category)
        if not os.path.isdir(folder):
            no_img = QLabel(f"Priečinok nenájdený:\n{folder}")
            no_img.setAlignment(Qt.AlignCenter)
            grid.addWidget(no_img, 0, 0)
            return

        files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in self.SUPPORTED_EXT
        )

        if not files:
            no_img = QLabel("Žiadne obrázky v priečinku.")
            no_img.setAlignment(Qt.AlignCenter)
            grid.addWidget(no_img, 0, 0)
            return

        for idx, fname in enumerate(files):
            path = os.path.join(folder, fname)
            row, col = divmod(idx, self.IMAGE_COLS)

            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)

            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            pix = QPixmap(path)
            if not pix.isNull():
                pix = pix.scaled(
                    self.THUMB_SIZE, self.THUMB_SIZE,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                img_label.setPixmap(pix)
            else:
                img_label.setText("(chyba)")

            name_label = QLabel(fname)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setWordWrap(True)
            name_label.setStyleSheet("font-size: 11px; color: #888;")

            cell_layout.addWidget(img_label)
            cell_layout.addWidget(name_label)
            grid.addWidget(cell, row, col)


class GalaxyRatingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galaxy Rating App")
        self.resize(1200, 750)

        # --- Načítanie dát ---
        self.df = pd.read_csv(CSV_PATH)
        self.main_param_columns = MAIN_PARAM_COLUMNS

        if PARAM_COLUMNS is None:
            numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
            self.param_columns = [
                c for c in numeric_cols
                if c not in (RA_COL, DEC_COL) and c not in self.main_param_columns
            ]
        else:
            self.param_columns = [c for c in PARAM_COLUMNS if c not in self.main_param_columns]

        self.current_pos = 0  # pozícia v self.order
        self.current_pixmap_raw = None  # originálny QPixmap aktuálneho obrázku (po custom škálovaní)
        self.zoom_factor = 1.0  # zoom navyše nad "fit do okna"

        self.exclude_elliptical = False

        # Načítaj už uložené hodnotenia
        self.existing_results = self._load_existing_results()

        # Poradie obrázkov - vždy round-robin cez base_type, náhodné v rámci tried
        self.order = self._build_order()

        self._category_windows: dict[str, CategoryPreviewWindow] = {}

        self._build_ui()
        self._load_current_image()

    # ------------------------------------------------------------ Helpers
    def _row_key(self, row):
        return str(row[NAME])

    def _build_order(self):
        """
        Zostaví poradie indexov do self.df pre neohodnotené galaxie.
        Algoritmus: round-robin cez triedy v stĺpci base_type.
          - Každá trieda dostane vlastný náhodne pomiešaný zoznam.
          - Ber po jednom z každej triedy (cyklicky), kým nie sú všetky prázdne.
        Výsledok: striedajú sa rôzne typy galaxií, nikdy nevidíš 50× rovnaký typ.
        Filter: ak self.exclude_elliptical == True, vynechajú sa riadky
                kde base_type == "E".
        """
        from collections import defaultdict

        has_class = CLASS_COL in self.df.columns

        def keep(i):
            if self._row_key(self.df.iloc[i]) in self.existing_results:
                return False
            if self.exclude_elliptical and has_class:
                if str(self.df.iloc[i][CLASS_COL]).strip() in ELLIPTICAL_VALUES:
                    return False
            return True

        candidates = [i for i in range(len(self.df)) if keep(i)]

        if not has_class:
            # base_type stĺpec neexistuje - padback na náhodné poradie
            random.shuffle(candidates)
            return candidates

        # Rozdeľ podľa base_type, každú skupinu pomiešaj
        groups = defaultdict(list)
        for i in candidates:
            cls = str(self.df.iloc[i][CLASS_COL]).strip()
            groups[cls].append(i)
        """for cls in groups:
            random.shuffle(groups[cls])"""
        
        CONFIDENCE_COL = "min_confidence"  # názov stĺpca v tvojom CSV
        for cls in groups:
            if CONFIDENCE_COL in self.df.columns:
                groups[cls].sort(key=lambda i: self.df.iloc[i][CONFIDENCE_COL])
                # .pop() berie z konca → najvyšší confidence príde prvý
            else:
                random.shuffle(groups[cls])

        # Round-robin: ber po jednom z každej triedy
        keys = list(groups.keys())
        random.shuffle(keys)  # poradie tried tiež náhodné
        result = []
        while any(groups[k] for k in keys):
            for k in keys:
                if groups[k]:
                    result.append(groups[k].pop())
        return result

    # ----------------------------------------------------------------- UI
    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # ---------- Ľavá strana: tlačidlá škálovania + obrázok + parametre ----------
        left_layout = QVBoxLayout()
        list_nazov = ["Linear", "Square 2", "Square 3", "AsinH"]
        scale_buttons_layout = QHBoxLayout()
        self.scale_buttons = {}
        for i, mode in enumerate(["scale_1", "scale_2", "scale_3", "scale_4"], start=1):
            btn = QPushButton(list_nazov[i - 1])
            btn.clicked.connect(lambda checked, m=mode: self._apply_scale(m))
            scale_buttons_layout.addWidget(btn)
            self.scale_buttons[mode] = btn
        left_layout.addLayout(scale_buttons_layout)

        self.image_label = QLabel("Načítavam obrázok...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.installEventFilter(self)  # kvôli zoomu kolieskom myši
        left_layout.addWidget(self.image_label, stretch=3)

        # Info o aktuálnom obrázku (ra, dec, poradie)
        self.info_label = QLabel("")
        left_layout.addWidget(self.info_label)

        # ---- Parametre pod obrázkom: len hlavné (výrazné) ----
        params_box = QGroupBox("Parametre")
        params_box_layout = QVBoxLayout()

        self.main_params_label = QLabel("")
        self.main_params_label.setWordWrap(True)
        self.main_params_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.main_params_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 6px;"
        )
        params_box_layout.addWidget(self.main_params_label)
        params_box.setLayout(params_box_layout)
        left_layout.addWidget(params_box)

        # Navigácia Prev/Next
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("<< Predošlý")
        self.prev_btn.clicked.connect(self._go_prev)
        self.next_btn = QPushButton("Ďalší >>")
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        left_layout.addLayout(nav_layout)

        main_layout.addLayout(left_layout, stretch=3)

        # ---------- Pravá strana: otázky (checkboxy) + poznámka + uloženie ----------
        right_layout = QVBoxLayout()

        questions_box = QGroupBox("Kategórie")
        questions_scroll = QScrollArea()
        questions_scroll.setWidgetResizable(True)
        questions_container = QWidget()
        questions_layout = QVBoxLayout(questions_container)
        questions_layout.setSpacing(2)
        self.checkboxes = []

        for q in QUESTIONS:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(2, 1, 2, 1)

            # "Neviem" dostane oddeľovač a nemá folder
            if q == "Neviem":
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                questions_layout.addWidget(sep)
                cb = QCheckBox(q)
            else:
                cb = HoverPreviewCheckBox(q, q)

            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            row_layout.addWidget(cb)
            self.checkboxes.append(cb)

            # Tlačidlo 👁 len pre kategórie s existujúcim priečinkom
            cat_folder = os.path.join(CATEGORY_IMAGES_DIR, q)
            if os.path.isdir(cat_folder):
                preview_btn = QPushButton("👁")
                preview_btn.setFixedSize(28, 28)
                preview_btn.setToolTip(f"Ukáž príklady: {QUESTION_DISPLAY_NAMES.get(q, q)}")
                preview_btn.clicked.connect(
                    lambda checked, cat=q: self._open_category_preview(cat)
                )
                row_layout.addWidget(preview_btn)

            questions_layout.addWidget(row_widget)

        questions_layout.addStretch()
        questions_scroll.setWidget(questions_container)

        questions_box_layout = QVBoxLayout()
        questions_box_layout.addWidget(questions_scroll)
        questions_box.setLayout(questions_box_layout)
        right_layout.addWidget(questions_box, stretch=1)

        note_box = QGroupBox("Poznámka")
        note_layout = QVBoxLayout()
        self.note_edit = QTextEdit()
        self.note_edit.setFixedHeight(120)
        note_layout.addWidget(self.note_edit)
        note_box.setLayout(note_layout)
        right_layout.addWidget(note_box)

        right_layout.addStretch()

        # Filter eliptických + info o poradí
        order_box = QGroupBox("Nastavenia poradia")
        order_layout = QVBoxLayout()
        self.excl_elliptical_cb = QCheckBox("Vynechať eliptické galaxie (base_type = E)")
        self.excl_elliptical_cb.setChecked(False)
        self.excl_elliptical_cb.stateChanged.connect(self._on_elliptical_filter_changed)
        order_layout.addWidget(self.excl_elliptical_cb)
        order_info = QLabel("Poradie: náhodný round-robin cez typy (base_type)")
        order_info.setStyleSheet("font-size: 11px; color: gray;")
        order_layout.addWidget(order_info)
        order_box.setLayout(order_layout)
        right_layout.addWidget(order_box)

        # Tlačidlo Uložiť (vpravo dole)
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self.save_btn = QPushButton("Uložiť")
        self.save_btn.setFixedSize(140, 45)
        self.save_btn.clicked.connect(self._save_current)
        save_layout.addWidget(self.save_btn)
        right_layout.addLayout(save_layout)

        main_layout.addLayout(right_layout, stretch=1)

    # ------------------------------------------------------------ Loading
    def _current_row(self):
        idx = self.order[self.current_pos]
        return self.df.iloc[idx]

    def _load_existing_results(self):
        """Načíta už uložené výsledky z OUTPUT_PATH, aby sa dali predvyplniť."""
        results = {}
        if os.path.exists(OUTPUT_PATH):
            try:
                with open(OUTPUT_PATH, "r", newline="", encoding="utf-8") as f:
                    reader = csvmod.DictReader(f)
                    for row in reader:
                        key = row.get(NAME)
                        if key is not None:
                            results[str(key)] = row
            except Exception:
                pass
        return results

    def _load_current_image(self):
        if not self.order:
            self.image_label.setText("Všetky obrázky sú už ohodnotené.")
            self.info_label.setText("")
            self.main_params_label.setText("")
            return

        self.zoom_factor = 1.0  # pri novom obrázku zoom resetni

        row = self._current_row()
        fname = image_filename(row)
        path = os.path.join(IMAGE_DIR, fname)

        if os.path.exists(path):
            pix = QPixmap(path)
        else:
            pix = QPixmap()  # prázdny -> ukáže sa chyba

        self.original_pixmap = pix  # úplne pôvodný, nezmenený obrázok
        if pix.isNull():
            self.current_pixmap_raw = None
            self.image_label.setText(f"Obrázok sa nenašiel:\n{path}")
        else:
            self._apply_scale("scale_1")  # predvolená mierka pri načítaní (aj zobrazí)

        ra_str  = f"  {RA_COL}={row[RA_COL]}"  if RA_COL  in row.index else ""
        dec_str = f"  {DEC_COL}={row[DEC_COL]}" if DEC_COL in row.index else ""
        self.info_label.setText(
            f"[{self.current_pos + 1}/{len(self.order)}]  "
            f"{NAME}={row[NAME]}{ra_str}{dec_str}  ({fname})"
        )

        # Hlavné (výrazné) parametre
        main_lines = [f"{col}: {row[col]}" for col in self.main_param_columns if col in row.index]
        self.main_params_label.setText(
            "\n".join(main_lines) if main_lines else "(žiadne hlavné parametre)"
        )

        # Predvyplnenie checkboxov a poznámky, ak už bol tento obrázok hodnotený
        key = self._row_key(row)
        existing = self.existing_results.get(key)
        for cb in self.checkboxes:
            cb.setChecked(False)
        self.note_edit.clear()
        if existing:
            checked_questions = set(
                q.strip() for q in existing.get("questions", "").split(";") if q.strip()
            )
            for cb in self.checkboxes:
                cb.setChecked(cb.text() in checked_questions)
            self.note_edit.setPlainText(existing.get("note", ""))

    def _apply_scale(self, mode):
        if getattr(self, "original_pixmap", None) is None or self.original_pixmap.isNull():
            return
        # aplikuje tvoju vlastnú transformačnú funkciu na pôvodný obrázok
        self.current_pixmap_raw = scale_pixmap(self.original_pixmap, mode)
        self._refresh_display()

    def _refresh_display(self):
        """Prispôsobí (zmenší/zväčší) aktuálny obrázok tak, aby vyplnil
        celú dostupnú plochu image_label-u, s prihliadnutím na zoom_factor."""
        if self.current_pixmap_raw is None or self.current_pixmap_raw.isNull():
            return

        label_size = self.image_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return

        target_w = max(1, int(label_size.width() * self.zoom_factor))
        target_h = max(1, int(label_size.height() * self.zoom_factor))

        scaled = self.current_pixmap_raw.scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_display()

    def eventFilter(self, source, event):
        # Zoom kolieskom myši, keď je kurzor nad obrázkom
        if source is self.image_label and event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            step = 0.1
            if delta > 0:
                self.zoom_factor = min(self.zoom_factor + step, 10.0)
            else:
                self.zoom_factor = max(self.zoom_factor - step, 0.1)
            self._refresh_display()
            return True
        return super().eventFilter(source, event)

    def _open_category_preview(self, category: str):
        """Otvorí (alebo prenesie do popredia) okno s ukážkami danej kategórie."""
        win = self._category_windows.get(category)
        if win is None or not win.isVisible():
            win = CategoryPreviewWindow(category, parent=None)
            self._category_windows[category] = win
        win.show()
        win.raise_()
        win.activateWindow()

    # ----------------------------------------------------------- Naviga.
    def _go_next(self):
        if self.current_pos < len(self.order) - 1:
            self.current_pos += 1
            self._load_current_image()
        else:
            QMessageBox.information(self, "Koniec", "Toto je posledný obrázok.")

    def _go_prev(self):
        if self.current_pos > 0:
            self.current_pos -= 1
            self._load_current_image()
        else:
            QMessageBox.information(self, "Začiatok", "Toto je prvý obrázok.")

    def _on_elliptical_filter_changed(self):
        self.exclude_elliptical = self.excl_elliptical_cb.isChecked()
        self.order = self._build_order()
        self.current_pos = 0
        self._load_current_image()

    # -------------------------------------------------------------- Save
    def _save_current(self):
        if not self.order:
            return

        row = self._current_row()
        checked = [cb.text() for cb in self.checkboxes if cb.isChecked()]
        note = self.note_edit.toPlainText().strip()

        key = self._row_key(row)
        record = {NAME: row[NAME], "questions": ";".join(checked), "note": note,
                  "timestamp": datetime.now().isoformat(timespec="seconds")}
        if RA_COL in row.index:
            record[RA_COL] = row[RA_COL]
        if DEC_COL in row.index:
            record[DEC_COL] = row[DEC_COL]
        self.existing_results[key] = record

        self._write_output_file()

        del self.order[self.current_pos]
        if self.current_pos >= len(self.order):
            self.current_pos = max(0, len(self.order) - 1)

        self._load_current_image()

    def _write_output_file(self):
        # Základné stĺpce vždy + ra/dec len ak existujú v dátach
        base = [NAME]
        for col in (RA_COL, DEC_COL):
            if col in self.df.columns:
                base.append(col)
        fieldnames = base + ["questions", "note", "timestamp"]
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csvmod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for key, rec in self.existing_results.items():
                writer.writerow(rec)


def main():
    app = QApplication(sys.argv)
    window = GalaxyRatingApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
