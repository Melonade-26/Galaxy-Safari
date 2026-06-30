"""
Galaxy Rating App
==================
PyQt5 aplikácia na manuálne hodnotenie galaxií z obrázkov.

- Obrázky sa nenačítavajú všetky naraz do pamäte - vždy je v pamäti
  iba aktuálne zobrazený obrázok (QPixmap sa vytvára/zahadzuje za chodu).
- Dáta (ra, dec, číselné parametre) sa berú z CSV súboru cez pandas
  (to je ľahké aj pre stovky tisíc riadkov, lebo sú to len čísla).
- Otázky sú checkboxy, dá sa zaškrtnúť viac naraz, plus poznámka (text).
- Tlačidlo "Uložiť" vpravo dole zapíše/aktualizuje riadok vo výstupnom
  textovom (CSV) súbore a posunie na ďalší obrázok.
- Poradie obrázkov: sekvenčné (podľa CSV) alebo náhodné (shuffle),
  prepínacie tlačidlá/menu vedľa tlačidla Uložiť.

DOPLŇ SI:
1) CONFIG sekciu nižšie (cesty, názvy stĺpcov, zoznam otázok).
2) Funkciu `scale_pixmap()` - vlož tam svoju vlastnú logiku škálovania,
   miesto je jasne označené komentárom "TU VLOŽ SVOJU FUNKCIU".
"""

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
from PyQt5.QtGui import QFont

# ============================== CONFIG ===================================
CSV_PATH = "GZ_all_data_new.csv"

# Priečinok, kde sú uložené obrázky
IMAGE_DIR = r"C:\Users\melo9\OneDrive\Desktop\Astronomy\GZ3\obrazky_podla_confidence"

# Názvy stĺpcov v CSV, ktoré obsahujú ra a dec
RA_COL = "ra_x"
DEC_COL = "dec_x"

# Ako sú pomenované súbory obrázkov - uprav podľa skutočného formátu.
# Príklad nižšie predpokladá napr. "123.456700_45.123400.jpg"
# Над {ra} a {dec} môžeš dať aj vlastné formátovanie (počet desatinných miest a pod.)
def image_filename(row):
    ra = row[RA_COL]
    dec = row[DEC_COL]
    return f"galaxy_{ra:.4f}_{dec:.4f}.jpg"

# Ktoré stĺpce z CSV zobrazovať pod obrázkom (číselné aj textové).
# Ak necháš None, použijú sa automaticky všetky číselné stĺpce okrem ra/dec
# (textové stĺpce sa v tom prípade NEzobrazia - preto ich radšej vymenuj ručne).
PARAM_COLUMNS = ["p_el","p_cw","p_acw","p_edge","p_dk","p_mg","p_cs","p_el_debiased","p_cs_debiased","petroMag_r",
                 "petroMagErr_r","petroR50_r","petroR90_r","modelMag_u","modelMag_g","modelMag_r","modelMag_i","modelMag_z","extinction_u","extinction_g","extinction_r",
                 "extinction_i","extinction_z","mRrCc_r","mRrCcErr_r","lnLStar_r","lnLExp_r","lnLDeV_r","P_EL_gz1","P_CW_gz1","P_ACW_gz1","P_EDGE_gz1","P_DK_gz1","P_MG_gz1",
                 "P_CS_gz1","P_EL_DEBIASED_gz1","P_CS_DEBIASED_gz1"]#None  # napr. ["P_EL", "P_CW", "P_ACW", "redshift", ...]

MAIN_PARAM_COLUMNS = [
    "gz2_class",
    "GZ1_type",
    "confidence",
    "class_from_conf",
    "spiral",
    "elliptical",
    "uncertain"
    #"nvote"
]

#"gz2_class","class_from_conf","confidence","GZ1_type", "nvote",

# Zoznam otázok - každá bude jeden checkbox, dá sa zaškrtnúť viac naraz
QUESTIONS = [
    "Priečka",
    "Grand design špirálne ramená",
    "Flocculent spiral arms",
    "One-armed spiral",
    "Delenie ramien podľa pitch angle",
    "Prstencové",
    "Nuclear ring",
    "Polar ring",
    "Tidal features",
    "Merger",
    "Warp",
    "Bulge",
    "Superthin disk",
    "Dust lane",
    "Extraplanar features",
    "Jellyfish",
    "Trpasličie galaxie",
    "Chybné"
]


# Kam sa ukladajú výsledky hodnotenia
OUTPUT_PATH = "ratings_output.csv"



# ============================ KONIEC CONFIG ===============================


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

def org_transform(wcs):
    return wcs

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
    wcs = linear_transform(wcs, f_range = (0, 1))
    #wcs /= 255
    wcs = np.true_divide(wcs, a)
    wcs = np.arcsinh(wcs)
    wcs = np.true_divide(wcs, np.arcsinh(1.0 / a))
    wcs = linear_transform(wcs)
    #wcs *= 255
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
    #"scale_1": org_transform, #linear_transform,
    "scale_1": linear_transform, #asinh_transform,
    "scale_2": square_two_transform,
    "scale_3": square_three_transform,
    #"scale_5": custom_transform,  # ak by si pridal aj piate tlačidlo
    "scale_4": asinh_transform
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

        # Načítaj už uložené hodnotenia (ak súbor existuje), aby sa dalo pokračovať
        # a aby sa už ohodnotené obrázky nezobrazovali znova.
        self.existing_results = self._load_existing_results()

        # Poradie obrázkov - na začiatku sekvenčné, ale len z tých, čo ešte nemajú výsledok
        self.order = self._build_order(shuffle=False)

        self._build_ui()
        self._load_current_image()

    # ------------------------------------------------------------ Helpers
    def _row_key(self, row):
        return (str(row[RA_COL]), str(row[DEC_COL]))

    def _build_order(self, shuffle: bool):
        """Vytvorí zoznam indexov do self.df, ktoré ešte NIE SÚ vo výstupnom
        súbore (t.j. ešte neboli ohodnotené). Voliteľne ich poprehadzuje."""
        indices = [
            i for i in range(len(self.df))
            if self._row_key(self.df.iloc[i]) not in self.existing_results
        ]
        if shuffle:
            random.shuffle(indices)
        return indices

    # ----------------------------------------------------------------- UI
    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # ---------- Ľavá strana: tlačidlá škálovania + obrázok + parametre ----------
        left_layout = QVBoxLayout()

        scale_buttons_layout = QHBoxLayout()
        self.scale_buttons = {}
        #nazvy = ["Pôvodné", "Lineárne", "Mocnina 2", "Mocnina 3", "Vlastné", "AsinH"]
        nazvy = ["Lineárne", "Mocnina 2", "Mocnina 3", "AsinH"]
        for i, mode in enumerate(["scale_1", "scale_2", "scale_3", "scale_4"], start=1):
            #btn = QPushButton(f"Mierka {i}")
            btn = QPushButton(str(nazvy[i - 1]))
            btn.clicked.connect(lambda checked, m=mode: self._apply_scale(m))
            scale_buttons_layout.addWidget(btn)
            self.scale_buttons[mode] = btn
        left_layout.addLayout(scale_buttons_layout)

        self.image_label = QLabel("Načítavam obrázok...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.installEventFilter(self)  # kvôli zoomu kolieskom myši
        left_layout.addWidget(self.image_label, stretch=1)

        # Info o aktuálnom obrázku (ra, dec, poradie)
        self.info_label = QLabel("")
        left_layout.addWidget(self.info_label)

        # ---- Parametre pod obrázkom: hlavné (výrazné) + ostatné ----
        params_box = QGroupBox("Parametre")
        params_box_layout = QVBoxLayout()

        self.main_params_label = QLabel("")
        self.main_params_label.setWordWrap(True)
        self.main_params_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.main_params_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; padding: 3px;"
        )
        params_box_layout.addWidget(self.main_params_label)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        params_box_layout.addWidget(separator)

        self.params_label = QLabel("")
        self.params_label.setWordWrap(True)
        self.params_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.params_label.setStyleSheet("font-size: 12px; padding: 2px;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll.setWidget(self.params_label)
        params_box_layout.addWidget(scroll, stretch=1)

        params_box.setLayout(params_box_layout)
        params_box.setFixedHeight(230) 
        left_layout.addWidget(params_box)#, stretch=1

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

        questions_box = QGroupBox("Otázky")
        questions_layout = QVBoxLayout()
        self.checkboxes = []
        for q in QUESTIONS:
            cb = QCheckBox(q)
            questions_layout.addWidget(cb)
            self.checkboxes.append(cb)
        questions_layout.addStretch()
        questions_box.setLayout(questions_layout)
        right_layout.addWidget(questions_box)

        note_box = QGroupBox("Poznámka")
        note_layout = QVBoxLayout()
        self.note_edit = QTextEdit()
        self.note_edit.setFixedHeight(120)
        note_layout.addWidget(self.note_edit)
        note_box.setLayout(note_layout)
        right_layout.addWidget(note_box)

        right_layout.addStretch()

        # Voľba poradia obrázkov
        order_box = QGroupBox("Poradie obrázkov")
        order_layout = QHBoxLayout()
        self.sequential_btn = QPushButton("Sekvenčné")
        self.sequential_btn.clicked.connect(self._set_sequential_order)
        self.random_btn = QPushButton("Náhodné")
        self.random_btn.clicked.connect(self._set_random_order)
        order_layout.addWidget(self.sequential_btn)
        order_layout.addWidget(self.random_btn)
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
                        key = (row.get(RA_COL), row.get(DEC_COL))
                        results[key] = row
            except Exception:
                pass
        return results

    def _load_current_image(self):
        if not self.order:
            self.image_label.setText("Všetky obrázky sú už ohodnotené.")
            self.info_label.setText("")
            self.main_params_label.setText("")
            self.params_label.setText("")
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

        self.info_label.setText(
            f"[{self.current_pos + 1}/{len(self.order)}]  "
            f"{RA_COL}={row[RA_COL]}  {DEC_COL}={row[DEC_COL]}  ({fname})"
        )

        # Hlavné (výrazné) parametre
        DISPLAY_NAMES = {
            "gz2_class": "GZ2",
            "class_from_conf": "Conf. class",
            "confidence": "Confidence",
            "GZ1_type": "GZ1",
            "nvote": "Votes"
        }
        
        html = "<table cellspacing='3' cellpadding='1'>"

        for i in range(0, len(self.main_param_columns), 2):

            html += "<tr>"

            for j in [i, i + 1]:
                if j < len(self.main_param_columns):
                    col = self.main_param_columns[j]
                    name = DISPLAY_NAMES.get(col, col)
                    value = row[col]

                    # Zaokrúhlenie čísel
                    if isinstance(value, float):
                        value = f"{value:.3f}"

                    html += f"""
                        <td><b>{name}:</b></td>
                        <td>{value}</td>
                    """

            html += "</tr>"

        html += "</table>"

        self.main_params_label.setText(html)

        # Ostatné parametre
        html = "<table cellspacing='2'>"

        for i in range(0, len(self.param_columns), 2):

            html += "<tr>"

            for j in [i, i+1]:
                if j < len(self.param_columns):
                    col = self.param_columns[j]
                    value = row[col]

                    if isinstance(value, float):
                        value = f"{value:.4f}"

                    html += (
                        f"<td><b>{col}</b></td>"
                        f"<td align='right'>{value}</td>"
                    )

            html += "</tr>"

        html += "</table>"

        self.params_label.setText(html)

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

    def _set_sequential_order(self):
        self.order = self._build_order(shuffle=False)
        self.current_pos = 0
        self._load_current_image()

    def _set_random_order(self):
        self.order = self._build_order(shuffle=True)
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
        self.existing_results[key] = {
            RA_COL: row[RA_COL],
            DEC_COL: row[DEC_COL],
            "questions": ";".join(checked),
            "note": note,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        self._write_output_file()

        # Tento obrázok je hotový - vyhoď ho z poradia, aby sa už neopakoval.
        # current_pos zostáva rovnaký - po odstránení na jeho mieste je ďalší obrázok.
        del self.order[self.current_pos]
        if self.current_pos >= len(self.order):
            self.current_pos = max(0, len(self.order) - 1)

        self._load_current_image()

    def _write_output_file(self):
        fieldnames = [RA_COL, DEC_COL, "questions", "note", "timestamp"]
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csvmod.DictWriter(f, fieldnames=fieldnames)
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
