from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "PySide6 is not installed. Install with: .\\.venv\\Scripts\\python.exe -m pip install PySide6"
    ) from exc

from full_plate_pipeline import (
    draw_bbox,
    load_ocr_model,
    YoloPlateDetector,
)
from demo_my_images import choose_best_ocr_for_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Desktop ANPR app (demo pipeline).")
    parser.add_argument("--ocr-weights", default="ocr_checkpoints/ocr_best.pt")
    parser.add_argument("--yolo-weights", default="runs/detect/runs_yolo/plate_detector_ru2/weights/best.pt")
    parser.add_argument("--csv-log", default="app_data/qt_logs/plate_log.csv")
    return parser.parse_args()


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "image_path",
                "plate",
                "raw",
                "detector_source",
                "detector_score",
                "ocr_confidence",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
            ]
        )


def append_csv_row(csv_path: Path, row: list[object]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def cv_to_qpixmap(image: np.ndarray) -> QPixmap:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    q_img = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_img.copy())


class ImageBox(QLabel):
    def __init__(self, title: str, min_h: int = 200) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self.setText(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(min_h)
        self.setStyleSheet(
            "QLabel{border:1px solid #2f3647;border-radius:10px;background:#151925;color:#7f8aa3;}"
        )

    def set_cv_image(self, image: np.ndarray) -> None:
        self._pixmap = cv_to_qpixmap(image)
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.csv_path = Path(args.csv_log)
        ensure_csv_header(self.csv_path)

        self.current_image_path: str | None = None
        self.current_image: np.ndarray | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_detector: YoloPlateDetector | None = None
        self.ocr_model = None
        self.converter = None

        self.setWindowTitle("ANPR Desktop - Dark")
        self.resize(1480, 920)
        self._setup_ui()
        self._apply_dark_theme()
        self._load_models()

    def _setup_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)\

             
        layout.setSpacing(16)

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        self.btn_open = QPushButton("Open Image")
        self.btn_open.clicked.connect(self.open_image)  # type: ignore[arg-type]

        self.btn_run = QPushButton("Recognize")
        self.btn_run.clicked.connect(self.run_recognition)  # type: ignore[arg-type]

        self.path_label = QLabel("File: -")
        self.path_label.setWordWrap(True)
        self.path_label.setObjectName("pathLabel")

        top_buttons = QHBoxLayout()
        top_buttons.addWidget(self.btn_open)
        top_buttons.addWidget(self.btn_run)

        self.main_image_box = ImageBox("Input Image + BBox", min_h=620)

        left_col.addLayout(top_buttons)
        left_col.addWidget(self.path_label)
        left_col.addWidget(self.main_image_box, stretch=1)

        self.crop_box = ImageBox("Cropped BBox", min_h=240)
        self.plate_box = ImageBox("OCR plate view", min_h=180)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(180)

        self.log_table = QTableWidget(0, 4)
        self.log_table.setHorizontalHeaderLabels(["Date/Time", "Plate", "Detector", "OCR conf"])
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.setMinimumHeight(220)

        right_col.addWidget(self.crop_box)
        right_col.addWidget(self.plate_box)
        right_col.addWidget(self.info_box)
        right_col.addWidget(self.log_table, stretch=1)

        layout.addLayout(left_col, stretch=7)
        layout.addLayout(right_col, stretch=4)
        self.statusBar().showMessage("Ready")

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #0f131d; color: #e8ecf3; }
            QWidget { color: #e8ecf3; font-size: 14px; }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #1f6aa5, stop:1 #1b7f84);
                color: white; border: none; border-radius: 10px; padding: 10px 14px; font-weight: 600;
            }
            QPushButton:hover { background: #238ab0; }
            QPushButton:pressed { background: #196985; }
            QTextEdit, QTableWidget {
                background: #151925; border: 1px solid #2f3647; border-radius: 10px; color: #e8ecf3;
            }
            QHeaderView::section {
                background: #1b2130; color: #b8c2d9; border: 0; padding: 6px; font-weight: 600;
            }
            QLabel#pathLabel { color: #a8b3ca; }
            QStatusBar { background: #121726; color: #9fb6d1; }
            """
        )

    def _load_models(self) -> None:
        try:
            self.statusBar().showMessage(f"Loading models on {self.device}...")
            yolo_path = Path(self.args.yolo_weights)
            if not yolo_path.exists():
                raise FileNotFoundError(f"YOLO weights not found: {yolo_path}")
            self.yolo_detector = YoloPlateDetector(str(yolo_path), device=str(self.device))
            self.ocr_model, self.converter = load_ocr_model(self.args.ocr_weights, self.device)
            self.statusBar().showMessage(f"Models loaded. Device: {self.device}, YOLO: {yolo_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Model loading error", str(exc))
            raise

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not file_path:
            return
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Error", "Could not read image file.")
            return
        self.current_image_path = file_path
        self.current_image = image
        self.path_label.setText(f"File: {file_path}")
        self.main_image_box.set_cv_image(image)
        self.info_box.setPlainText("Image loaded. Press 'Recognize'.")
        self.statusBar().showMessage("Image loaded")

    def run_recognition(self) -> None:
        if self.current_image is None or self.current_image_path is None:
            QMessageBox.information(self, "No image", "Select a file first.")
            return
        if self.yolo_detector is None or self.ocr_model is None or self.converter is None:
            QMessageBox.critical(self, "Error", "Models are not initialized.")
            return

        self.btn_run.setEnabled(False)
        self.statusBar().showMessage("Running recognition...")
        QApplication.processEvents()

        try:
            image = self.current_image.copy()
            detection = self.yolo_detector.detect(image)
            if detection is None:
                raise RuntimeError("No YOLO detection on this image.")
            best_ocr = choose_best_ocr_for_detection(
                image=image,
                bbox=detection.bbox,
                ocr_model=self.ocr_model,
                converter=self.converter,
                device=self.device,
            )
            if best_ocr is None:
                raise RuntimeError("Empty crop after OCR variants.")
            ocr_result, crop, plate_view, vis_bbox = best_ocr

            plate = ocr_result.text or "-"
            raw = ocr_result.raw_text or "-"
            bbox = vis_bbox
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            vis = image.copy()
            draw_bbox(
                vis,
                bbox,
                f"{plate if plate != '-' else raw} | {detection.source} | {detection.score:.2f}",
            )

            self.main_image_box.set_cv_image(vis)
            self.crop_box.set_cv_image(crop)
            self.plate_box.set_cv_image(plate_view)

            details = [
                f"date_time: {timestamp}",
                f"image: {self.current_image_path}",
                f"plate: {plate}",
                f"raw: {raw}",
                f"variant: {ocr_result.variant}",
                f"detector: {detection.source}",
                f"detector_score: {detection.score:.3f}",
                f"ocr_confidence: {ocr_result.confidence:.3f}",
                f"bbox: {bbox}",
                f"csv_log: {self.csv_path}",
            ]
            self.info_box.setPlainText("\n".join(details))

            append_csv_row(
                self.csv_path,
                [
                    timestamp,
                    self.current_image_path,
                    plate,
                    raw,
                    detection.source,
                    f"{detection.score:.6f}",
                    f"{ocr_result.confidence:.6f}",
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                ],
            )
            self._append_table_row(timestamp, plate, detection.source, ocr_result.confidence)
            self.statusBar().showMessage(f"Done: {plate}")
        except Exception as exc:
            QMessageBox.critical(self, "Recognition error", str(exc))
            self.statusBar().showMessage("Recognition error")
        finally:
            self.btn_run.setEnabled(True)

    def _append_table_row(self, ts: str, plate: str, detector: str, ocr_conf: float) -> None:
        row = self.log_table.rowCount()
        self.log_table.insertRow(row)
        self.log_table.setItem(row, 0, QTableWidgetItem(ts))
        self.log_table.setItem(row, 1, QTableWidgetItem(plate))
        self.log_table.setItem(row, 2, QTableWidgetItem(detector))
        self.log_table.setItem(row, 3, QTableWidgetItem(f"{ocr_conf:.3f}"))
        if self.log_table.rowCount() > 200:
            self.log_table.removeRow(0)


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
