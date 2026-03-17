from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class PlateStorage:
    def __init__(self, db_path: str = "plates.db") -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plate_reads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                image_path TEXT NOT NULL,
                plate_text TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                detector_source TEXT NOT NULL,
                detector_score REAL NOT NULL,
                ocr_confidence REAL NOT NULL,
                bbox_json TEXT NOT NULL,
                vis_path TEXT,
                crop_path TEXT,
                plate_image_path TEXT
            )
            """
        )
        columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(plate_reads)").fetchall()
        }
        if "vis_path" not in columns:
            self.conn.execute("ALTER TABLE plate_reads ADD COLUMN vis_path TEXT")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_plate_reads_created_at ON plate_reads(created_at)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_plate_reads_plate_text ON plate_reads(plate_text)"
        )
        self.conn.commit()

    def insert_read(
        self,
        image_path: str,
        plate_text: str,
        raw_text: str,
        detector_source: str,
        detector_score: float,
        ocr_confidence: float,
        bbox: tuple[int, int, int, int],
        vis_path: str | None = None,
        crop_path: str | None = None,
        plate_image_path: str | None = None,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO plate_reads (
                created_at,
                image_path,
                plate_text,
                raw_text,
                detector_source,
                detector_score,
                ocr_confidence,
                bbox_json,
                vis_path,
                crop_path,
                plate_image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                image_path,
                plate_text,
                raw_text,
                detector_source,
                float(detector_score),
                float(ocr_confidence),
                json.dumps({"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}),
                vis_path,
                crop_path,
                plate_image_path,
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def recent_reads(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT
                id,
                created_at,
                image_path,
                plate_text,
                raw_text,
                detector_source,
                detector_score,
                ocr_confidence,
                bbox_json,
                vis_path,
                crop_path,
                plate_image_path
            FROM plate_reads
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_stats(self) -> dict[str, int]:
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_reads,
                COUNT(DISTINCT plate_text) AS unique_plates
            FROM plate_reads
            """
        ).fetchone()
        return {
            "total_reads": int(row["total_reads"] or 0),
            "unique_plates": int(row["unique_plates"] or 0),
        }

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        bbox = json.loads(row["bbox_json"])
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "image_path": row["image_path"],
            "plate_text": row["plate_text"],
            "raw_text": row["raw_text"],
            "detector_source": row["detector_source"],
            "detector_score": float(row["detector_score"]),
            "ocr_confidence": float(row["ocr_confidence"]),
            "bbox": bbox,
            "vis_path": row["vis_path"],
            "crop_path": row["crop_path"],
            "plate_image_path": row["plate_image_path"],
        }
