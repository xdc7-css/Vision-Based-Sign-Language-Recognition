"""Volunteer Camera Recording Dialog for Independent Human Evaluation.

Provides:
- Selection and registration of human volunteers with anonymous default IDs.
- Selection of 10 Arabic digits (0001-0010) and idle resting baseline.
- Real-time display of reference sign instructions and hand pose guidance.
- Selection of webcam hardware (Laptop camera or iVCam).
- Live camera viewport with start/stop recording controls.
- Recorded video playback and review with Accept / Retry / Discard confirmation.
- Safe local MP4 saving, non-duplicate repetition indexing, and volunteer progress tracking.
- Volunteer data privacy controls with instant local data deletion.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.research.independent_evaluation import IndependentEvaluationManager

logger = logging.getLogger("arsl.presentation.volunteer_recording")


class VolunteerRecordingDialog(QDialog):
    """Recording studio dialog for independent volunteer evaluations."""

    recording_saved = Signal(dict)

    def __init__(
        self,
        manager: Optional[IndependentEvaluationManager] = None,
        initial_volunteer_id: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Independent Volunteer Recognition Recording Studio")
        self.resize(860, 780)
        self.setMinimumSize(780, 700)

        self.manager = manager or IndependentEvaluationManager()
        self.active_volunteer_id = initial_volunteer_id

        # Video capture state
        self._cap: Optional[cv2.VideoCapture] = None
        self._camera_timer = QTimer(self)
        self._camera_timer.timeout.connect(self._on_camera_frame)

        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._on_playback_frame)

        self._is_recording = False
        self._recorded_frames: list[np.ndarray] = []
        self._recording_start_time = 0.0
        self._playback_idx = 0
        self._mode = "LIVE"  # "LIVE", "RECORDING", "REVIEW"

        self._setup_style()
        self._build_ui()
        self._populate_volunteers()
        self._populate_signs()
        self._start_camera()

    def _setup_style(self) -> None:
        self.setStyleSheet("""
            QDialog {
                background-color: #101419;
                color: #F3F4F6;
            }
            QLabel {
                color: #F3F4F6;
            }
            QComboBox {
                background-color: #1B2330;
                border: 1px solid #2B3442;
                border-radius: 6px;
                padding: 6px 12px;
                color: #F3F4F6;
                font-size: 12px;
            }
            QComboBox:focus {
                border-color: #8B5CF6;
            }
            QPushButton {
                background-color: #1B2330;
                border: 1px solid #2B3442;
                border-radius: 6px;
                padding: 8px 14px;
                color: #F3F4F6;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #242E3E;
            }
            QPushButton#actionButton {
                background-color: #8B5CF6;
                border: 1px solid #7C3AED;
                color: #FFFFFF;
            }
            QPushButton#actionButton:hover {
                background-color: #7C3AED;
            }
            QPushButton#successButton {
                background-color: #10B981;
                border: 1px solid #059669;
                color: #FFFFFF;
            }
            QPushButton#successButton:hover {
                background-color: #059669;
            }
            QPushButton#dangerButton {
                background-color: #EF4444;
                border: 1px solid #DC2626;
                color: #FFFFFF;
            }
            QPushButton#dangerButton:hover {
                background-color: #DC2626;
            }
            QProgressBar {
                background-color: #1B2330;
                border: 1px solid #2B3442;
                border-radius: 4px;
                text-align: center;
                color: #F3F4F6;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #8B5CF6;
                border-radius: 3px;
            }
        """)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ── Header Banner ──────────────────────────────────────────────────
        hdr_frame = QFrame()
        hdr_frame.setStyleSheet("background-color: #151B24; border: 1px solid #2B3442; border-radius: 8px; padding: 10px;")
        hdr_layout = QHBoxLayout(hdr_frame)
        hdr_layout.setContentsMargins(10, 8, 10, 8)

        title_box = QVBoxLayout()
        t_lbl = QLabel("📹  Volunteer Recognition Recording Studio")
        t_lbl.setStyleSheet("font-size: 15px; font-weight: 700; color: #F3F4F6;")
        title_box.addWidget(t_lbl)

        sub_lbl = QLabel("Independent Human Recognition Testing • 10 Arabic Digits (5 repetitions each) • Strict Privacy")
        sub_lbl.setStyleSheet("font-size: 11px; color: #94A3B8;")
        title_box.addWidget(sub_lbl)
        hdr_layout.addLayout(title_box)

        hdr_layout.addStretch()

        self._consent_btn = QPushButton("ℹ️ Consent & Privacy Notice")
        self._consent_btn.clicked.connect(self._show_consent_notice)
        hdr_layout.addWidget(self._consent_btn)
        layout.addWidget(hdr_frame)

        # ── Controls Bar: Volunteer, Sign, Camera ──────────────────────────
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("background-color: #151B24; border: 1px solid #2B3442; border-radius: 8px; padding: 10px;")
        ctrl_layout = QHBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(10, 8, 10, 8)
        ctrl_layout.setSpacing(10)

        # Volunteer selector
        ctrl_layout.addWidget(QLabel("Volunteer:"))
        self._vol_combo = QComboBox()
        self._vol_combo.setMinimumWidth(180)
        self._vol_combo.currentIndexChanged.connect(self._on_volunteer_changed)
        ctrl_layout.addWidget(self._vol_combo)

        self._add_vol_btn = QPushButton("➕ New Volunteer")
        self._add_vol_btn.clicked.connect(self._on_register_volunteer)
        ctrl_layout.addWidget(self._add_vol_btn)

        # Sign selector
        ctrl_layout.addWidget(QLabel("Target Sign:"))
        self._sign_combo = QComboBox()
        self._sign_combo.setMinimumWidth(160)
        self._sign_combo.currentIndexChanged.connect(self._on_sign_changed)
        ctrl_layout.addWidget(self._sign_combo)

        # Camera selector
        ctrl_layout.addWidget(QLabel("Camera:"))
        self._camera_combo = QComboBox()
        self._camera_combo.addItems([
            "Laptop Camera (Index 0)",
            "iVCam / External (Index 1)",
            "External Camera (Index 2)",
        ])
        self._camera_combo.currentIndexChanged.connect(self._on_camera_device_changed)
        ctrl_layout.addWidget(self._camera_combo)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl_frame)

        # ── Sign Reference & Pose Guide Card ───────────────────────────────
        self._guide_frame = QFrame()
        self._guide_frame.setStyleSheet(
            "background-color: #1E1B4B; border: 1px solid #4338CA; border-radius: 8px; padding: 10px;"
        )
        guide_layout = QHBoxLayout(self._guide_frame)
        guide_layout.setContentsMargins(12, 8, 12, 8)

        self._sign_name_lbl = QLabel("إشارة: صفر (Zero)")
        self._sign_name_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #A5B4FC; font-family: Tahoma, Arial;")
        guide_layout.addWidget(self._sign_name_lbl)

        self._sign_tip_lbl = QLabel("تعليمات الحركة: ضم أصابع اليد وتشكيل قبضة مغلقة.")
        self._sign_tip_lbl.setStyleSheet("font-size: 12px; color: #CBD5E1;")
        self._sign_tip_lbl.setWordWrap(True)
        guide_layout.addWidget(self._sign_tip_lbl, stretch=1)

        self._rep_badge = QLabel("Repetition 1 / 5")
        self._rep_badge.setStyleSheet(
            "background-color: #4338CA; color: #FFFFFF; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
        )
        guide_layout.addWidget(self._rep_badge)
        layout.addWidget(self._guide_frame)

        # ── Live Camera Viewport / Review Area ─────────────────────────────
        self._viewport_frame = QFrame()
        self._viewport_frame.setStyleSheet(
            "background-color: #0A0D12; border: 1px solid #2B3442; border-radius: 8px; min-height: 380px;"
        )
        viewport_layout = QVBoxLayout(self._viewport_frame)
        viewport_layout.setContentsMargins(4, 4, 4, 4)

        self._preview_lbl = QLabel("Initializing camera preview...")
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._preview_lbl.setStyleSheet("color: #64748B; font-size: 13px;")
        viewport_layout.addWidget(self._preview_lbl, stretch=1)

        layout.addWidget(self._viewport_frame, stretch=1)

        # ── Action Buttons Bar ─────────────────────────────────────────────
        btn_frame = QFrame()
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(10)

        # Normal mode buttons
        self._record_btn = QPushButton("🔴  Start Recording")
        self._record_btn.setObjectName("dangerButton")
        self._record_btn.setMinimumHeight(42)
        self._record_btn.clicked.connect(self._toggle_recording)
        btn_layout.addWidget(self._record_btn)

        # Review mode buttons (hidden until recording stops)
        self._accept_btn = QPushButton("✅  Accept & Save Recording")
        self._accept_btn.setObjectName("successButton")
        self._accept_btn.setMinimumHeight(42)
        self._accept_btn.setVisible(False)
        self._accept_btn.clicked.connect(self._accept_recording)
        btn_layout.addWidget(self._accept_btn)

        self._retry_btn = QPushButton("🔄  Retry (Record Again)")
        self._retry_btn.setMinimumHeight(42)
        self._retry_btn.setVisible(False)
        self._retry_btn.clicked.connect(self._retry_recording)
        btn_layout.addWidget(self._retry_btn)

        self._discard_btn = QPushButton("❌  Discard")
        self._discard_btn.setMinimumHeight(42)
        self._discard_btn.setVisible(False)
        self._discard_btn.clicked.connect(self._discard_recording)
        btn_layout.addWidget(self._discard_btn)

        btn_layout.addStretch()

        self._status_text_lbl = QLabel("Camera Active. Press 'Start Recording' when volunteer is ready.")
        self._status_text_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        btn_layout.addWidget(self._status_text_lbl)

        layout.addWidget(btn_frame)

        # ── Bottom Volunteer Progress & Privacy Strip ──────────────────────
        btm_frame = QFrame()
        btm_frame.setStyleSheet("background-color: #151B24; border: 1px solid #2B3442; border-radius: 8px; padding: 10px;")
        btm_layout = QHBoxLayout(btm_frame)
        btm_layout.setContentsMargins(10, 6, 10, 6)
        btm_layout.setSpacing(12)

        self._prog_label = QLabel("Volunteer Progress:")
        btm_layout.addWidget(self._prog_label)

        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 50)
        self._prog_bar.setValue(0)
        self._prog_bar.setFormat("%v / 50 signs (%p%)")
        self._prog_bar.setMinimumWidth(220)
        btm_layout.addWidget(self._prog_bar, stretch=1)

        self._delete_vol_btn = QPushButton("🗑️ Delete Volunteer Data")
        self._delete_vol_btn.setStyleSheet("color: #EF4444; border-color: #7F1D1D;")
        self._delete_vol_btn.clicked.connect(self._on_delete_volunteer)
        btm_layout.addWidget(self._delete_vol_btn)

        layout.addWidget(btm_frame)

    # ─── Camera Management ─────────────────────────────────────────────────

    def _start_camera(self) -> None:
        cam_idx = self._camera_combo.currentIndex()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._cap = cv2.VideoCapture(cam_idx)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(0)

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._camera_timer.start(33)  # ~30 FPS
            self._mode = "LIVE"
            self._update_ui_state()
        else:
            self._preview_lbl.setText("❌ Could not connect to selected camera hardware.")

    def _stop_camera(self) -> None:
        self._camera_timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _on_camera_device_changed(self) -> None:
        self._stop_camera()
        self._start_camera()

    def _on_camera_frame(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return

        # Mirror display for natural signing orientation
        frame = cv2.flip(frame, 1)

        if self._is_recording:
            self._recorded_frames.append(frame.copy())
            elapsed = time.perf_counter() - self._recording_start_time
            # Draw prominent recording indicator
            cv2.circle(frame, (30, 30), 12, (0, 0, 235), -1)
            cv2.putText(
                frame,
                f"REC  {elapsed:.1f}s ({len(self._recorded_frames)} frames)",
                (55, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        self._display_frame(frame)

    def _display_frame(self, bgr_frame: np.ndarray) -> None:
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self._preview_lbl.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._preview_lbl.setPixmap(scaled)

    # ─── Recording & Playback Review ───────────────────────────────────────

    def _toggle_recording(self) -> None:
        if not self._is_recording:
            # Start recording
            vol_id = self._vol_combo.currentData()
            if not vol_id:
                QMessageBox.warning(self, "No Volunteer", "Please register or select a volunteer before recording.")
                return

            self._recorded_frames = []
            self._is_recording = True
            self._recording_start_time = time.perf_counter()
            self._record_btn.setText("⏹  Stop Recording")
            self._status_text_lbl.setText("Recording in progress... Perform the sign clearly in frame.")
        else:
            # Stop recording and enter review
            self._is_recording = False
            self._record_btn.setText("🔴  Start Recording")

            if len(self._recorded_frames) < 10:
                QMessageBox.warning(self, "Clip Too Short", "Recorded clip was under 10 frames. Please record again.")
                self._recorded_frames = []
                self._status_text_lbl.setText("Recording discarded (too short). Ready to record.")
                return

            self._mode = "REVIEW"
            self._camera_timer.stop()
            self._playback_idx = 0
            self._playback_timer.start(33)
            self._update_ui_state()

    def _on_playback_frame(self) -> None:
        if not self._recorded_frames:
            return
        frame = self._recorded_frames[self._playback_idx].copy()
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"REVIEW PLAYBACK: Frame {self._playback_idx + 1} / {len(self._recorded_frames)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        self._display_frame(frame)
        self._playback_idx = (self._playback_idx + 1) % len(self._recorded_frames)

    def _accept_recording(self) -> None:
        """Confirm and save accepted recording to disk, updating progress and manifest."""
        self._playback_timer.stop()
        vol_id = self._vol_combo.currentData()
        sign_id = self._sign_combo.currentData()
        is_idle = (sign_id == "idle")

        next_rep = self.manager.get_next_repetition_index(vol_id, sign_id) if not is_idle else 1

        try:
            record_entry = self.manager.save_recording(
                volunteer_id=vol_id,
                sign_id=sign_id,
                rep_idx=next_rep,
                frames=self._recorded_frames,
                fps=30.0,
                is_idle=is_idle,
            )
            self.recording_saved.emit(record_entry)
            QMessageBox.information(
                self,
                "Recording Saved",
                f"Accepted recording saved successfully!\n"
                f"Sign: {record_entry['name_ar']}\n"
                f"Repetition: {record_entry['rep_idx']}\n"
                f"Frames: {record_entry['frame_count']}\n"
                f"File: {record_entry['file_path']}"
            )
            self._recorded_frames = []
            self._update_volunteer_progress_ui()
            self._on_sign_changed()
            self._resume_live_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error Saving", f"Failed to save recording: {e}")
            self._resume_live_preview()

    def _retry_recording(self) -> None:
        self._playback_timer.stop()
        self._recorded_frames = []
        self._status_text_lbl.setText("Clip discarded. Reopening live camera for a new take.")
        self._resume_live_preview()

    def _discard_recording(self) -> None:
        self._playback_timer.stop()
        self._recorded_frames = []
        self._status_text_lbl.setText("Recording discarded.")
        self._resume_live_preview()

    def _resume_live_preview(self) -> None:
        self._mode = "LIVE"
        self._update_ui_state()
        self._camera_timer.start(33)

    def _update_ui_state(self) -> None:
        is_review = (self._mode == "REVIEW")
        self._record_btn.setVisible(not is_review)
        self._accept_btn.setVisible(is_review)
        self._retry_btn.setVisible(is_review)
        self._discard_btn.setVisible(is_review)

        if is_review:
            self._status_text_lbl.setText(
                f"Reviewing recorded take ({len(self._recorded_frames)} frames). Accept, retry, or discard."
            )

    # ─── Data & Selection Handlers ─────────────────────────────────────────

    def _populate_volunteers(self) -> None:
        vols = self.manager.list_volunteers()
        self._vol_combo.clear()
        if vols:
            for v in vols:
                self._vol_combo.addItem(f"{v['name']} ({v.get('total_recordings', 0)}/50)", v["volunteer_id"])
            if self.active_volunteer_id:
                idx = self._vol_combo.findData(self.active_volunteer_id)
                if idx >= 0:
                    self._vol_combo.setCurrentIndex(idx)
        else:
            self._vol_combo.addItem("No volunteers registered", "")

        self._update_volunteer_progress_ui()

    def _populate_signs(self) -> None:
        self._sign_combo.clear()
        for c in self.manager.classes:
            self._sign_combo.addItem(f"{c['sign_id']}: {c['name_ar']} ({c['name_en']})", c["sign_id"])
        self._sign_combo.addItem("idle: استراحة / خمول (Resting Baseline)", "idle")
        self._on_sign_changed()

    def _on_volunteer_changed(self) -> None:
        self.active_volunteer_id = self._vol_combo.currentData()
        self._update_volunteer_progress_ui()
        self._on_sign_changed()

    def _on_sign_changed(self) -> None:
        sign_id = self._sign_combo.currentData()
        if not sign_id:
            return

        vol_id = self._vol_combo.currentData()
        info = self.manager.get_sign_reference_info(sign_id)
        if not info:
            return

        self._sign_name_lbl.setText(f"إشارة: {info['name_ar']} ({info['name_en']})")
        self._sign_tip_lbl.setText(f"تعليمات الحركة: {info['tip_ar']}")

        if vol_id and sign_id != "idle":
            prog = self.manager.get_volunteer_progress(vol_id)
            sign_prog = prog["progress_by_sign"].get(sign_id, {})
            reps_done = sign_prog.get("reps_completed", 0)
            next_rep = sign_prog.get("next_rep_idx", reps_done + 1)
            self._rep_badge.setText(f"Next Repetition: {min(next_rep, 5)} / 5 (Completed: {reps_done}/5)")
            if reps_done >= 5:
                self._rep_badge.setStyleSheet(
                    "background-color: #064E3B; color: #10B981; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
                )
            else:
                self._rep_badge.setStyleSheet(
                    "background-color: #4338CA; color: #FFFFFF; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
                )
        elif sign_id == "idle":
            self._rep_badge.setText("Idle Baseline Take")
            self._rep_badge.setStyleSheet(
                "background-color: #374151; color: #9CA3AF; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
            )

    def _update_volunteer_progress_ui(self) -> None:
        vol_id = self._vol_combo.currentData()
        if not vol_id:
            self._prog_bar.setValue(0)
            self._prog_label.setText("Volunteer Progress: 0 / 50")
            return

        prog = self.manager.get_volunteer_progress(vol_id)
        total_rec = prog["total_recorded"]
        self._prog_bar.setValue(min(total_rec, 50))
        self._prog_label.setText(f"Volunteer Progress: {total_rec} / 50 signs")

    def _on_register_volunteer(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Register New Volunteer",
            "Enter volunteer name or alias (leave blank for anonymous ID):",
        )
        if not ok:
            return
        notes, _ = QInputDialog.getText(
            self,
            "Camera / Setup Notes",
            "Enter environment / camera setup notes (optional):",
        )
        try:
            entry = self.manager.register_volunteer(name=name.strip(), notes=notes.strip())
            self.active_volunteer_id = entry["volunteer_id"]
            self._populate_volunteers()
            QMessageBox.information(
                self,
                "Volunteer Registered",
                f"Volunteer registered successfully:\nID: {entry['volunteer_id']}\nName: {entry['name']}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Registration Error", f"Failed to register volunteer: {e}")

    def _on_delete_volunteer(self) -> None:
        vol_id = self._vol_combo.currentData()
        if not vol_id:
            return

        reply = QMessageBox.warning(
            self,
            "Confirm Volunteer Deletion",
            f"Are you sure you want to permanently delete volunteer data for:\n{vol_id}?\n\n"
            "This will delete all local MP4 videos and manifest records for this volunteer.\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.manager.delete_volunteer(vol_id)
            self.active_volunteer_id = None
            self._populate_volunteers()
            QMessageBox.information(self, "Deleted", f"Volunteer data for {vol_id} has been permanently deleted.")

    def _show_consent_notice(self) -> None:
        notice = self.manager.get_consent_notice()
        QMessageBox.information(
            self,
            notice["title_ar"],
            f"{notice['title_ar']} / {notice['title_en']}\n\n"
            f"{notice['body_ar']}\n\n"
            f"English:\n{notice['body_en']}",
        )

    # ─── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event: Any) -> None:
        self._stop_camera()
        self._playback_timer.stop()
        super().closeEvent(event)

    def reject(self) -> None:
        self._stop_camera()
        self._playback_timer.stop()
        super().reject()

