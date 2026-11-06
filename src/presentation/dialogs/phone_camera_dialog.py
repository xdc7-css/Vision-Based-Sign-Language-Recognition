from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class PhoneCameraGuideDialog(QDialog):
    """Step-by-step setup guide for using smartphones as cameras (Wi-Fi & USB)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Smartphone Camera Setup Guide")
        self.setMinimumSize(560, 480)
        self.resize(600, 520)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("📱 How to Use Your Smartphone as a Camera")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #4ec9b0;")
        layout.addWidget(title)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 8px 16px;
                font-weight: 600;
            }
        """)

        # Tab 1: Wi-Fi Stream
        wifi_widget = QWidget()
        wifi_layout = QVBoxLayout(wifi_widget)
        wifi_browser = QTextBrowser()
        wifi_browser.setOpenExternalLinks(True)
        wifi_browser.setHtml("""
            <h3 style="color:#007acc;">Method A: Wi-Fi / IP Camera Stream</h3>
            <p>Stream live video from your phone over your local Wi-Fi network without cables.</p>

            <h4 style="color:#e0e0e0;">Recommended Mobile Apps:</h4>
            <ul>
                <li><b>Android:</b> <i>IP Webcam</i> (by Pavel Khlebovich) or <i>DroidCam</i></li>
                <li><b>iPhone (iOS):</b> <i>DroidCam Webcam & OBS</i>, <i>IP Camera Lite</i>, or <i>EpocCam</i></li>
            </ul>

            <h4 style="color:#e0e0e0;">Step-by-Step Instructions:</h4>
            <ol>
                <li><b>Same Network:</b> Connect your smartphone and laptop to the <b>same Wi-Fi network</b> or hotspot.</li>
                <li><b>Launch App:</b> Open the camera streaming app on your smartphone.</li>
                <li><b>Start Server:</b> Tap <i>Start Server</i> (or equivalent) in the app.</li>
                <li><b>Find URL:</b> Note the stream URL displayed on your phone's screen. Common patterns:
                    <ul>
                        <li><code>http://192.168.1.X:8080/video</code> (IP Webcam)</li>
                        <li><code>http://192.168.1.X:4747/video</code> (DroidCam)</li>
                        <li><code>http://192.168.1.X:8080/mjpegfeed</code> (Generic MJPEG)</li>
                    </ul>
                </li>
                <li><b>Connect:</b> Select <b>Phone Camera (Wi-Fi)</b> in this desktop app, enter the URL, and click <b>Connect</b>.</li>
            </ol>
            <p style="color:#9d9d9d; font-size:11px;">Tip: For best frame rates, select 640×480 or 1280×720 resolution in your phone app settings.</p>
        """)
        wifi_layout.addWidget(wifi_browser)
        tabs.addTab(wifi_widget, "📶 Wi-Fi Stream (Method A)")

        # Tab 2: USB / Virtual Webcam
        usb_widget = QWidget()
        usb_layout = QVBoxLayout(usb_widget)
        usb_browser = QTextBrowser()
        usb_browser.setOpenExternalLinks(True)
        usb_browser.setHtml("""
            <h3 style="color:#007acc;">Method B: USB / Virtual Webcam Driver</h3>
            <p>Use your smartphone as a standard plug-and-play USB webcam with ultra-low latency.</p>

            <h4 style="color:#e0e0e0;">Popular Solutions:</h4>
            <ul>
                <li><b>Iriun 4K Webcam:</b> Free for Android and iOS, high quality, zero-config USB driver.</li>
                <li><b>DroidCam:</b> Reliable USB & Wi-Fi virtual webcam for Android and iPhone.</li>
                <li><b>e2eSoft iVCam:</b> High-performance Windows virtual webcam driver for iOS/Android.</li>
                <li><b>Camo:</b> High-fidelity webcam app for iPhone and Android.</li>
            </ul>

            <h4 style="color:#e0e0e0;">Step-by-Step Instructions:</h4>
            <ol>
                <li><b>Install Companion:</b> Install the Windows PC client (e.g. Iriun Webcam or DroidCam PC Client).</li>
                <li><b>Install Mobile App:</b> Install the matching app on your Android phone or iPhone.</li>
                <li><b>Connect Cable:</b> Connect your phone to your PC using a USB data cable.</li>
                <li><b>Launch:</b> Open the app on both PC and smartphone. Windows will recognize it as a standard webcam.</li>
                <li><b>Select Device:</b> In this application, click <b>Refresh Cameras</b>, select <b>Phone Camera (USB / Virtual)</b>, and choose your virtual device.</li>
            </ol>
            <p style="color:#9d9d9d; font-size:11px;">Tip: USB mode provides lower latency and avoids Wi-Fi network congestion.</p>
        """)
        usb_layout.addWidget(usb_browser)
        tabs.addTab(usb_widget, "🔌 USB / Virtual Webcam (Method B)")

        layout.addWidget(tabs, stretch=1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
