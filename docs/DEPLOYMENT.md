# Deployment Guide — ArSL Translation Overlay

## Prerequisites

- Windows 10/11, Ubuntu 20.04+, or macOS 12+
- Python 3.10+
- Webcam (for live recognition)
- NVIDIA GPU recommended for PyTorch training

## Installation

### Development Install

```bash
git clone <repo-url>
cd arsl-overlay
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[all]"
python src/main.py
```

### Docker

```bash
docker compose build
# Run with display:
docker compose up
# Training with GPU:
docker compose --profile training up
```

### Windows Standalone Executable

```bash
pip install pyinstaller
make build-exe
# Output: dist/ArSL Overlay.exe
```

Or use NSIS for a full installer:

```bash
pip install pyinstaller nsis
pyinstaller --name "ArSL Overlay" --onedir --windowed src/main.py
makensis installer.nsi
```

## Configuration

The application reads configuration in order:

1. `config/default.yaml` — shipped defaults
2. `config/user.yaml` — user overrides (gitignored)
3. Environment variables (`.env`) — deployment overrides

### Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `performance.profile` | `balanced` | `max_quality`, `balanced`, `max_performance`, `battery_saver` |
| `dialect.active` | `iraqi` | Active sign language dialect |
| `recognition.hybrid.word_threshold` | `0.85` | Word recognition confidence threshold |
| `overlay.enabled` | `true` | Enable transparent overlay |

## Performance Profiles

| Profile | FPS Target | Resolution | Model Complexity |
|---------|-----------|------------|------------------|
| max_quality | 60 | 1280x720 | Full |
| balanced | 30 | 640x480 | Lite |
| max_performance | 30 | 320x240 | Lite |
| battery_saver | 15 | 320x240 | Lite |

## Data Directories

```
data/
├── datasets/       # Gesture samples per dialect
├── models/         # Trained models per dialect
│   ├── iraqi/
│   ├── saudi/
│   └── egyptian/
├── cache/          # Temporary cached data
├── vocabulary/     # Gesture vocabulary files
└── logs/           # Application logs
```

## Training

```bash
# Collect static gesture dataset first (via GUI Dataset tab)
make train-static
# Collect dynamic gesture sequences
make train-dynamic
```

## Troubleshooting

- **Camera not detected**: Run `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
- **Overlay not showing**: Ensure `QT_QPA_PLATFORM` is not set to `offscreen`
- **Low FPS**: Switch to `max_performance` profile in Settings
