# User Guide — ArSL Translation Overlay

## Getting Started

1. Connect your webcam
2. Run `python src/main.py`
3. The main window appears with 8 tabs
4. Go to **Recognition** tab → click **Start**
5. Sign in front of the camera
6. Captions appear on the transparent overlay

## Dashboard

The **Dashboard** tab shows system status, active dialect, and quick actions.

## Dataset Collection

1. Go to **Dataset** tab
2. Select dialect (Iraqi/Saudi/Egyptian)
3. Select gesture type (word/dynamic/alphabet)
4. Enter the gesture name in Arabic
5. Click **Start Recording** — perform the gesture
6. Click **Stop** when done
7. Repeat for each gesture

## Training

1. Collect at least 10-20 samples per gesture
2. Go to **Training** tab
3. Click **Train Static Models** for alphabet/word gestures
4. Click **Train Dynamic Models** for motion gestures
5. Models are saved to `data/models/{dialect}/`

## Recognition

The hybrid pipeline prioritizes:

1. **Word recognition** — whole word gestures (high confidence)
2. **Dynamic gestures** — motion-based signs (medium confidence)
3. **Alphabet** — finger-spelling individual letters (low confidence)

### Arabic Text Controls

- **Space**: Commit current word
- **Delete**: Remove last letter / last word
- **Clear**: Reset entire text

## Dialect Switching

- Use the **Dialect** menu in the menu bar
- Or select dialect from the **Recognition** tab
- Each dialect has independent datasets and models

## Overlay Configuration

1. Go to **Overlay** tab
2. Adjust position, size, font, opacity
3. Click **Apply Changes**
4. Toggle with `Ctrl+T` or system tray

## Analytics

The **Analytics** tab shows real-time:

- FPS gauge
- CPU / GPU usage
- Memory consumption
- Recognition accuracy
- Translation delay

## Performance Profiles

- **Max Quality**: 60 FPS, full resolution (requires powerful GPU)
- **Balanced**: 30 FPS, moderate settings (recommended)
- **Max Performance**: 30 FPS, low resolution (for low-end hardware)
- **Battery Saver**: 15 FPS, minimal resource usage (laptops)

## Troubleshooting

### Camera not working
- Ensure no other app is using the camera
- Try a different USB port
- Check `data/logs/arsl.log`

### Low FPS
- Switch to **Max Performance** profile
- Close other applications
- Reduce camera resolution

### Arabic text not displaying
- Ensure arabic-reshaper and python-bidi are installed
- Check overlay font supports Arabic characters
- Verify RTL mode is enabled

### Overlay not visible
- Press `Ctrl+T` to toggle overlay
- Check overlay position is on-screen
- Ensure no hardware overlay is blocking
