# Recording Guide

This guide explains how to use the web-based recording tool to capture BSL sign data.

## Overview

The recording tool captures hand landmarks using your webcam and MediaPipe's hand detection model. Recordings are saved as JSON files that can be imported into the processing pipeline.

## Requirements

- Modern web browser (Chrome, Firefox, Edge, or Safari)
- Webcam with decent resolution (720p or higher recommended)
- Good lighting conditions
- Quiet background (solid color preferred)

## Getting Started

### 1. Open the Recording Tool

Open `recording-tool/index.html` in your browser. You can either:

- Open the file directly (File → Open)
- Serve it locally with a simple HTTP server:
  ```bash
  cd recording-tool
  python -m http.server 8000
  # Then open http://localhost:8000
  ```

### 2. Allow Camera Access

When prompted, allow the browser to access your camera. The video feed should appear in the main panel.

### 3. Wait for Model Loading

The status indicator will show "Loading hand detection model..." while the TensorFlow.js model downloads. This may take a few seconds on first load.

Once ready, you'll see "Ready to record" with a green indicator.

## Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  [Camera Select ▼]  [Mirror ☑]                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                                                                  │
│                     Video Feed                                   │
│                  (with landmark overlay)                         │
│                                                                  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Status: Ready to record                                    ● ● │
├───────────────────────┬─────────────────────────────────────────┤
│  Sign Selection       │  Recording Controls                     │
│  ┌───────────────┐   │  Duration: [3] seconds                   │
│  │ Category ▼    │   │                                          │
│  └───────────────┘   │  [ Record ]  [ Stop ]                    │
│  ┌───────────────┐   │                                          │
│  │ Sign ▼        │   │  Progress: 12/26 (46%)                   │
│  └───────────────┘   │  ███████░░░░░░░                          │
│                      │                                          │
│  Description:        │                                          │
│  Wave hand with...   │                                          │
└───────────────────────┴─────────────────────────────────────────┘
```

## Recording a Sign

### Step 1: Select the Sign

1. Choose a category from the dropdown (e.g., "Alphabet", "Greetings")
2. Select the specific sign to record
3. Read the description to understand the sign

### Step 2: Prepare Your Position

- Position your hand(s) in the camera frame
- Ensure good lighting on your hands
- Verify the landmark overlay is tracking accurately
- For two-handed signs, both hands should be visible

### Step 3: Set Recording Duration

Default is 3 seconds. Adjust if needed:
- **1-2 seconds**: For simple, static signs
- **3-4 seconds**: For most signs
- **5+ seconds**: For complex or dynamic signs

### Step 4: Record

1. Click **Record** or press **R**
2. A countdown (3, 2, 1) will appear
3. Perform the sign naturally
4. Recording stops automatically

### Step 5: Review

After recording, a review panel appears:

```
┌─────────────────────────────────────────┐
│  Review Recording                        │
├─────────────────────────────────────────┤
│  Frames: 90                             │
│  Duration: 3.0s                         │
│  Hands detected: 98%                    │
│  Quality: Good                          │
├─────────────────────────────────────────┤
│  [ Save ]         [ Discard ]           │
└─────────────────────────────────────────┘
```

- **Frames**: Number of frames captured
- **Duration**: Actual recording time
- **Hands detected**: Percentage of frames with detection
- **Quality**: Overall quality assessment

### Step 6: Save or Discard

- **Save**: Downloads the recording as JSON
- **Discard**: Deletes the recording and returns to ready state

## Recording Tips

### Lighting

Good:
- Natural daylight
- Front-facing soft light
- Even illumination on hands

Avoid:
- Backlight (window behind you)
- Harsh shadows
- Very dim conditions

### Background

Good:
- Solid color wall
- Uncluttered space
- Contrasting colors to skin tone

Avoid:
- Busy patterns
- Moving objects
- Other people's hands

### Hand Position

Good:
- Hands clearly in frame
- Not too close or far
- Natural, relaxed position

Avoid:
- Hands cut off at edges
- Very fast movements
- Occluding fingers

### Camera Angle

Good:
- Camera at chest/shoulder height
- Straight-on view
- Slight angle is okay

Avoid:
- Extreme angles
- Looking down at hands
- Profile view

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `R` | Start recording |
| `S` | Stop recording |
| `Space` | Save recording |
| `Escape` | Discard recording |
| `M` | Toggle mirror mode |

## File Output

Recordings are saved as JSON files with the naming convention:

```
{sign_id}_{timestamp}.json
```

Example: `hello_2024-01-15T10-30-00.json`

### Moving Files to Pipeline

After downloading, move files to the raw recordings directory:

```bash
mv ~/Downloads/*.json data/raw-recordings/
```

Or specify a custom download location in your browser settings.

## Troubleshooting

### "Camera access denied"

1. Check browser permissions (padlock icon in address bar)
2. Ensure no other app is using the camera
3. Try refreshing the page

### "Failed to load model"

1. Check internet connection (model loads from CDN)
2. Try a different browser
3. Clear browser cache and reload

### Poor tracking quality

1. Improve lighting
2. Reduce background clutter
3. Move hands closer to camera
4. Try slower, clearer movements

### Low frame rate

1. Close other browser tabs
2. Use a less demanding browser profile
3. Reduce video resolution in camera settings

### Mirror mode confusion

- **Mirror ON**: Video appears as if looking in a mirror (natural for signing)
- **Mirror OFF**: Video shows actual camera orientation

For BSL, mirror mode is typically preferred as signers naturally sign in a mirrored orientation.

## Quality Guidelines

### Minimum Requirements

- At least 60% of frames with hand detection
- No major tracking dropouts
- Hand fully visible in most frames

### Good Quality

- 80%+ frames with detection
- Smooth, consistent tracking
- Clear landmark positions

### Excellent Quality

- 95%+ frames with detection
- No tracking glitches
- Perfect landmark visibility

## Workflow Recommendations

### Recording Multiple Signs

1. Record related signs in batches (e.g., all alphabet letters)
2. Take short breaks to avoid fatigue
3. Review recordings periodically
4. Re-record if quality is poor

### Recording Variants

Some signs have regional or personal variations:

1. Record the "standard" version first
2. Note any variants in the metadata
3. Record variants as separate files if needed

### Building a Complete Dataset

1. Start with high-priority signs (common words)
2. Record 3-5 samples per sign for averaging
3. Use different sessions for variety
4. Track progress using the checklist

## Advanced Options

### Using External Camera

If you have multiple cameras:

1. Use the camera dropdown to select your preferred device
2. External webcams often provide better quality
3. USB cameras may need permission setup

### Custom Duration

For signs with specific timing:

1. Set appropriate duration before recording
2. Consider the full motion of the sign
3. Include start and end positions

### Session Notes

The metadata field can include notes about:

- Performer name/ID
- Recording conditions
- Known issues
- Variant information
