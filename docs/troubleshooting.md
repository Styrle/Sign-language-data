# Troubleshooting Guide

This guide covers common issues and solutions for the BSL Data Extraction Tool.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [CLI Issues](#cli-issues)
3. [Recording Tool Issues](#recording-tool-issues)
4. [Processing Issues](#processing-issues)
5. [Validation Errors](#validation-errors)
6. [Performance Issues](#performance-issues)

---

## Installation Issues

### "Command not found: bsl-tool"

**Cause**: Package not installed or not in PATH.

**Solutions**:

1. Ensure you installed the package:
   ```bash
   pip install -e .
   ```

2. Check if the entry point is registered:
   ```bash
   pip show bsl-data-extraction
   ```

3. If using a virtual environment, ensure it's activated:
   ```bash
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

4. Try running directly:
   ```bash
   python -m scripts.cli --help
   ```

### "ModuleNotFoundError: No module named 'pydantic'"

**Cause**: Dependencies not installed.

**Solution**:
```bash
pip install -e ".[dev]"
```

Or install dependencies manually:
```bash
pip install pydantic numpy scipy click
```

### "Python version not supported"

**Cause**: Python version too old.

**Solution**: Install Python 3.10 or newer:
```bash
# Check version
python --version

# Install with pyenv (recommended)
pyenv install 3.11.0
pyenv local 3.11.0
```

---

## CLI Issues

### "No such command 'xxx'"

**Cause**: Typo in command name or outdated installation.

**Solution**:
1. Check available commands:
   ```bash
   bsl-tool --help
   ```

2. Reinstall the package:
   ```bash
   pip install -e . --force-reinstall
   ```

### "Error: Invalid value for 'PATH'"

**Cause**: File or directory doesn't exist.

**Solution**:
1. Verify the path exists:
   ```bash
   ls /path/to/file
   ```

2. Use absolute paths to avoid confusion:
   ```bash
   bsl-tool analyze kaggle $(pwd)/data/kaggle-bsl/
   ```

### Configuration not loading

**Cause**: Config file syntax error or wrong location.

**Solutions**:

1. Validate JSON syntax:
   ```bash
   python -c "import json; json.load(open('bsl-tool.config.json'))"
   ```

2. Specify config explicitly:
   ```bash
   bsl-tool --config ./bsl-tool.config.json info
   ```

3. Check for default location:
   ```bash
   ls -la bsl-tool.config.json
   ```

### Dry run showing unexpected behavior

**Cause**: Misunderstanding dry run scope.

**Clarification**:
- `--dry-run` prevents file writes only
- It still reads files and runs computations
- Use with `-v` to see what would be written

---

## Recording Tool Issues

### "Camera access denied"

**Causes**:
- Browser permissions not granted
- Camera in use by another app
- HTTPS required for some browsers

**Solutions**:

1. Check browser permissions:
   - Click padlock icon in address bar
   - Allow camera access

2. Close other apps using camera:
   ```bash
   # Linux: find camera processes
   lsof /dev/video0

   # macOS: check camera usage
   lsof | grep -i camera
   ```

3. Use HTTPS or localhost:
   ```bash
   # Serve with Python
   python -m http.server 8000
   # Access at http://localhost:8000
   ```

### "Failed to load model"

**Causes**:
- Network connectivity issues
- CDN blocked by firewall
- Browser cache corrupted

**Solutions**:

1. Check network:
   ```bash
   curl -I https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240
   ```

2. Clear browser cache and reload

3. Download model for offline use (advanced)

### No hand detection

**Causes**:
- Poor lighting
- Hands out of frame
- Camera resolution too low

**Solutions**:

1. Improve lighting:
   - Face a window or lamp
   - Avoid backlight

2. Adjust hand position:
   - Keep hands in center of frame
   - Move closer to camera

3. Check camera settings:
   - Ensure auto-focus is enabled
   - Increase exposure if needed

### Landmarks jittering

**Causes**:
- Rapid hand movement
- Inconsistent lighting
- Low frame rate

**Solutions**:

1. Move hands more slowly
2. Stabilize lighting
3. Close other browser tabs
4. Use a better webcam

### Mirror mode confusion

**Issue**: Signs appear reversed.

**Solution**:
- Toggle mirror mode with the checkbox
- For BSL, mirror mode ON is usually correct (matches viewer's perspective)

---

## Processing Issues

### "No valid recordings loaded"

**Causes**:
- Wrong file format
- Missing required fields
- Files in wrong directory

**Solutions**:

1. Check file structure:
   ```bash
   python -c "import json; print(json.load(open('recording.json')).keys())"
   ```

2. Verify required fields exist:
   ```json
   {
     "sign_id": "required",
     "frames": [{"timestamp_ms": 0, "right_hand": {...}}]
   }
   ```

3. Ensure files are in correct directory:
   ```bash
   ls data/raw-recordings/*.json
   ```

### "Processing failed - no valid frames"

**Causes**:
- All frames below confidence threshold
- No hand detections in recording
- Corrupted landmark data

**Solutions**:

1. Lower confidence threshold:
   ```bash
   bsl-tool process single recording.json --min-confidence 0.3
   ```

2. Analyze the recording:
   ```bash
   bsl-tool analyze recordings ./recording.json --detailed
   ```

3. Re-record with better conditions

### Normalization produces wrong results

**Causes**:
- Landmarks in wrong order
- Missing wrist landmark
- Invalid coordinate values

**Solutions**:

1. Validate landmarks:
   ```bash
   bsl-tool validate landmarks recording.json
   ```

2. Check for NaN or Inf values:
   ```python
   import json
   import math
   data = json.load(open('recording.json'))
   for frame in data['frames']:
       for lm in frame.get('right_hand', {}).get('landmarks', []):
           if not all(math.isfinite(v) for v in [lm['x'], lm['y'], lm['z']]):
               print(f"Invalid: {lm}")
   ```

### Angle calculation errors

**Causes**:
- Collinear points (zero-length vectors)
- Extreme landmark positions
- Invalid normalization

**Solutions**:

1. Check for warnings in verbose output:
   ```bash
   bsl-tool -vv process single recording.json
   ```

2. Disable anatomical validation for debugging:
   ```json
   {"validate_anatomical_limits": false}
   ```

---

## Validation Errors

### "Landmarks must have exactly 21 points"

**Cause**: Missing or extra landmarks in data.

**Solution**:
Check and fix landmark count:
```python
data = json.load(open('recording.json'))
for i, frame in enumerate(data['frames']):
    if 'right_hand' in frame:
        count = len(frame['right_hand']['landmarks'])
        if count != 21:
            print(f"Frame {i}: {count} landmarks (expected 21)")
```

### "Coordinate out of range"

**Cause**: Landmarks outside expected bounds.

**Solution**:
For raw data, coordinates should be 0-1 (normalized to image).
For processed data, coordinates are in canonical space (~-1 to 1).

Check and clip:
```python
for lm in landmarks:
    lm['x'] = max(-10, min(10, lm['x']))
    lm['y'] = max(-10, min(10, lm['y']))
    lm['z'] = max(-10, min(10, lm['z']))
```

### "Invalid angle value"

**Cause**: Calculated angle outside valid range.

**Solution**:
Angles should be -180 to 180 degrees. Values outside this range indicate calculation errors:

```bash
bsl-tool -v validate dictionary bsl-dictionary.json
```

### Dictionary validation failures

**Causes**:
- Missing required fields
- Invalid category values
- Malformed entries

**Solution**:
Run strict validation:
```bash
bsl-tool validate dictionary bsl-dictionary.json --strict
```

---

## Performance Issues

### Processing is very slow

**Causes**:
- Large number of files
- Complex outlier detection
- Single-threaded execution

**Solutions**:

1. Use batch processing:
   ```bash
   bsl-tool process batch ./data/raw-recordings/
   ```

2. Limit files for testing:
   ```bash
   bsl-tool import kaggle data.csv --limit 100
   ```

3. Simplify outlier detection:
   ```json
   {"outlier_method": "none"}
   ```

### High memory usage

**Causes**:
- Loading all files at once
- Large batch sizes
- NumPy array accumulation

**Solutions**:

1. Process in smaller batches:
   ```bash
   # Process subdirectories separately
   for dir in data/raw-recordings/*/; do
       bsl-tool process batch "$dir"
   done
   ```

2. Reduce batch size in config:
   ```json
   {"batch_size": 50}
   ```

### Recording tool lag

**Causes**:
- High resolution video
- Complex model inference
- Too many browser tabs

**Solutions**:

1. Reduce video resolution (in camera settings)
2. Close unnecessary tabs
3. Use a dedicated browser profile
4. Disable browser extensions

---

## Getting Help

### Verbose Output

Use `-v` flags for more information:
```bash
bsl-tool -v pipeline run      # Basic verbose
bsl-tool -vv pipeline run     # More detail
bsl-tool -vvv pipeline run    # Maximum detail
```

### Debug Mode

For Python debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Reporting Issues

When reporting bugs, include:

1. Command that failed
2. Full error message
3. Output of `bsl-tool info`
4. Sample of problematic data (if applicable)
5. Python version and OS

File issues at the project repository or contact the development team.
