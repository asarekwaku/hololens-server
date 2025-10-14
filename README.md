# HoloLens AI Detection Server

Real-time object detection server using YOLO-World for zero-shot detection on HoloLens camera streams.

## Features

- **Zero-shot Object Detection**: Detect any object by simply typing its name
- **YOLO-World Integration**: State-of-the-art real-time detection
- **Live HoloLens Streaming**: Receives frames from HoloLens via TCP
- **WebSocket Communication**: Sends detection results back to HoloLens
- **Dynamic Object Switching**: Change target object on-the-fly without restarting

## Installation

```bash
pip install ultralytics opencv-python numpy websockets
```

## Usage

### 1. Start the Server

```bash
python3 recv_ai.py
```

The server will:
- Load the YOLO-World model (downloads on first run)
- Start listening on ports 8080 (TCP) and 8772 (WebSocket)
- Begin detecting the default object: "person"

### 2. Change Target Object

Simply edit the `target_object.txt` file:

```bash
echo "cat" > target_object.txt
echo "bottle" > target_object.txt
echo "backpack" > target_object.txt
echo "car" > target_object.txt
```

The server will automatically detect the new object within 0.5 seconds!

### 3. Connect HoloLens

Configure your HoloLens app to connect to:
- **Server IP**: Check logs for current IP (e.g., `11.28.81.180`)
- **TCP Port**: 8080 (send frames)
- **WebSocket Port**: 8772 (receive detections)

## Supported Objects

YOLO-World supports detecting **any object** you can describe in text:
- Common objects: person, cat, dog, car, bottle, backpack, phone, laptop, etc.
- Specific objects: "red car", "open laptop", "coffee cup"
- Multiple words: "person wearing hat", "black cat"

## Output Format

Detections are sent as compact JSON:

```json
{"detections":[{"class":"person","confidence":0.85,"bbox":{"x":0.2,"y":0.3,"width":0.15,"height":0.25}}]}
```

Where:
- `class`: Detected object class
- `confidence`: Detection confidence (0-1)
- `bbox`: Normalized bounding box (x, y, width, height in 0-1 range)

## Configuration

Edit `recv_ai.py` to modify:
- Detection confidence threshold (default: 0.3)
- Update rate (default: 0.5s / 2 FPS)
- Model size (yolov8s-world.pt, yolov8m-world.pt, yolov8l-world.pt)

## Logs

Check `hl_server.log` for:
- Model loading status
- Frame reception stats
- Detection results
- Connection events

## Architecture

```
HoloLens Camera → TCP (8080) → Python Server → YOLO-World
                                      ↓
User Types Object → target_object.txt → Model Updates
                                      ↓
Detection Results → WebSocket (8772) → HoloLens → Renders Boxes
```

## Troubleshooting

**Model not loading?**
- First run downloads the model (~25MB)
- Check internet connection
- Server falls back to random detections if model fails

**No detections?**
- Lower confidence threshold in `recv_ai.py` (line 126)
- Try more generic object names ("person" vs "person wearing glasses")
- Check HoloLens is sending frames (watch logs)

**Connection timeout?**
- Verify server IP address matches HoloLens config
- Check firewall allows ports 8080 and 8772
- Ensure HoloLens and server are on same network

