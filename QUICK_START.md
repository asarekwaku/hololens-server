# Quick Start Guide - YOLO-World Integration

## ✅ What's New

Your HoloLens server now has **real AI detection** using YOLO-World! Instead of random boxes, it detects actual objects from HoloLens camera frames.

## 🚀 How to Use

### 1. Start the Server
```bash
cd /Users/kwakuasare/Desktop/hololens-server
python3 recv_ai.py
```

Wait for: `YOLO-World model loaded successfully! Detecting: person`

### 2. Change What to Detect

**Just edit the file!**
```bash
echo "cat" > target_object.txt
echo "bottle" > target_object.txt
echo "backpack" > target_object.txt
echo "car" > target_object.txt
```

The server auto-updates every 0.5 seconds!

### 3. Connect HoloLens

Update your HoloLens app IP address to: **`11.28.81.180`**

Ports:
- TCP: `8080` (send frames)
- WebSocket: `8772` (receive detections)

## 📊 What Happens

```
HoloLens Camera Frame → Server (11.28.81.180:8080)
                             ↓
                        YOLO-World AI Model
                             ↓
                    Reads target_object.txt
                             ↓
                    Detects objects in frame
                             ↓
HoloLens Receives ← Compact JSON (8772)
                             ↓
                    Green boxes appear!
```

## 🎯 Supported Objects

Try any of these (or anything else!):
- **People**: `person`, `man`, `woman`, `child`
- **Animals**: `cat`, `dog`, `bird`, `horse`
- **Vehicles**: `car`, `truck`, `bus`, `bicycle`
- **Objects**: `bottle`, `cup`, `phone`, `laptop`, `backpack`, `chair`, `table`
- **Specific**: `red car`, `black cat`, `person wearing hat`

## 📝 Example Session

```bash
# Start detecting people
echo "person" > target_object.txt

# Switch to cats
echo "cat" > target_object.txt

# Switch to bottles
echo "bottle" > target_object.txt

# Detect phones
echo "phone" > target_object.txt
```

Each change takes effect within 0.5 seconds!

## 🔍 Check Logs

```bash
tail -f hl_server.log
```

Look for:
```
INFO - Target object updated to: cat
INFO - YOLO-World now detecting: cat
INFO - YOLO detected 2 cat(s)
INFO - Sending detection to HoloLens: {"detections":[...]}
```

## ⚙️ Configuration

Edit `recv_ai.py` to adjust:

**Line 126** - Detection confidence (default 0.3):
```python
results = yolo_model(frame, verbose=False, conf=0.3)  # Lower = more detections
```

**Line 207** - Update rate (default 0.5s):
```python
await asyncio.sleep(0.5)  # Faster = more updates
```

**Line 84** - Model size:
```python
yolo_model = YOLO('yolov8s-world.pt')  # s=small, m=medium, l=large
```

## 🐛 Troubleshooting

**No detections?**
- Try lowering confidence: change `conf=0.3` to `conf=0.2`
- Use simpler object names: "person" instead of "person with red shirt"
- Make sure HoloLens is sending frames (check logs)

**Model loading slow?**
- First run downloads model (~25MB + 338MB weights)
- Subsequent starts are much faster (~5 seconds)

**HoloLens not connecting?**
- Server IP changed to `11.28.81.180`
- Update HoloLens C++ code with new IP
- Check firewall allows ports 8080 and 8772

## 🎉 Success!

When working, you'll see:
1. Server logs showing detections
2. HoloLens logs showing received messages
3. Green boxes moving based on detected objects
4. Boxes following the object you specified in `target_object.txt`

Enjoy your AI-powered HoloLens! 🚀

