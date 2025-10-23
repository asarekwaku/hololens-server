# OPTIMIZED HoloLens Server - 3-Thread Architecture
# Thread 1: TCP Reception (FAST - no blocking)
# Thread 2: YOLO Detection (async, runs at 30 FPS for responsive boxes)
# Thread 3: OpenCV Display (separate, doesn't block reception)

import asyncio, socket, struct, time, os, json, logging
import cv2, numpy as np
import websockets
from datetime import datetime
from typing import Optional, Dict, Any
from ultralytics import YOLO
from pathlib import Path
import threading
import concurrent.futures
from collections import deque
import queue

# Server configuration
LISTEN_TCP = ('0.0.0.0', 8080)
WS_AI_DETECTION = ('0.0.0.0', 8772)
MAX_FRAME_SIZE = 10_000_000
MIN_FRAME_SIZE = 1_000
MAX_FRAME_ID = 1_000_000

# Thread-safe queues
display_queue = queue.Queue(maxsize=2)  # For OpenCV display (drop if full)
detection_frame_queue = deque(maxlen=1)  # For YOLO (only keep latest)
detection_frame_lock = threading.Lock()

# Global state
ai_detection_clients = set()
yolo_model = None
target_object = "person"
latest_depth_frame = None
latest_depth_lock = threading.Lock()
last_detections = []

# Logging setup
logging.basicConfig(
    level=logging.INFO,  # INFO level for performance (use DEBUG if diagnosing issues)
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('hl_server.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
def init_yolo_world():
    """Initialize YOLO model"""
    global yolo_model, target_object
    try:
        logger.info("Loading YOLO model...")
        try:
            yolo_model = YOLO('yolov8n-world.pt')
            logger.info("✅ Using YOLOv8n (nano) - fast mode!")
        except:
            yolo_model = YOLO('yolov8s-world.pt')
            logger.info("✅ Using YOLOv8s (small)")
        
        yolo_model.set_classes([target_object])
        logger.info(f"🎯 Detecting: {target_object}")
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO: {e}")

def read_target_object():
    """Read target from file"""
    global target_object, yolo_model
    try:
        target_file = Path("target_object.txt")
        if target_file.exists():
            new_target = target_file.read_text().strip()
            if new_target and new_target != target_object:
                target_object = new_target
                if yolo_model:
                    yolo_model.set_classes([target_object])
                logger.info(f"🎯 Target updated: {target_object}")
    except Exception as e:
        logger.warning(f"Error reading target: {e}")

def run_yolo_detection_on_frame(frame: np.ndarray) -> Dict[str, Any]:
    """Run YOLO on a single frame (called from detection thread)"""
    global yolo_model, target_object
    
    if yolo_model is None or frame is None:
        return {"detections": []}
    
    try:
        read_target_object()
        
        h, w = frame.shape[:2]
        logger.debug(f"🔍 Processing frame: {w}×{h}")
        
        # OPTIMIZATION: Run YOLO at native resolution with optimized settings
        # Lower imgsz for faster inference, lower conf for more detections
        results = yolo_model(frame, verbose=False, conf=0.25, imgsz=416, half=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Get pixel coordinates
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                
                # Normalize to 0-1 range
                x = x1 / w
                y = y1 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                logger.debug(f"📦 Detection: pixel=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) norm=({x:.3f},{y:.3f},{width:.3f},{height:.3f}) conf={conf:.3f}")
                
                detections.append({
                    "class": target_object,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "width": round(width, 3),
                        "height": round(height, 3)
                    }
                })
        
        logger.debug(f"✅ Found {len(detections)} detection(s)")
        return {"detections": detections}
    
    except Exception as e:
        logger.warning(f"YOLO error: {e}")
        return {"detections": []}

def get_depth_at_detection(detection, depth_frame, rgb_width, rgb_height):
    """Get depth value at detection center"""
    if depth_frame is None:
        return None
    
    depth_h, depth_w = depth_frame.shape
    bbox = detection['bbox']
    center_norm_x = bbox['x'] + bbox['width'] / 2.0
    center_norm_y = bbox['y'] + bbox['height'] / 2.0
    
    depth_x = int(center_norm_x * depth_w)
    depth_y = int(center_norm_y * depth_h)
    depth_x = max(0, min(depth_x, depth_w - 1))
    depth_y = max(0, min(depth_y, depth_h - 1))
    
    depth_mm = depth_frame[depth_y, depth_x]
    
    if depth_mm > 0 and depth_mm < 10000:
        return depth_mm / 1000.0
    
    return None

def detections_changed(new_dets, old_dets, threshold=0.02):
    """Check if detections changed significantly (lower threshold for faster updates)"""
    if len(new_dets) != len(old_dets):
        return True
    for new, old in zip(new_dets, old_dets):
        if abs(new['bbox']['x'] - old['bbox']['x']) > threshold:
            return True
        if abs(new['bbox']['y'] - old['bbox']['y']) > threshold:
            return True
        if abs(new['bbox']['width'] - old['bbox']['width']) > threshold:
            return True
        if abs(new['bbox']['height'] - old['bbox']['height']) > threshold:
            return True
    return False

# ──────────────────────────────────────────────────────────────────────────────
# THREAD 1: TCP Reception (FAST - no blocking!)
# ──────────────────────────────────────────────────────────────────────────────

def read_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly n bytes"""
    if n <= 0:
        return None
    buf = bytearray()
    start_time = time.time()
    while len(buf) < n:
        if time.time() - start_time > 10.0:
            return None
        try:
            remaining = n - len(buf)
            chunk = sock.recv(min(remaining, 65536))
            if not chunk:
                return None
            buf.extend(chunk)
        except Exception as e:
            logger.warning(f"Socket read error: {e}")
            return None
    return bytes(buf)

def tcp_reception_thread():
    """OPTIMIZED: TCP reception - NO BLOCKING operations!"""
    logger.info("🚀 TCP reception thread started")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(LISTEN_TCP)
    server_sock.listen(1)
    logger.info(f"📡 TCP listening on {LISTEN_TCP}")

    while True:
        try:
            conn, addr = server_sock.accept()
            logger.info(f"✅ HoloLens connected: {addr}")

            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(15.0)

            frame_count = 0
            fps_timer = time.time()
            fps_count = 0

            try:
                while True:
                    # Read header length
                    header_length_data = read_exact(conn, 4)
                    if not header_length_data:
                        break

                    header_length = struct.unpack("!I", header_length_data)[0]
                    if header_length < 10 or header_length > 1000:
                        continue

                    # Read JSON header
                    header_data = read_exact(conn, header_length)
                    if not header_data:
                        break

                    header = json.loads(header_data.decode('utf-8'))
                    width = header.get('width', 0)
                    height = header.get('height', 0)
                    data_size = header.get('dataSize', 0)
                    has_depth = header.get('hasDepth', False)
                    depth_data_size = header.get('depthDataSize', 0)
                    depth_width = header.get('depthWidth', 0)
                    depth_height = header.get('depthHeight', 0)

                    if data_size < MIN_FRAME_SIZE or data_size > MAX_FRAME_SIZE:
                        continue

                    # Read RGBA data
                    rgba_data = read_exact(conn, data_size)
                    if not rgba_data:
                        break

                    # Read depth data if present
                    if has_depth and depth_data_size > 0:
                        depth_bytes = read_exact(conn, depth_data_size)
                        if depth_bytes:
                            depth_frame = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((depth_height, depth_width))
                            with latest_depth_lock:
                                global latest_depth_frame
                                latest_depth_frame = depth_frame
                    
                    frame_count += 1
                    fps_count += 1
                    
                    # Convert to numpy (FAST operation)
                    rgba = np.frombuffer(rgba_data, dtype=np.uint8).reshape((height, width, 4))
                    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
                    
                    # Validate frame dimensions on first frame
                    if frame_count == 1:
                        logger.info(f"📐 First frame resolution: {width}×{height} (expected: 640×360 or 1280×720)")
                        if width != 640 and width != 1280:
                            logger.warning(f"⚠️  Unexpected width: {width} (expected 640 or 1280)")
                        if height != 360 and height != 720:
                            logger.warning(f"⚠️  Unexpected height: {height} (expected 360 or 720)")
                    
                    # Queue for display (non-blocking - drop if full)
                    try:
                        display_queue.put_nowait((rgb.copy(), frame_count))
                    except queue.Full:
                        pass  # Drop frame if display can't keep up
                    
                    # Queue for detection (keep only latest)
                    with detection_frame_lock:
                        detection_frame_queue.append(rgb.copy())
                    
                    # FPS logging
                    if time.time() - fps_timer >= 2.0:
                        fps = fps_count / (time.time() - fps_timer)
                        logger.info(f"📊 Reception: {fps:.1f} FPS, {width}×{height}, Frame #{frame_count}")
                        fps_timer = time.time()
                        fps_count = 0
            
            except Exception as e:
                logger.error(f"❌ Connection error: {e}")
            finally:
                conn.close()
                logger.info(f"📴 HoloLens disconnected: {addr}")
        
        except Exception as e:
            logger.error(f"❌ TCP server error: {e}")
            time.sleep(1)

# ──────────────────────────────────────────────────────────────────────────────
# THREAD 2: YOLO Detection (runs at ~30 FPS for responsive bounding boxes)
# ──────────────────────────────────────────────────────────────────────────────

async def yolo_detection_loop():
    """OPTIMIZED: YOLO detection at 20-30 FPS for responsive bounding boxes"""
    global last_detections
    logger.info("🔍 YOLO detection loop started")
    
    detection_count = 0
    fps_timer = time.time()
    
    while True:
        try:
            # Run at 20-30 Hz for fast, responsive bounding boxes
            await asyncio.sleep(0.033)  # ~30 FPS detection rate
            
            # Get latest frame
            with detection_frame_lock:
                if len(detection_frame_queue) == 0:
                    continue
                frame = detection_frame_queue[-1]
            
            # Run YOLO (async)
            loop = asyncio.get_running_loop()
            detection_result = await loop.run_in_executor(
                None, 
                run_yolo_detection_on_frame, 
                frame
            )
            
            # Add depth to detections if present
            if detection_result["detections"]:
                with latest_depth_lock:
                    depth_frame = latest_depth_frame
                
                for detection in detection_result["detections"]:
                    if depth_frame is not None:
                        depth_meters = get_depth_at_detection(
                            detection, depth_frame, 640, 360
                        )
                        detection["depth"] = depth_meters
                    else:
                        detection["depth"] = None
            
            # Send if changed (INCLUDING when detections become empty!)
            if detections_changed(detection_result["detections"], last_detections):
                json_str = json.dumps(detection_result, separators=(',', ':'))
                
                logger.info(f"📤 Sending to HoloLens: {json_str}")
                
                # Send to all connected HoloLens clients
                sent_count = 0
                for websocket in list(ai_detection_clients):
                    try:
                        await websocket.send(json_str)
                        sent_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to send to client: {e}")
                
                logger.debug(f"✅ Sent to {sent_count} client(s)")
                
                detection_count += 1
                last_detections = detection_result["detections"].copy()  # Clear if empty!
                
                # FPS logging
                if time.time() - fps_timer >= 5.0:
                    fps = detection_count / (time.time() - fps_timer)
                    logger.info(f"🎯 Detection: {fps:.1f} FPS, {len(detection_result['detections'])} objects")
                    fps_timer = time.time()
                    detection_count = 0
        
        except Exception as e:
            logger.warning(f"Detection loop error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# THREAD 3: OpenCV Display (separate, doesn't block reception)
# ──────────────────────────────────────────────────────────────────────────────

def opencv_display_thread():
    """OPTIMIZED: OpenCV display in separate thread"""
    logger.info("🖥️  OpenCV display thread started")
    
    while True:
        try:
            # Get frame from queue (blocking with timeout)
            frame, frame_id = display_queue.get(timeout=1.0)
            
            # Convert to BGR for display
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Get latest detections and draw boxes
            if last_detections:
                h, w = bgr.shape[:2]
                for det in last_detections:
                    bbox = det["bbox"]
                    x1 = int(bbox["x"] * w)
                    y1 = int(bbox["y"] * h)
                    x2 = int((bbox["x"] + bbox["width"]) * w)
                    y2 = int((bbox["y"] + bbox["height"]) * h)
                    
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"{det['class']} {det['confidence']:.2f}"
                    if det.get('depth'):
                        label += f" {det['depth']:.2f}m"
                    
                    cv2.putText(bgr, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display (this blocks, but it's OK - it's in a separate thread!)
            cv2.imshow("HoloLens Stream", bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User pressed 'q' - closing")
                break
        
        except queue.Empty:
            continue
        except Exception as e:
            logger.warning(f"Display error: {e}")
    
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Handler
# ──────────────────────────────────────────────────────────────────────────────

async def ai_detection_ws_handler(websocket):
    """Handle AI Detection WebSocket"""
    client_addr = websocket.remote_address
    logger.info(f"✅ AI Detection client connected: {client_addr}")
    ai_detection_clients.add(websocket)
    
    try:
        async for message in websocket:
            pass  # Just keep connection alive
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"📴 AI Detection client disconnected: {client_addr}")
    finally:
        ai_detection_clients.discard(websocket)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def main():
    """Main async entry point"""
    logger.info("🚀 Starting OPTIMIZED HoloLens AI Server")

    # Initialize YOLO
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, init_yolo_world)

    # Start WebSocket server
    ai_detection_server = await websockets.serve(
        ai_detection_ws_handler, 
        WS_AI_DETECTION[0], 
        WS_AI_DETECTION[1],
        ping_interval=20, 
        ping_timeout=10, 
        max_size=16*1024*1024
    )
    logger.info(f"✅ WebSocket on ws://{WS_AI_DETECTION[0]}:{WS_AI_DETECTION[1]}")
    
    # Start TCP reception thread
    tcp_thread = threading.Thread(target=tcp_reception_thread, daemon=True)
    tcp_thread.start()
    logger.info("✅ TCP thread started")
    
    # Start OpenCV display thread
    display_thread = threading.Thread(target=opencv_display_thread, daemon=True)
    display_thread.start()
    logger.info("✅ Display thread started")
    
    # Start YOLO detection loop
    detection_task = asyncio.create_task(yolo_detection_loop())
    logger.info("✅ Detection loop started")
    
    logger.info("✅ Server fully operational!")

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    finally:
        detection_task.cancel()
        ai_detection_server.close()
        await ai_detection_server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        logger.critical(f"❌ Server crashed: {e}")
        raise
