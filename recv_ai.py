import asyncio, socket, struct, time, os, json, logging, random
import cv2, numpy as np
import websockets
from datetime import datetime
from typing import Optional, Dict, Any

# ──────────────────────────────────────────────────────────────────────────────
# Server configuration - MATCHES HOLOLENS CLIENT
LISTEN_TCP = ('0.0.0.0', 8080)        # HL -> PC frames (changed from 8766)
WS_JSON    = ('0.0.0.0', 8770)        # PC -> HL JSON
WS_VIDEO   = ('0.0.0.0', 8771)        # PC -> HL binary (jpg/png/rgba bytes)
WS_AI_DETECTION = ('0.0.0.0', 8772)   # AI Detection WebSocket (bidirectional)

# Limits and safety
MAX_FRAME_SIZE   = 10_000_000  # 10MB max frame
MIN_FRAME_SIZE   = 1_000       # 1KB min frame
STATS_INTERVAL   = 2.0         # seconds
MAX_FRAME_ID     = 1_000_000   # Reset frame counter at this point

# Global client sets
json_clients = set()
video_clients = set()
ai_detection_clients = set()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('hl_server.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
class FrameStats:
    """Track frame reception statistics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.bytes_received = 0
        self.decode_errors = 0
        self.last_size = (0, 0)

    def update(self, frame_size: int, decode_success: bool, img_size: tuple = (0, 0)):
        self.frame_count += 1
        self.bytes_received += frame_size
        if not decode_success:
            self.decode_errors += 1
        if decode_success:
            self.last_size = img_size

    def get_rates(self) -> Dict[str, float]:
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return {"fps": 0, "mbps": 0, "decode_error_rate": 0}
        fps = self.frame_count / elapsed
        mbps = (self.bytes_received * 8) / elapsed / 1e6
        error_rate = self.decode_errors / self.frame_count if self.frame_count > 0 else 0
        return {
            "fps": fps,
            "mbps": mbps,
            "decode_error_rate": error_rate,
            "total_frames": self.frame_count,
            "avg_frame_size": self.bytes_received / self.frame_count if self.frame_count > 0 else 0,
        }

# ──────────────────────────────────────────────────────────────────────────────
def generate_random_detections() -> Dict[str, Any]:
    """Generate random bounding box detections - MATCHES HOLOLENS CLIENT FORMAT EXACTLY"""
    detections = []
    num_detections = random.randint(1, 3)
    for _ in range(num_detections):
        x = random.uniform(0.1, 0.6)
        y = random.uniform(0.1, 0.6)
        width  = random.uniform(0.1, 0.3)
        height = random.uniform(0.1, 0.4)
        if x + width > 1.0:  width  = 1.0 - x
        if y + height > 1.0: height = 1.0 - y
        detections.append({
            "class": random.choice(["person", "car", "dog", "cat", "bottle"]),
            "confidence": round(random.uniform(0.7, 0.95), 3),
            "bbox": {"x": round(x, 3), "y": round(y, 3), "width": round(width, 3), "height": round(height, 3)},
        })
    return {"detections":detections}

async def ai_detection_ws_handler(websocket):
    """Handle AI Detection WebSocket connections (bidirectional)"""
    client_addr = websocket.remote_address
    logger.info(f"AI DETECTION WS connected: {client_addr}")
    ai_detection_clients.add(websocket)
    try:
        detection_task = asyncio.create_task(send_random_detections(websocket))
        async for message in websocket:
            if isinstance(message, bytes):
                await handle_ai_video_frame(message, websocket)
            else:
                logger.debug(f"AI Detection JSON from {client_addr}: {message[:100]}")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"AI DETECTION WS client {client_addr} disconnected normally")
    except Exception as e:
        logger.warning(f"AI DETECTION WS client {client_addr} error: {e}")
    finally:
        ai_detection_clients.discard(websocket)
        logger.info(f"AI DETECTION WS client {client_addr} removed")
        if 'detection_task' in locals():
            detection_task.cancel()

async def send_random_detections(websocket):
    """Send random detection results to HoloLens every few seconds"""
    try:
        while True:
            await asyncio.sleep(2.0)
            detection_result = generate_random_detections()
            json_str = json.dumps(detection_result)
            logger.info(f"Sending detection to HoloLens: {json_str}")
            await websocket.send(json_str)  # text message
            logger.info(f"Sent {len(detection_result['detections'])} random detections to HoloLens")
    except asyncio.CancelledError:
        logger.info("Detection sending task cancelled")
    except Exception as e:
        logger.warning(f"Error sending detections: {e}")

async def handle_ai_video_frame(frame_data: bytes, websocket):
    """Process video frame received from HoloLens for AI detection"""
    try:
        if len(frame_data) < 16:
            logger.warning(f"AI frame too small: {len(frame_data)} bytes")
            return
        frame_id, width, height, data_size = struct.unpack('!IIII', frame_data[:16])
        if data_size != len(frame_data) - 16:
            logger.warning(f"AI frame size mismatch: expected {data_size}, got {len(frame_data) - 16}")
            return
        if width <= 0 or height <= 0 or width > 4096 or height > 4096:
            logger.warning(f"AI frame invalid dimensions: {width}x{height}")
            return
        bgra_bytes = frame_data[16:]
        expected_size = width * height * 4
        if len(bgra_bytes) != expected_size:
            logger.warning(f"AI frame data size mismatch: expected {expected_size}, got {len(bgra_bytes)}")
            return
        logger.debug(f"Received AI video frame: {frame_id}, {width}x{height}, {len(bgra_bytes)} bytes")
    except Exception as e:
        logger.warning(f"Error processing AI video frame: {e}")

async def json_ws_handler(websocket):
    """Handle JSON WebSocket connections"""
    client_addr = websocket.remote_address
    logger.info(f"JSON WS connected: {client_addr}")
    json_clients.add(websocket)
    try:
        async for message in websocket:
            logger.debug(f"Received JSON message from {client_addr}: {message[:100]}")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"JSON WS client {client_addr} disconnected normally")
    except Exception as e:
        logger.warning(f"JSON WS client {client_addr} error: {e}")
    finally:
        json_clients.discard(websocket)
        logger.info(f"JSON WS client {client_addr} removed")

async def video_ws_handler(websocket):
    """Handle video WebSocket connections - send fake green video frames"""
    client_addr = websocket.remote_address
    logger.info(f"VIDEO WS connected: {client_addr}")
    video_clients.add(websocket)
    try:
        while True:
            frame = np.zeros((256, 256, 4), dtype=np.uint8)
            frame[:, :, 1] = 255   # Green
            frame[:, :, 3] = 255   # Alpha
            await websocket.send(frame.tobytes())
            await asyncio.sleep(0.05)  # 20 fps
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"VIDEO WS client {client_addr} disconnected normally")
    finally:
        video_clients.discard(websocket)

async def ws_json_broadcast(payload: Dict[str, Any]):
    """Broadcast JSON data to all JSON WebSocket clients"""
    if not json_clients:
        return
    data = json.dumps(payload, default=str)
    dead = []
    for ws in list(json_clients):
        try:
            await ws.send(data)
        except websockets.exceptions.ConnectionClosed:
            dead.append(ws)
        except Exception as e:
            logger.warning(f"JSON broadcast error to {ws.remote_address}: {e}")
            dead.append(ws)
    for ws in dead:
        json_clients.discard(ws)

async def ws_video_broadcast(binary: bytes):
    """Broadcast binary video data to all video WebSocket clients"""
    if not video_clients:
        return
    dead = []
    for ws in list(video_clients):
        try:
            await ws.send(binary)
        except websockets.exceptions.ConnectionClosed:
            dead.append(ws)
        except Exception as e:
            logger.warning(f"VIDEO broadcast error to {ws.remote_address}: {e}")
            dead.append(ws)
    for ws in dead:
        video_clients.discard(ws)

# ──────────────────────────────────────────────────────────────────────────────
def read_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly n bytes from socket or return None if connection lost"""
    if n <= 0:
        return None
    buf = bytearray()
    start_time = time.time()
    while len(buf) < n:
        if time.time() - start_time > 10.0:
            logger.warning(f"Socket read timeout after 10s, wanted {n} bytes, got {len(buf)}")
            return None
        try:
            remaining = n - len(buf)
            chunk = sock.recv(min(remaining, 65536))
            if not chunk:
                logger.info(f"Socket closed during read, got {len(buf)}/{n} bytes")
                return None
            buf.extend(chunk)
        except socket.timeout:
            logger.warning("Socket timeout during read")
            return None
        except Exception as e:
            logger.warning(f"Socket read error: {e}")
            return None
    return bytes(buf)

# ──────────────────────────────────────────────────────────────────────────────
def validate_and_decode_frame(bgra_data: bytes, width: int, height: int) -> tuple[bool, tuple[int, int]]:
    """
    Validate BGRA data and return (success, (H, W)).
    NOTE: HoloLens sends BGRA (B,G,R,A) bytes.
    """
    try:
        expected_size = width * height * 4
        if len(bgra_data) != expected_size:
            return False, (0, 0)
        # Try a cheap reshape to ensure it’s well-formed
        _ = np.frombuffer(bgra_data, dtype=np.uint8).reshape((height, width, 4))
        return True, (height, width)  # return (H, W) so later `h, w = img_size` works
    except Exception as e:
        logger.debug(f"Frame decode error: {e}")
        return False, (0, 0)

# ... [imports and other code remain unchanged above]

def save_diagnostic_frame(bgra_data: bytes, width: int, height: int, frame_id: int, is_first: bool = False):
    """Save frame for diagnostic purposes (treat input as BGRA)."""
    try:
        os.makedirs("hl_diagnostics", exist_ok=True)
        bgra = np.frombuffer(bgra_data, dtype=np.uint8).reshape((height, width, 4))
        # Writable copy for OpenCV
        bgr = cv2.cvtColor(bgra.copy(), cv2.COLOR_BGRA2BGR)

        if is_first:
            filepath = "hl_diagnostics/first_frame.jpg"
            cv2.imwrite(filepath, bgr)
            logger.info(f"Saved first frame: {filepath} ({len(bgra_data)} bytes)")

        if frame_id % 100 == 0:
            filepath = f"hl_diagnostics/frame_{frame_id:06d}.jpg"
            cv2.imwrite(filepath, bgr)
            logger.debug(f"Saved diagnostic frame: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to save diagnostic frame: {e}")


def tcp_loop(loop: asyncio.AbstractEventLoop):
    """Main TCP receiver loop - MATCHES HOLOLENS CLIENT FORMAT"""
    logger.info("Starting TCP receiver thread")

    os.makedirs("hl_diagnostics", exist_ok=True)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(LISTEN_TCP)
    server_sock.listen(1)

    logger.info(f"TCP listening on {LISTEN_TCP}")
    logger.info(f"JSON WebSocket on ws://{WS_JSON[0]}:{WS_JSON[1]}")
    logger.info(f"VIDEO WebSocket on ws://{WS_VIDEO[0]}:{WS_VIDEO[1]}")
    logger.info(f"AI DETECTION WebSocket on ws://{WS_AI_DETECTION[0]}:{WS_AI_DETECTION[1]}")

    while True:
        try:
            conn, addr = server_sock.accept()
            logger.info(f"HoloLens connected from {addr}")

            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(15.0)

            stats = FrameStats()
            frame_id = 0
            first_frame_saved = False
            connection_start = datetime.now()

            try:
                while True:
                    # Read header length
                    header_length_data = read_exact(conn, 4)
                    if header_length_data is None:
                        logger.info("Connection closed (no header length)")
                        break

                    header_length = struct.unpack("!I", header_length_data)[0]
                    if header_length < 10 or header_length > 1000:
                        logger.warning(f"Invalid header length: {header_length}, skipping")
                        continue

                    # Read JSON header
                    header_data = read_exact(conn, header_length)
                    if header_data is None:
                        logger.warning(f"Failed to read header ({header_length} bytes)")
                        break

                    try:
                        header_json = json.loads(header_data.decode('utf-8'))
                        width     = header_json.get('width', 0)
                        height    = header_json.get('height', 0)
                        data_size = header_json.get('dataSize', 0)
                        timestamp = header_json.get('timestamp', 0)
                        logger.debug(f"Received frame header: {width}x{height}, {data_size} bytes, ts={timestamp}")
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON header: {e}")
                        continue

                    if data_size < MIN_FRAME_SIZE or data_size > MAX_FRAME_SIZE:
                        logger.warning(f"Invalid frame data size: {data_size}")
                        continue

                    # Read BGRA payload
                    bgra_data = read_exact(conn, data_size)
                    if bgra_data is None:
                        logger.warning(f"Failed to read frame data ({data_size} bytes)")
                        break

                    frame_id += 1
                    if frame_id > MAX_FRAME_ID:
                        frame_id = 1

                    decode_success, img_size = validate_and_decode_frame(bgra_data, width, height)
                    stats.update(data_size, decode_success, img_size)

                    if not first_frame_saved:
                        save_diagnostic_frame(bgra_data, width, height, frame_id, is_first=True)
                        first_frame_saved = True
                    if frame_id % 100 == 0:
                        save_diagnostic_frame(bgra_data, width, height, frame_id)

                    payload = {
                        "frame_id": frame_id,
                        "size": list(img_size),  # [H, W]
                        "frame_bytes": data_size,
                        "decode_success": decode_success,
                        "timestamp": datetime.now().isoformat(),
                        "connection_duration": (datetime.now() - connection_start).total_seconds(),
                    }

                    if time.time() - stats.start_time >= STATS_INTERVAL:
                        rates = stats.get_rates()
                        payload.update(rates)
                        logger.info(
                            f"Stats: {rates['fps']:.1f} fps, "
                            f"{rates['mbps']:.2f} Mbps, "
                            f"Size(HxW): {img_size}, "
                            f"Avg frame: {rates['avg_frame_size']:.0f}B, "
                            f"Errors: {rates['decode_error_rate']:.1%}"
                        )
                        stats.reset()

                    # If valid, display and broadcast
                    if decode_success:
                        try:
                            h, w = img_size  # img_size is (H, W)
                            bgra = np.frombuffer(bgra_data, dtype=np.uint8).reshape((h, w, 4))

                            # Drop alpha and copy for OpenCV
                            bgr = bgra[:, :, :3].copy()

                            # ✅ Writable now, rectangle works
                            cv2.rectangle(bgr, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 4)

                            # Show live stream window
                            cv2.imshow("HoloLens Stream", bgr)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                logger.info("User pressed 'q' – closing OpenCV window")
                                cv2.destroyAllWindows()
                                conn.close()
                                return  # exit tcp_loop

                            # Convert BGR -> RGBA for broadcasting
                            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
                            out_frame = rgba.tobytes()

                        except Exception as e:
                            logger.warning(f"Box drawing / viewer failed: {e}")
                            out_frame = bgra_data  # fall back to original bytes
                    else:
                        out_frame = bgra_data  # invalid; forward as-is

                    # Broadcast telemetry + video
                    asyncio.run_coroutine_threadsafe(ws_json_broadcast(payload), loop)
                    asyncio.run_coroutine_threadsafe(ws_video_broadcast(out_frame), loop)

            except Exception as e:
                logger.error(f"Error handling connection from {addr}: {e}")

            finally:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                try:
                    conn.close()
                except:
                    pass

                duration = (datetime.now() - connection_start).total_seconds()
                final_stats = stats.get_rates()
                logger.info(
                    f"HoloLens {addr} disconnected after {duration:.1f}s, "
                    f"processed {final_stats['total_frames']} frames"
                )

        except Exception as e:
            logger.error(f"TCP server error: {e}")
            time.sleep(1)


# ──────────────────────────────────────────────────────────────────────────────
async def main():
    """Main async entry point"""
    logger.info("Starting HoloLens AI Detection Server")

    loop = asyncio.get_running_loop()

    json_server = await websockets.serve(
        json_ws_handler, WS_JSON[0], WS_JSON[1],
        ping_interval=20, ping_timeout=10, max_size=16*1024*1024
    )

    video_server = await websockets.serve(
        video_ws_handler, WS_VIDEO[0], WS_VIDEO[1],
        ping_interval=20, ping_timeout=10, max_size=16*1024*1024
    )

    ai_detection_server = await websockets.serve(
        ai_detection_ws_handler, WS_AI_DETECTION[0], WS_AI_DETECTION[1],
        ping_interval=20, ping_timeout=10, max_size=16*1024*1024
    )

    logger.info("WebSocket servers started")
    logger.info(f"AI Detection ready on ws://{WS_AI_DETECTION[0]}:{WS_AI_DETECTION[1]}")

    import threading
    tcp_thread = threading.Thread(target=tcp_loop, args=(loop,), daemon=True)
    tcp_thread.start()

    logger.info("Server fully initialized and running")
    logger.info("Ready to receive frames from HoloLens and send random detections!")

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        json_server.close()
        video_server.close()
        ai_detection_server.close()
        await json_server.wait_closed()
        await video_server.wait_closed()
        await ai_detection_server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        logger.critical(f"Server crashed: {e}")
        raise
