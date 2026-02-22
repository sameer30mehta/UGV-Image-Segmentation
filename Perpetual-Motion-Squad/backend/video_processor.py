"""
TerrainAI Backend -- Video Processor
Extracts frames at fixed time intervals, processes each through the segmentation
pipeline, and reconstructs H.264-encoded output videos for browser playback.
"""
import cv2
import numpy as np
import os
import subprocess
import tempfile


def _get_ffmpeg():
    """Get path to ffmpeg binary."""
    # Try system ffmpeg first
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return 'ffmpeg'
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # Fall back to imageio-ffmpeg bundled binary
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found. Install imageio-ffmpeg: pip install imageio-ffmpeg")


def extract_frames_by_interval(video_path, interval_ms=200, max_frames=300):
    """
    Extract frames from a video at fixed time intervals.
    
    Args:
        video_path: Path to input video file
        interval_ms: Time between frames in milliseconds (100 or 200)
        max_frames: Maximum number of frames to extract
    
    Returns:
        dict with frames list and video metadata
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_ms = (total_frames / max(fps, 1)) * 1000
    
    frames = []
    timestamps = []
    
    current_ms = 0
    while current_ms < duration_ms and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamps.append(current_ms)
        current_ms += interval_ms
    
    cap.release()
    
    # Output FPS = 1000 / interval_ms (e.g., 200ms -> 5 FPS, 100ms -> 10 FPS)
    output_fps = 1000.0 / interval_ms
    
    return {
        'frames': frames,
        'timestamps': timestamps,
        'input_fps': fps,
        'output_fps': output_fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration_ms': duration_ms,
        'interval_ms': interval_ms,
        'extracted': len(frames),
    }


def _write_frames_to_raw(frames_rgb, temp_path, fps):
    """Write frames to a raw mp4v file using OpenCV."""
    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    for frame in frames_rgb:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def _reencode_h264(input_path, output_path):
    """Re-encode a video to H.264 for browser compatibility."""
    ffmpeg = _get_ffmpeg()
    subprocess.run([
        ffmpeg, '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ], check=True, capture_output=True)
    # Remove the raw temp file
    if os.path.exists(input_path):
        os.remove(input_path)


def stitch_video(frames_rgb, output_path, fps, width=None, height=None):
    """
    Stitch a list of RGB frames into a browser-compatible H.264 MP4 video.
    """
    if not frames_rgb:
        raise ValueError("No frames to stitch")
    
    h, w = frames_rgb[0].shape[:2]
    out_w = width or w
    out_h = height or h
    
    # Resize frames if needed
    resized = []
    for frame in frames_rgb:
        if frame.shape[:2] != (out_h, out_w):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        resized.append(frame)
    
    # Write raw mp4v to temp file, then re-encode to H.264
    temp_path = output_path + '.raw.mp4'
    _write_frames_to_raw(resized, temp_path, fps)
    _reencode_h264(temp_path, output_path)
    
    return output_path


def stitch_sidebyside(originals, processed, output_path, fps):
    """
    Stitch original and processed frames side-by-side into a browser-compatible video.
    """
    if not originals or not processed:
        raise ValueError("No frames to stitch")
    
    h, w = originals[0].shape[:2]
    combined_frames = []
    
    for orig, proc in zip(originals, processed):
        if proc.shape[:2] != (h, w):
            proc = cv2.resize(proc, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = np.hstack([orig, proc])
        combined_frames.append(combined)
    
    # Write raw, then re-encode
    temp_path = output_path + '.raw.mp4'
    _write_frames_to_raw(combined_frames, temp_path, fps)
    _reencode_h264(temp_path, output_path)
    
    return output_path
