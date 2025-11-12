from typing import List, Tuple, Optional
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger

class VideoProcessor:
    """Handles video frame extraction, superframe generation, and caching with LRU policy."""
    
    def __init__(self, video_path: str, target_size=(1920, 1080), cache_limit=100):
        """
        Initialize video processor.
        
        Args:
            video_path: Path to video file
            target_size: Target resolution for frames (width, height)
            cache_limit: Maximum number of frames to cache (LRU policy)
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            logger.warning(f"Invalid FPS detected: {self.fps}, using default 30.0")
            self.fps = 30.0
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.target_size = target_size
        self.cache_limit = cache_limit
        
        # LRU caches using OrderedDict
        self.frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.superframe_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        
        logger.info(
            f"Video loaded: {video_path} | "
            f"Resolution: {self.width}x{self.height} | "
            f"FPS: {self.fps:.2f} | "
            f"Frames: {self.total_frames} | "
            f"Duration: {self.total_frames/self.fps:.2f}s"
        )

    def _evict_cache_if_needed(self, cache: OrderedDict):
        """Evict oldest entry from cache if limit exceeded (LRU policy)."""
        while len(cache) >= self.cache_limit:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
            logger.debug(f"Evicted frame {oldest_key} from cache")
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get frame at specified index with caching.
        
        Args:
            frame_idx: Frame index to retrieve
            
        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        # Validate frame index
        if frame_idx < 0 or frame_idx >= self.total_frames:
            logger.warning(f"Frame index {frame_idx} out of range [0, {self.total_frames-1}]")
            return None
        
        # Check cache (and move to end for LRU)
        if frame_idx in self.frame_cache:
            self.frame_cache.move_to_end(frame_idx)
            return self.frame_cache[frame_idx].copy()

        # Read from video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            logger.error(f"Failed to read frame {frame_idx}")
            return None
            
        # Resize to target resolution
        frame = cv2.resize(frame, self.target_size)
        
        # Add to cache with LRU eviction
        self._evict_cache_if_needed(self.frame_cache)
        self.frame_cache[frame_idx] = frame
        
        return frame.copy()

    def create_superframe(self, center_frame_idx: int) -> Optional[np.ndarray]:
        """Create 3-frame RGB superframe: [N-1, N, N+1] → [R, G, B].
        
        Args:
            center_frame_idx: Center frame index for superframe generation
            
        Returns:
            Superframe as numpy array or None if failed
        """
        # Check cache (and move to end for LRU)
        if center_frame_idx in self.superframe_cache:
            self.superframe_cache.move_to_end(center_frame_idx)
            return self.superframe_cache[center_frame_idx].copy()

        # Calculate frame indices with boundary handling
        indices = [
            max(0, center_frame_idx - 1),
            center_frame_idx,
            min(self.total_frames - 1, center_frame_idx + 1)
        ]
        
        # Extract and convert frames to grayscale
        g_frames = []
        for idx in indices:
            frame = self.get_frame(idx)
            if frame is None:
                logger.warning(f"Failed to get frame {idx} for superframe")
                # Use black frame as fallback
                g_frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                g_frames.append(gray)

        # Ensure we have exactly 3 frames
        while len(g_frames) < 3:
            g_frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))

        # Merge grayscale frames into RGB channels
        superframe = cv2.merge(g_frames)  # [H, W, 3] — R=N-1, G=N, B=N+1
        
        # Add to cache with LRU eviction
        self._evict_cache_if_needed(self.superframe_cache)
        self.superframe_cache[center_frame_idx] = superframe
        
        return superframe.copy()

    def get_duration(self) -> float:
        """Get video duration in seconds."""
        return self.total_frames / self.fps
    
    def clear_cache(self):
        """Clear all cached frames and superframes."""
        self.frame_cache.clear()
        self.superframe_cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "frames_cached": len(self.frame_cache),
            "superframes_cached": len(self.superframe_cache),
            "cache_limit": self.cache_limit,
            "total_frames": self.total_frames
        }
    
    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info(f"Video released: {self.video_path}")
