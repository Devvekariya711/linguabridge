"""
LinguaBridge Bounded Queue
===========================
Async queue with max size and drop-oldest policy.
Prevents backpressure from stalling the pipeline.
"""

import asyncio
import logging
from typing import TypeVar, Generic, Optional
from dataclasses import dataclass, field
from collections import deque
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# BOUNDED QUEUE
# =============================================================================

class BoundedQueue(asyncio.Queue, Generic[T]):
    """
    Async queue with maximum size and drop policy.
    
    When full, drops oldest items to make room for new ones.
    """
    
    def __init__(self, maxsize: int = 8, name: str = "queue"):
        super().__init__(maxsize=maxsize)
        self.name = name
        self.dropped_count = 0
        self.total_put = 0
    
    def put_nowait_drop_oldest(self, item: T) -> int:
        """
        Put item, dropping oldest if queue is full.
        
        Returns: number of items dropped
        """
        dropped = 0
        while self.qsize() >= self.maxsize:
            try:
                self.get_nowait()
                dropped += 1
                self.dropped_count += 1
            except asyncio.QueueEmpty:
                break
        
        self.put_nowait(item)
        self.total_put += 1
        
        if dropped > 0:
            logger.warning(f"[{self.name}] Dropped {dropped} items (total: {self.dropped_count})")
        
        return dropped
    
    async def put_drop_oldest(self, item: T) -> int:
        """Async version of put_nowait_drop_oldest."""
        return self.put_nowait_drop_oldest(item)
    
    def is_overloaded(self, threshold: float = 0.75) -> bool:
        """Check if queue is above threshold capacity."""
        return self.qsize() >= int(self.maxsize * threshold)
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "name": self.name,
            "current_size": self.qsize(),
            "max_size": self.maxsize,
            "total_put": self.total_put,
            "dropped": self.dropped_count,
            "drop_rate": round(self.dropped_count / max(1, self.total_put), 3),
        }


# =============================================================================
# PRIORITY QUEUE ITEM
# =============================================================================

@dataclass(order=True)
class PriorityItem:
    """Item with priority for priority queue."""
    priority: int
    timestamp: float = field(compare=False)
    data: any = field(compare=False)
    
    @classmethod
    def create(cls, priority: int, data: any) -> 'PriorityItem':
        return cls(priority=priority, timestamp=time.time(), data=data)


# =============================================================================
# PIPELINE QUEUES
# =============================================================================

# Default queue sizes
MAX_STT_QUEUE = 10
MAX_NMT_QUEUE = 8
MAX_TTS_QUEUE = 8

# Queue instances (created lazily)
_queues: dict[str, BoundedQueue] = {}


def get_queue(name: str, maxsize: int = 8) -> BoundedQueue:
    """Get or create a named queue."""
    if name not in _queues:
        _queues[name] = BoundedQueue(maxsize=maxsize, name=name)
    return _queues[name]


def get_all_queue_stats() -> dict:
    """Get stats for all queues."""
    return {name: q.get_stats() for name, q in _queues.items()}


# =============================================================================
# BACKPRESSURE SIGNAL
# =============================================================================

@dataclass
class BackpressureSignal:
    """Signal to client about server load."""
    overloaded: bool
    queue_name: str
    queue_size: int
    max_size: int
    recommendation: str  # "slow_down", "pause", "ok"
    
    @classmethod
    def from_queue(cls, queue: BoundedQueue) -> 'BackpressureSignal':
        if queue.is_overloaded(0.9):
            return cls(
                overloaded=True,
                queue_name=queue.name,
                queue_size=queue.qsize(),
                max_size=queue.maxsize,
                recommendation="pause"
            )
        elif queue.is_overloaded(0.75):
            return cls(
                overloaded=True,
                queue_name=queue.name,
                queue_size=queue.qsize(),
                max_size=queue.maxsize,
                recommendation="slow_down"
            )
        else:
            return cls(
                overloaded=False,
                queue_name=queue.name,
                queue_size=queue.qsize(),
                max_size=queue.maxsize,
                recommendation="ok"
            )
