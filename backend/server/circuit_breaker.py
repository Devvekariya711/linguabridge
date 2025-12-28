"""
LinguaBridge Circuit Breaker
=============================
Protects against LLM failures and timeouts.
Prevents cascading failures in the pipeline.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# CIRCUIT STATES
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking all calls
    HALF_OPEN = "half_open"  # Testing with single call


class CircuitOpenError(Exception):
    """Raised when circuit is open and calls are blocked."""
    pass


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

@dataclass
class CircuitBreaker:
    """
    Circuit breaker for LLM and other unreliable services.
    
    States:
    - CLOSED: Normal, calls pass through
    - OPEN: Blocking all calls (after failures exceed threshold)
    - HALF_OPEN: Testing with one call to see if service recovered
    """
    name: str = "circuit"
    failure_threshold: int = 3
    recovery_timeout: float = 30.0  # seconds
    call_timeout: float = 10.0  # seconds
    
    # State
    state: CircuitState = field(default=CircuitState.CLOSED)
    failures: int = field(default=0)
    successes: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    total_calls: int = field(default=0)
    total_failures: int = field(default=0)
    
    def is_closed(self) -> bool:
        """Check if circuit allows calls."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"[{self.name}] Circuit half-open, testing...")
                return True
            return False
        
        # HALF_OPEN allows one call
        return True
    
    def record_success(self):
        """Record successful call."""
        self.successes += 1
        self.failures = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info(f"[{self.name}] Circuit closed (recovered)")
    
    def record_failure(self, error: Exception = None):
        """Record failed call."""
        self.failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"[{self.name}] Circuit OPEN after {self.failures} failures. "
                f"Recovery in {self.recovery_timeout}s. Error: {error}"
            )
    
    async def call(self, coro) -> Any:
        """
        Execute coroutine through circuit breaker.
        
        Raises:
            CircuitOpenError: If circuit is open
            TimeoutError: If call times out
            Exception: Original exception from coro
        """
        if not self.is_closed():
            raise CircuitOpenError(
                f"Circuit {self.name} is OPEN. "
                f"Recovery in {self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
            )
        
        self.total_calls += 1
        
        try:
            result = await asyncio.wait_for(coro, timeout=self.call_timeout)
            self.record_success()
            return result
            
        except asyncio.TimeoutError:
            self.record_failure(TimeoutError(f"Timeout after {self.call_timeout}s"))
            raise
            
        except Exception as e:
            self.record_failure(e)
            raise
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self.failures,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "failure_rate": round(self.total_failures / max(1, self.total_calls), 3),
        }


# =============================================================================
# LLM CIRCUIT BREAKER
# =============================================================================

# Global LLM circuit breaker
_llm_breaker: Optional[CircuitBreaker] = None

# Timeouts by mode
LLM_TIMEOUT_LIVE = 4.0   # Very short for live mode
LLM_TIMEOUT_FINAL = 20.0  # Longer for final pass


def get_llm_breaker() -> CircuitBreaker:
    """Get or create LLM circuit breaker."""
    global _llm_breaker
    if _llm_breaker is None:
        _llm_breaker = CircuitBreaker(
            name="llm",
            failure_threshold=3,
            recovery_timeout=30.0,
            call_timeout=LLM_TIMEOUT_LIVE,
        )
    return _llm_breaker


async def call_llm_safe(coro, timeout: float = None) -> Any:
    """
    Call LLM through circuit breaker.
    
    Args:
        coro: Coroutine to execute
        timeout: Override timeout (optional)
    
    Returns:
        Result from LLM
        
    Raises:
        CircuitOpenError: If LLM is disabled due to failures
    """
    breaker = get_llm_breaker()
    
    if timeout:
        breaker.call_timeout = timeout
    
    return await breaker.call(coro)


def is_llm_available() -> bool:
    """Check if LLM calls are allowed."""
    return get_llm_breaker().is_closed()


def get_llm_stats() -> dict:
    """Get LLM circuit breaker stats."""
    return get_llm_breaker().get_stats()


# =============================================================================
# GENERIC SAFE CALL
# =============================================================================

async def safe_call(
    coro,
    timeout: float = 5.0,
    fallback: Any = None,
    on_error: Callable[[Exception], Any] = None
) -> Any:
    """
    Execute coroutine with timeout and fallback.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        fallback: Value to return on error
        on_error: Callback for errors
    
    Returns:
        Result or fallback value
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Call timed out after {timeout}s")
        if on_error:
            on_error(TimeoutError())
        return fallback
    except Exception as e:
        logger.error(f"Call failed: {e}")
        if on_error:
            on_error(e)
        return fallback
