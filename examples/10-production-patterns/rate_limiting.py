"""Production Rate Limiting Patterns.

This example demonstrates how to implement robust rate limiting
for LLM API calls in production systems.

Features demonstrated:
- Token bucket rate limiting
- Request queuing
- Graceful degradation
- Rate limit monitoring
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_hub import Message, OpenAIProvider, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    max_queue_size: int = 100
    queue_timeout_seconds: float = 30.0


@dataclass
class RateLimitStats:
    """Statistics for rate limiting."""

    requests_made: int = 0
    requests_queued: int = 0
    requests_dropped: int = 0
    tokens_used: int = 0
    average_wait_ms: float = 0.0
    _wait_times: list[float] = field(default_factory=list)

    def record_wait(self, wait_ms: float) -> None:
        """Record a wait time."""
        self._wait_times.append(wait_ms)
        self.average_wait_ms = sum(self._wait_times) / len(self._wait_times)


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize the rate limiter.

        Args:
            config: Rate limiting configuration.
        """
        self.config = config
        self.stats = RateLimitStats()

        # Request rate limiting
        self.request_tokens = float(config.requests_per_minute)
        self.request_rate = config.requests_per_minute / 60.0  # per second
        self.last_request_update = time.time()

        # Token rate limiting
        self.api_tokens = float(config.tokens_per_minute)
        self.token_rate = config.tokens_per_minute / 60.0  # per second
        self.last_token_update = time.time()

        # Queue for requests
        self.queue: deque[asyncio.Event] = deque()

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def _refill_buckets(self) -> None:
        """Refill token buckets based on elapsed time."""
        now = time.time()

        # Refill request bucket
        elapsed = now - self.last_request_update
        self.request_tokens = min(
            self.config.requests_per_minute,
            self.request_tokens + elapsed * self.request_rate,
        )
        self.last_request_update = now

        # Refill token bucket
        elapsed = now - self.last_token_update
        self.api_tokens = min(
            self.config.tokens_per_minute,
            self.api_tokens + elapsed * self.token_rate,
        )
        self.last_token_update = now

    async def acquire(self, estimated_tokens: int = 1000) -> bool:
        """Acquire permission to make a request.

        Args:
            estimated_tokens: Estimated tokens for the request.

        Returns:
            True if acquired, False if dropped.
        """
        async with self._lock:
            self._refill_buckets()

            # Check if we have capacity
            if self.request_tokens >= 1 and self.api_tokens >= estimated_tokens:
                self.request_tokens -= 1
                self.api_tokens -= estimated_tokens
                self.stats.requests_made += 1
                return True

            # Queue if possible
            if len(self.queue) < self.config.max_queue_size:
                event = asyncio.Event()
                self.queue.append(event)
                self.stats.requests_queued += 1

        # Wait for capacity
        start_wait = time.time()
        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=self.config.queue_timeout_seconds,
            )
            wait_ms = (time.time() - start_wait) * 1000
            self.stats.record_wait(wait_ms)

            async with self._lock:
                self.request_tokens -= 1
                self.api_tokens -= estimated_tokens
                self.stats.requests_made += 1
            return True

        except asyncio.TimeoutError:
            self.stats.requests_dropped += 1
            logger.warning("request_dropped_timeout")
            return False

    async def release_queued(self) -> None:
        """Release a queued request when capacity is available."""
        async with self._lock:
            self._refill_buckets()
            while self.queue and self.request_tokens >= 1:
                event = self.queue.popleft()
                event.set()

    def get_stats(self) -> dict[str, Any]:
        """Get current rate limiting statistics."""
        return {
            "requests_made": self.stats.requests_made,
            "requests_queued": self.stats.requests_queued,
            "requests_dropped": self.stats.requests_dropped,
            "tokens_used": self.stats.tokens_used,
            "average_wait_ms": self.stats.average_wait_ms,
            "current_request_tokens": self.request_tokens,
            "current_api_tokens": self.api_tokens,
        }


class RateLimitedProvider:
    """LLM provider with built-in rate limiting."""

    def __init__(
        self,
        config: RateLimitConfig | None = None,
    ) -> None:
        """Initialize the rate-limited provider.

        Args:
            config: Rate limiting configuration.
        """
        self.config = config or RateLimitConfig()
        self.rate_limiter = TokenBucketRateLimiter(self.config)
        self.provider = OpenAIProvider(default_model="gpt-4o-mini")

    async def complete(
        self,
        messages: list[Message],
        estimated_tokens: int = 1000,
        **kwargs: Any,
    ) -> str | None:
        """Complete with rate limiting.

        Args:
            messages: Conversation messages.
            estimated_tokens: Estimated request tokens.
            **kwargs: Additional arguments for the provider.

        Returns:
            Response content or None if rate limited.
        """
        if not await self.rate_limiter.acquire(estimated_tokens):
            logger.warning("request_rate_limited")
            return None

        try:
            response = self.provider.complete(messages, **kwargs)

            # Update actual token usage
            if response.usage:
                self.rate_limiter.stats.tokens_used += response.usage.total_tokens

            return response.content

        finally:
            # Release queued requests
            await self.rate_limiter.release_queued()

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiting statistics."""
        return self.rate_limiter.get_stats()


async def simulate_burst_traffic() -> None:
    """Simulate burst traffic to demonstrate rate limiting."""
    print("\n--- Simulating Burst Traffic ---")

    # Configure aggressive rate limiting for demo
    config = RateLimitConfig(
        requests_per_minute=10,  # Low limit for demo
        tokens_per_minute=10000,
        max_queue_size=20,
        queue_timeout_seconds=5.0,
    )

    provider = RateLimitedProvider(config)

    # Simulate 20 concurrent requests
    async def make_request(i: int) -> tuple[int, bool]:
        result = await provider.complete(
            [Message.user(f"Say 'Response {i}' in exactly those words.")],
            estimated_tokens=50,
            max_tokens=20,
        )
        return i, result is not None

    print("Sending 20 concurrent requests with limit of 10/minute...")
    start = time.time()

    tasks = [make_request(i) for i in range(20)]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start

    # Analyze results
    successful = sum(1 for _, success in results if success)
    failed = sum(1 for _, success in results if not success)

    print(f"\nResults (elapsed: {elapsed:.1f}s):")
    print(f"  Successful: {successful}")
    print(f"  Rate limited: {failed}")

    stats = provider.get_stats()
    print("\nRate Limiter Stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    """Run the rate limiting examples."""
    print("=" * 60)
    print("Production Rate Limiting Patterns")
    print("=" * 60)

    print("\n--- Rate Limiting Strategies ---")
    print(
        """
1. TOKEN BUCKET
   - Smooth rate limiting with burst capability
   - Configurable bucket size and refill rate
   - Best for: General API rate limiting

2. SLIDING WINDOW
   - More precise rate tracking
   - Memory-efficient with windowed counting
   - Best for: Strict rate enforcement

3. ADAPTIVE RATE LIMITING
   - Adjusts based on 429 responses
   - Learns optimal rate from API feedback
   - Best for: Unknown or changing limits

4. PRIORITY QUEUING
   - Different priority levels for requests
   - Critical requests bypass queue
   - Best for: Mixed workload importance
"""
    )

    # Run simulation
    asyncio.run(simulate_burst_traffic())

    print("\n--- Best Practices ---")
    print(
        """
1. Always implement client-side rate limiting (don't rely only on API limits)
2. Use exponential backoff for retries
3. Monitor rate limit headers in responses
4. Implement request queuing with timeouts
5. Have graceful degradation strategies
6. Log rate limiting events for observability
"""
    )


if __name__ == "__main__":
    main()
