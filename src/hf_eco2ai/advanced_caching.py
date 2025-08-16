"""Advanced Caching & Storage Engine for Enterprise-Scale Carbon Metrics.

This module implements high-performance caching and storage solutions for carbon
tracking data, including Redis-compatible distributed caching, time-series database
optimization, and intelligent metric aggregation with compression.
"""

import asyncio
import logging
import time
import json
import gzip
import pickle
import hashlib
import threading
import weakref
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque, OrderedDict
from enum import Enum
import statistics
import struct
import mmap
from pathlib import Path
import tempfile
import sqlite3
import zstandard as zstd

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import influxdb_client
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for hierarchical storage."""
    L1_MEMORY = "l1_memory"          # In-memory, fastest access
    L2_LOCAL_SSD = "l2_local_ssd"    # Local SSD cache
    L3_NETWORK = "l3_network"        # Network cache (Redis)
    L4_PERSISTENT = "l4_persistent"   # Persistent storage (DB)
    L5_ARCHIVE = "l5_archive"        # Archive storage (compressed)


class CompressionAlgorithm(Enum):
    """Compression algorithms for data storage."""
    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"           # Zstandard - good balance
    LZ4 = "lz4"             # Fast compression
    BROTLI = "brotli"       # High compression ratio


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"             # Least Recently Used
    LFU = "lfu"             # Least Frequently Used
    TTL = "ttl"             # Time To Live
    ADAPTIVE = "adaptive"   # Machine learning based
    SIZE = "size"           # Size-based eviction
    CARBON = "carbon"       # Carbon-footprint based


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent eviction."""
    
    key: str
    data: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    compression_ratio: float = 1.0
    carbon_footprint: float = 0.0  # Storage carbon cost
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.timestamp
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    @property
    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per second)."""
        age = max(self.age_seconds, 1.0)  # Avoid division by zero
        return self.access_count / age


@dataclass
class CacheStats:
    """Cache performance statistics."""
    
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    
    # Size metrics
    current_size_bytes: int = 0
    max_size_bytes: int = 0
    entries_count: int = 0
    
    # Performance metrics
    avg_access_time_ms: float = 0.0
    avg_storage_time_ms: float = 0.0
    compression_ratio: float = 1.0
    
    # Carbon metrics
    storage_carbon_kg: float = 0.0
    bandwidth_carbon_kg: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class InMemoryCache:
    """High-performance in-memory cache with intelligent eviction."""
    
    def __init__(self,
                 max_size_bytes: int = 1024 * 1024 * 1024,  # 1GB
                 max_entries: int = 1000000,
                 eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
                 default_ttl: Optional[float] = None):
        """Initialize in-memory cache.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
            max_entries: Maximum number of entries
            eviction_policy: Cache eviction policy
            default_ttl: Default TTL for entries (seconds)
        """
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats(max_size_bytes=max_size_bytes)
        
        # Eviction frequency tracking for adaptive policy
        self._access_frequencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info(f"In-memory cache initialized: {max_size_bytes // (1024*1024)}MB, {eviction_policy.value} policy")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            self.stats.total_requests += 1
            
            if key not in self._cache:
                self.stats.cache_misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self.stats.cache_misses += 1
                self.stats.evictions += 1
                self._update_size_stats()
                return None
            
            # Update access metadata
            entry.last_access = time.time()
            entry.access_count += 1
            
            # Move to end for LRU
            self._cache.move_to_end(key)
            
            # Track access frequency
            self._access_frequencies[key].append(time.time())
            
            self.stats.cache_hits += 1
            return entry.data
    
    def put(self, 
            key: str, 
            data: Any, 
            ttl: Optional[float] = None,
            metadata: Dict[str, Any] = None) -> bool:
        """Put value in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
            metadata: Additional metadata
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            # Calculate data size
            try:
                data_bytes = len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                data_bytes = 1024  # Fallback estimate
            
            # Check if single entry exceeds cache size
            if data_bytes > self.max_size_bytes:
                logger.warning(f"Entry too large for cache: {data_bytes} bytes")
                return False
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=time.time(),
                size_bytes=data_bytes,
                ttl=ttl or self.default_ttl,
                metadata=metadata or {}
            )
            
            # Evict if necessary
            while (self.stats.current_size_bytes + data_bytes > self.max_size_bytes or
                   self.stats.entries_count >= self.max_entries):
                if not self._evict_entry():
                    return False  # Cannot evict enough space
            
            # Store entry
            self._cache[key] = entry
            self.stats.current_size_bytes += data_bytes
            self.stats.entries_count += 1
            
            # Move to end for LRU
            self._cache.move_to_end(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self.stats.current_size_bytes -= entry.size_bytes
                self.stats.entries_count -= 1
                
                # Clean up frequency tracking
                if key in self._access_frequencies:
                    del self._access_frequencies[key]
                
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_frequencies.clear()
            self.stats.current_size_bytes = 0
            self.stats.entries_count = 0
    
    def _evict_entry(self) -> bool:
        """Evict one entry based on eviction policy.
        
        Returns:
            True if an entry was evicted
        """
        if not self._cache:
            return False
        
        if self.eviction_policy == EvictionPolicy.LRU:
            victim_key = next(iter(self._cache))  # First item (oldest)
        elif self.eviction_policy == EvictionPolicy.LFU:
            victim_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].access_count)
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Find entries that are expired or closest to expiration
            now = time.time()
            victims = [(k, e) for k, e in self._cache.items() 
                      if e.ttl and (now - e.timestamp) > e.ttl * 0.9]
            if victims:
                victim_key = victims[0][0]
            else:
                victim_key = next(iter(self._cache))  # Fallback to LRU
        elif self.eviction_policy == EvictionPolicy.SIZE:
            # Evict largest entry
            victim_key = max(self._cache.keys(), 
                           key=lambda k: self._cache[k].size_bytes)
        elif self.eviction_policy == EvictionPolicy.CARBON:
            # Evict entry with highest carbon footprint
            victim_key = max(self._cache.keys(), 
                           key=lambda k: self._cache[k].carbon_footprint)
        else:  # ADAPTIVE
            victim_key = self._adaptive_eviction()
        
        # Remove victim
        entry = self._cache[victim_key]
        del self._cache[victim_key]
        self.stats.current_size_bytes -= entry.size_bytes
        self.stats.entries_count -= 1
        self.stats.evictions += 1
        
        # Clean up frequency tracking
        if victim_key in self._access_frequencies:
            del self._access_frequencies[victim_key]
        
        return True
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction using multiple criteria."""
        now = time.time()
        scores = {}
        
        for key, entry in self._cache.items():
            score = 0.0
            
            # Age factor (older = higher score = more likely to evict)
            age_factor = entry.age_seconds / 3600.0  # Normalize to hours
            score += age_factor * 20
            
            # Frequency factor (less frequent = higher score)
            freq_factor = 1.0 / max(entry.access_frequency, 0.01)
            score += freq_factor * 30
            
            # Size factor (larger = higher score)
            size_factor = entry.size_bytes / (1024 * 1024)  # Normalize to MB
            score += size_factor * 10
            
            # Carbon factor (higher carbon = higher score)
            carbon_factor = entry.carbon_footprint
            score += carbon_factor * 15
            
            # TTL factor (closer to expiration = higher score)
            if entry.ttl:
                ttl_remaining = entry.ttl - entry.age_seconds
                ttl_factor = max(0, 1.0 - (ttl_remaining / entry.ttl))
                score += ttl_factor * 25
            
            scores[key] = score
        
        # Return key with highest score
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _update_size_stats(self):
        """Update size statistics."""
        with self._lock:
            self.stats.current_size_bytes = sum(entry.size_bytes for entry in self._cache.values())
            self.stats.entries_count = len(self._cache)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._update_size_stats()
            return self.stats


class RedisDistributedCache:
    """Redis-compatible distributed cache for carbon metrics."""
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 key_prefix: str = "hf_eco2ai:",
                 compression: CompressionAlgorithm = CompressionAlgorithm.ZSTD,
                 default_ttl: int = 3600):
        """Initialize Redis distributed cache.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys
            compression: Compression algorithm
            default_ttl: Default TTL in seconds
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.compression = compression
        self.default_ttl = default_ttl
        
        # Redis connections
        self._redis_sync: Optional[redis.Redis] = None
        self._redis_async: Optional[aioredis.Redis] = None
        
        # Compression
        self._compressor = self._get_compressor()
        
        # Statistics
        self.stats = CacheStats()
        
        # Initialize connections
        self._initialize_connections()
        
        logger.info(f"Redis distributed cache initialized: {redis_url}")
    
    def _initialize_connections(self):
        """Initialize Redis connections."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, distributed cache disabled")
            return
        
        try:
            # Synchronous connection
            self._redis_sync = redis.from_url(
                self.redis_url,
                decode_responses=False,  # We handle binary data
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            
            # Test connection
            self._redis_sync.ping()
            
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self._redis_sync = None
    
    async def _get_async_redis(self) -> Optional[aioredis.Redis]:
        """Get async Redis connection."""
        if not REDIS_AVAILABLE:
            return None
        
        if self._redis_async is None:
            try:
                self._redis_async = aioredis.from_url(
                    self.redis_url,
                    decode_responses=False,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0
                )
                
                # Test connection
                await self._redis_async.ping()
                
            except Exception as e:
                logger.error(f"Async Redis connection failed: {e}")
                self._redis_async = None
        
        return self._redis_async
    
    def _get_compressor(self):
        """Get compression/decompression functions."""
        if self.compression == CompressionAlgorithm.GZIP:
            return {
                'compress': lambda data: gzip.compress(pickle.dumps(data)),
                'decompress': lambda data: pickle.loads(gzip.decompress(data))
            }
        elif self.compression == CompressionAlgorithm.ZSTD:
            compressor = zstd.ZstdCompressor(level=3)
            decompressor = zstd.ZstdDecompressor()
            return {
                'compress': lambda data: compressor.compress(pickle.dumps(data)),
                'decompress': lambda data: pickle.loads(decompressor.decompress(data))
            }
        elif self.compression == CompressionAlgorithm.LZ4 and LZ4_AVAILABLE:
            return {
                'compress': lambda data: lz4.frame.compress(pickle.dumps(data)),
                'decompress': lambda data: pickle.loads(lz4.frame.decompress(data))
            }
        else:  # No compression
            return {
                'compress': lambda data: pickle.dumps(data),
                'decompress': lambda data: pickle.loads(data)
            }
    
    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self._redis_sync:
            return None
        
        try:
            self.stats.total_requests += 1
            
            redis_key = self._make_key(key)
            compressed_data = self._redis_sync.get(redis_key)
            
            if compressed_data is None:
                self.stats.cache_misses += 1
                return None
            
            # Decompress and deserialize
            data = self._compressor['decompress'](compressed_data)
            self.stats.cache_hits += 1
            
            return data
            
        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            self.stats.cache_misses += 1
            return None
    
    async def aget(self, key: str) -> Optional[Any]:
        """Async get value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        redis_conn = await self._get_async_redis()
        if not redis_conn:
            return None
        
        try:
            self.stats.total_requests += 1
            
            redis_key = self._make_key(key)
            compressed_data = await redis_conn.get(redis_key)
            
            if compressed_data is None:
                self.stats.cache_misses += 1
                return None
            
            # Decompress and deserialize
            data = self._compressor['decompress'](compressed_data)
            self.stats.cache_hits += 1
            
            return data
            
        except Exception as e:
            logger.error(f"Async Redis get failed for key {key}: {e}")
            self.stats.cache_misses += 1
            return None
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Put value in Redis cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        if not self._redis_sync:
            return False
        
        try:
            # Compress and serialize
            compressed_data = self._compressor['compress'](data)
            
            redis_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            
            # Store in Redis with TTL
            result = self._redis_sync.setex(redis_key, ttl, compressed_data)
            
            if result:
                self.stats.current_size_bytes += len(compressed_data)
                self.stats.entries_count += 1
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis put failed for key {key}: {e}")
            return False
    
    async def aput(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Async put value in Redis cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        redis_conn = await self._get_async_redis()
        if not redis_conn:
            return False
        
        try:
            # Compress and serialize
            compressed_data = self._compressor['compress'](data)
            
            redis_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            
            # Store in Redis with TTL
            result = await redis_conn.setex(redis_key, ttl, compressed_data)
            
            if result:
                self.stats.current_size_bytes += len(compressed_data)
                self.stats.entries_count += 1
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Async Redis put failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        if not self._redis_sync:
            return False
        
        try:
            redis_key = self._make_key(key)
            result = self._redis_sync.delete(redis_key)
            
            if result > 0:
                self.stats.entries_count -= result
            
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern.
        
        Args:
            pattern: Key pattern (with wildcards)
            
        Returns:
            Number of keys deleted
        """
        if not self._redis_sync:
            return 0
        
        try:
            redis_pattern = self._make_key(pattern)
            keys = self._redis_sync.keys(redis_pattern)
            
            if keys:
                result = self._redis_sync.delete(*keys)
                self.stats.entries_count -= result
                return result
            
            return 0
            
        except Exception as e:
            logger.error(f"Redis clear pattern failed for {pattern}: {e}")
            return 0


class TimeSeriesOptimizer:
    """Time-series database optimization for carbon metrics."""
    
    def __init__(self,
                 database_url: str = None,
                 retention_policy_days: int = 365,
                 aggregation_intervals: List[str] = None):
        """Initialize time-series optimizer.
        
        Args:
            database_url: Database connection URL
            retention_policy_days: Data retention period
            aggregation_intervals: Aggregation intervals for downsampling
        """
        self.database_url = database_url
        self.retention_policy_days = retention_policy_days
        self.aggregation_intervals = aggregation_intervals or ["5m", "1h", "1d", "1w"]
        
        # Database connections
        self._influxdb_client: Optional[InfluxDBClient] = None
        self._postgres_conn = None
        
        # Aggregation cache
        self._aggregation_cache: Dict[str, Any] = {}
        
        # Initialize connections
        self._initialize_databases()
        
        logger.info("Time-series optimizer initialized")
    
    def _initialize_databases(self):
        """Initialize database connections."""
        # InfluxDB for time-series data
        if INFLUXDB_AVAILABLE and self.database_url and "influx" in self.database_url:
            try:
                self._influxdb_client = InfluxDBClient.from_env_properties()
                logger.info("InfluxDB connection established")
            except Exception as e:
                logger.warning(f"InfluxDB connection failed: {e}")
        
        # PostgreSQL with TimescaleDB for hybrid approach
        if POSTGRES_AVAILABLE and self.database_url and "postgres" in self.database_url:
            try:
                self._postgres_conn = psycopg2.connect(self.database_url)
                self._setup_timescale_tables()
                logger.info("PostgreSQL/TimescaleDB connection established")
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}")
    
    def _setup_timescale_tables(self):
        """Setup TimescaleDB tables for carbon metrics."""
        if not self._postgres_conn:
            return
        
        try:
            with self._postgres_conn.cursor() as cursor:
                # Create carbon metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS carbon_metrics (
                        timestamp TIMESTAMPTZ NOT NULL,
                        cluster_id TEXT NOT NULL,
                        gpu_id INTEGER,
                        metric_type TEXT NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        unit TEXT,
                        metadata JSONB,
                        PRIMARY KEY (timestamp, cluster_id, gpu_id, metric_type)
                    )
                """)
                
                # Create hypertable for time-series optimization
                cursor.execute("""
                    SELECT create_hypertable('carbon_metrics', 'timestamp', 
                                            chunk_time_interval => INTERVAL '1 hour',
                                            if_not_exists => TRUE)
                """)
                
                # Create aggregated metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS carbon_metrics_aggregated (
                        time_bucket TIMESTAMPTZ NOT NULL,
                        cluster_id TEXT NOT NULL,
                        interval_type TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        avg_value DOUBLE PRECISION,
                        min_value DOUBLE PRECISION,
                        max_value DOUBLE PRECISION,
                        sum_value DOUBLE PRECISION,
                        count_value BIGINT,
                        PRIMARY KEY (time_bucket, cluster_id, interval_type, metric_type)
                    )
                """)
                
                self._postgres_conn.commit()
                logger.info("TimescaleDB tables created successfully")
                
        except Exception as e:
            logger.error(f"TimescaleDB setup failed: {e}")
            self._postgres_conn.rollback()
    
    async def store_metrics_batch(self, metrics: List[Dict[str, Any]]) -> bool:
        """Store batch of metrics with optimization.
        
        Args:
            metrics: List of carbon metrics
            
        Returns:
            True if successfully stored
        """
        if not metrics:
            return True
        
        # Parallel storage to multiple backends
        tasks = []
        
        if self._influxdb_client:
            tasks.append(self._store_to_influxdb(metrics))
        
        if self._postgres_conn:
            tasks.append(self._store_to_timescale(metrics))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return any(isinstance(r, bool) and r for r in results)
        
        return False
    
    async def _store_to_influxdb(self, metrics: List[Dict[str, Any]]) -> bool:
        """Store metrics to InfluxDB.
        
        Args:
            metrics: List of carbon metrics
            
        Returns:
            True if successful
        """
        try:
            write_api = self._influxdb_client.write_api(write_options=SYNCHRONOUS)
            points = []
            
            for metric in metrics:
                point = (
                    Point("carbon_metrics")
                    .tag("cluster_id", metric.get("cluster_id", "unknown"))
                    .tag("gpu_id", str(metric.get("gpu_id", 0)))
                    .tag("metric_type", metric.get("metric_type", "unknown"))
                    .field("value", float(metric.get("value", 0)))
                    .time(metric.get("timestamp", time.time()), WritePrecision.S)
                )
                
                # Add metadata as tags
                if "metadata" in metric:
                    for key, value in metric["metadata"].items():
                        if isinstance(value, (str, int, float, bool)):
                            point = point.tag(key, str(value))
                
                points.append(point)
            
            # Write batch
            write_api.write(bucket="carbon-metrics", record=points)
            return True
            
        except Exception as e:
            logger.error(f"InfluxDB write failed: {e}")
            return False
    
    async def _store_to_timescale(self, metrics: List[Dict[str, Any]]) -> bool:
        """Store metrics to TimescaleDB.
        
        Args:
            metrics: List of carbon metrics
            
        Returns:
            True if successful
        """
        try:
            with self._postgres_conn.cursor() as cursor:
                # Prepare batch insert
                insert_query = """
                    INSERT INTO carbon_metrics 
                    (timestamp, cluster_id, gpu_id, metric_type, value, unit, metadata)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """
                
                values = []
                for metric in metrics:
                    values.append((
                        datetime.fromtimestamp(metric.get("timestamp", time.time())),
                        metric.get("cluster_id", "unknown"),
                        metric.get("gpu_id", 0),
                        metric.get("metric_type", "unknown"),
                        float(metric.get("value", 0)),
                        metric.get("unit", ""),
                        json.dumps(metric.get("metadata", {}))
                    ))
                
                # Execute batch insert
                psycopg2.extras.execute_values(cursor, insert_query, values, page_size=1000)
                self._postgres_conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"TimescaleDB write failed: {e}")
            self._postgres_conn.rollback()
            return False
    
    async def aggregate_metrics(self, 
                              start_time: datetime,
                              end_time: datetime,
                              interval: str = "1h") -> Dict[str, Any]:
        """Aggregate metrics for specified time range.
        
        Args:
            start_time: Start time for aggregation
            end_time: End time for aggregation
            interval: Aggregation interval
            
        Returns:
            Aggregated metrics
        """
        cache_key = f"agg_{start_time}_{end_time}_{interval}"
        
        # Check cache first
        if cache_key in self._aggregation_cache:
            cached_result = self._aggregation_cache[cache_key]
            if time.time() - cached_result["timestamp"] < 300:  # 5 minutes
                return cached_result["data"]
        
        # Aggregate from database
        if self._postgres_conn:
            result = await self._aggregate_from_timescale(start_time, end_time, interval)
        elif self._influxdb_client:
            result = await self._aggregate_from_influxdb(start_time, end_time, interval)
        else:
            result = {"error": "No time-series database available"}
        
        # Cache result
        self._aggregation_cache[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        return result
    
    async def _aggregate_from_timescale(self,
                                      start_time: datetime,
                                      end_time: datetime,
                                      interval: str) -> Dict[str, Any]:
        """Aggregate metrics from TimescaleDB.
        
        Args:
            start_time: Start time
            end_time: End time
            interval: Aggregation interval
            
        Returns:
            Aggregated data
        """
        try:
            with self._postgres_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                query = f"""
                    SELECT 
                        time_bucket('{interval}', timestamp) AS time_bucket,
                        cluster_id,
                        metric_type,
                        AVG(value) as avg_value,
                        MIN(value) as min_value,
                        MAX(value) as max_value,
                        SUM(value) as sum_value,
                        COUNT(*) as count_value
                    FROM carbon_metrics 
                    WHERE timestamp >= %s AND timestamp <= %s
                    GROUP BY time_bucket, cluster_id, metric_type
                    ORDER BY time_bucket
                """
                
                cursor.execute(query, (start_time, end_time))
                rows = cursor.fetchall()
                
                # Group by time bucket
                result = defaultdict(list)
                for row in rows:
                    time_bucket = row["time_bucket"].isoformat()
                    result[time_bucket].append(dict(row))
                
                return {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "interval": interval,
                    "data": dict(result),
                    "total_points": len(rows)
                }
                
        except Exception as e:
            logger.error(f"TimescaleDB aggregation failed: {e}")
            return {"error": str(e)}
    
    async def _aggregate_from_influxdb(self,
                                     start_time: datetime,
                                     end_time: datetime,
                                     interval: str) -> Dict[str, Any]:
        """Aggregate metrics from InfluxDB.
        
        Args:
            start_time: Start time
            end_time: End time
            interval: Aggregation interval
            
        Returns:
            Aggregated data
        """
        try:
            query_api = self._influxdb_client.query_api()
            
            flux_query = f"""
                from(bucket: "carbon-metrics")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "carbon_metrics")
                |> group(columns: ["cluster_id", "metric_type"])
                |> aggregateWindow(every: {interval}, fn: mean)
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            """
            
            tables = query_api.query(flux_query)
            
            result = []
            for table in tables:
                for record in table.records:
                    result.append({
                        "time": record.get_time().isoformat(),
                        "cluster_id": record.values.get("cluster_id"),
                        "metric_type": record.values.get("metric_type"),
                        "value": record.get_value()
                    })
            
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "interval": interval,
                "data": result,
                "total_points": len(result)
            }
            
        except Exception as e:
            logger.error(f"InfluxDB aggregation failed: {e}")
            return {"error": str(e)}


class HierarchicalStorageManager:
    """Hierarchical storage management for carbon metrics."""
    
    def __init__(self,
                 storage_tiers: Dict[CacheLevel, Dict[str, Any]] = None):
        """Initialize hierarchical storage manager.
        
        Args:
            storage_tiers: Configuration for each storage tier
        """
        self.storage_tiers = storage_tiers or self._default_storage_config()
        
        # Storage tier implementations
        self.tiers: Dict[CacheLevel, Any] = {}
        
        # Access patterns for optimization
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Statistics
        self.tier_stats: Dict[CacheLevel, CacheStats] = {}
        
        self._initialize_tiers()
        
        logger.info("Hierarchical storage manager initialized")
    
    def _default_storage_config(self) -> Dict[CacheLevel, Dict[str, Any]]:
        """Get default storage tier configuration."""
        return {
            CacheLevel.L1_MEMORY: {
                "enabled": True,
                "max_size_mb": 512,
                "max_entries": 100000,
                "ttl_seconds": 300
            },
            CacheLevel.L2_LOCAL_SSD: {
                "enabled": True,
                "max_size_mb": 4096,
                "cache_dir": "/tmp/hf_eco2ai_cache",
                "ttl_seconds": 3600
            },
            CacheLevel.L3_NETWORK: {
                "enabled": REDIS_AVAILABLE,
                "redis_url": "redis://localhost:6379",
                "ttl_seconds": 7200
            },
            CacheLevel.L4_PERSISTENT: {
                "enabled": True,
                "database_url": "sqlite:///carbon_metrics.db",
                "retention_days": 30
            },
            CacheLevel.L5_ARCHIVE: {
                "enabled": True,
                "archive_dir": "/var/lib/hf_eco2ai/archive",
                "compression": CompressionAlgorithm.ZSTD,
                "retention_years": 7
            }
        }
    
    def _initialize_tiers(self):
        """Initialize storage tiers."""
        # L1: In-memory cache
        if self.storage_tiers[CacheLevel.L1_MEMORY]["enabled"]:
            config = self.storage_tiers[CacheLevel.L1_MEMORY]
            self.tiers[CacheLevel.L1_MEMORY] = InMemoryCache(
                max_size_bytes=config["max_size_mb"] * 1024 * 1024,
                max_entries=config["max_entries"],
                default_ttl=config["ttl_seconds"]
            )
        
        # L2: Local SSD cache
        if self.storage_tiers[CacheLevel.L2_LOCAL_SSD]["enabled"]:
            self.tiers[CacheLevel.L2_LOCAL_SSD] = LocalSSDCache(
                self.storage_tiers[CacheLevel.L2_LOCAL_SSD]
            )
        
        # L3: Network cache (Redis)
        if self.storage_tiers[CacheLevel.L3_NETWORK]["enabled"]:
            config = self.storage_tiers[CacheLevel.L3_NETWORK]
            self.tiers[CacheLevel.L3_NETWORK] = RedisDistributedCache(
                redis_url=config["redis_url"],
                default_ttl=config["ttl_seconds"]
            )
        
        # L4: Persistent database
        if self.storage_tiers[CacheLevel.L4_PERSISTENT]["enabled"]:
            self.tiers[CacheLevel.L4_PERSISTENT] = TimeSeriesOptimizer(
                database_url=self.storage_tiers[CacheLevel.L4_PERSISTENT]["database_url"]
            )
        
        # L5: Archive storage
        if self.storage_tiers[CacheLevel.L5_ARCHIVE]["enabled"]:
            self.tiers[CacheLevel.L5_ARCHIVE] = ArchiveStorage(
                self.storage_tiers[CacheLevel.L5_ARCHIVE]
            )
        
        # Initialize statistics
        for tier in self.tiers:
            self.tier_stats[tier] = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from hierarchical storage.
        
        Args:
            key: Storage key
            
        Returns:
            Cached value or None
        """
        # Record access pattern
        self.access_patterns[key].append(time.time())
        
        # Try each tier in order
        for tier_level in [CacheLevel.L1_MEMORY, CacheLevel.L2_LOCAL_SSD, 
                          CacheLevel.L3_NETWORK, CacheLevel.L4_PERSISTENT]:
            
            if tier_level not in self.tiers:
                continue
            
            tier = self.tiers[tier_level]
            
            try:
                if hasattr(tier, 'aget'):
                    value = await tier.aget(key)
                else:
                    value = tier.get(key)
                
                if value is not None:
                    # Cache hit - promote to higher tiers
                    await self._promote_to_higher_tiers(key, value, tier_level)
                    
                    # Update statistics
                    if tier_level in self.tier_stats:
                        self.tier_stats[tier_level].cache_hits += 1
                        self.tier_stats[tier_level].total_requests += 1
                    
                    return value
                else:
                    # Cache miss
                    if tier_level in self.tier_stats:
                        self.tier_stats[tier_level].cache_misses += 1
                        self.tier_stats[tier_level].total_requests += 1
                
            except Exception as e:
                logger.warning(f"Error accessing tier {tier_level.value}: {e}")
                continue
        
        return None
    
    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in hierarchical storage.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Time to live in seconds
            
        Returns:
            True if successfully stored
        """
        # Determine storage strategy based on access patterns
        storage_strategy = self._determine_storage_strategy(key)
        
        success = False
        
        # Store to appropriate tiers based on strategy
        for tier_level in storage_strategy:
            if tier_level not in self.tiers:
                continue
            
            tier = self.tiers[tier_level]
            
            try:
                if hasattr(tier, 'aput'):
                    tier_success = await tier.aput(key, value, ttl)
                else:
                    tier_success = tier.put(key, value, ttl)
                
                if tier_success:
                    success = True
                    
                    # Update statistics
                    if tier_level in self.tier_stats:
                        self.tier_stats[tier_level].entries_count += 1
                
            except Exception as e:
                logger.warning(f"Error storing to tier {tier_level.value}: {e}")
                continue
        
        return success
    
    def _determine_storage_strategy(self, key: str) -> List[CacheLevel]:
        """Determine which tiers to store data based on access patterns.
        
        Args:
            key: Storage key
            
        Returns:
            List of cache levels to store data
        """
        # Analyze access pattern
        accesses = self.access_patterns.get(key, deque())
        
        if not accesses:
            # New key - store in all active tiers
            return [tier for tier in self.tiers.keys()]
        
        # Calculate access frequency
        now = time.time()
        recent_accesses = [t for t in accesses if now - t < 3600]  # Last hour
        access_frequency = len(recent_accesses) / 3600.0  # Accesses per second
        
        strategy = []
        
        # Always store in L1 for frequently accessed data
        if access_frequency > 0.01:  # More than once per 100 seconds
            strategy.append(CacheLevel.L1_MEMORY)
        
        # Store in L2 for moderately accessed data
        if access_frequency > 0.001:  # More than once per 1000 seconds
            strategy.append(CacheLevel.L2_LOCAL_SSD)
        
        # Always store in network cache for sharing
        strategy.append(CacheLevel.L3_NETWORK)
        
        # Store in persistent database for long-term retention
        strategy.append(CacheLevel.L4_PERSISTENT)
        
        # Archive old data
        if len(accesses) > 100 and now - accesses[0] > 86400:  # Older than 1 day
            strategy.append(CacheLevel.L5_ARCHIVE)
        
        return strategy
    
    async def _promote_to_higher_tiers(self, 
                                     key: str, 
                                     value: Any, 
                                     source_tier: CacheLevel):
        """Promote data to higher performance tiers.
        
        Args:
            key: Storage key
            value: Value to promote
            source_tier: Source tier where data was found
        """
        higher_tiers = []
        
        if source_tier == CacheLevel.L2_LOCAL_SSD:
            higher_tiers = [CacheLevel.L1_MEMORY]
        elif source_tier == CacheLevel.L3_NETWORK:
            higher_tiers = [CacheLevel.L1_MEMORY, CacheLevel.L2_LOCAL_SSD]
        elif source_tier == CacheLevel.L4_PERSISTENT:
            higher_tiers = [CacheLevel.L1_MEMORY, CacheLevel.L2_LOCAL_SSD, CacheLevel.L3_NETWORK]
        
        for tier_level in higher_tiers:
            if tier_level in self.tiers:
                try:
                    tier = self.tiers[tier_level]
                    if hasattr(tier, 'aput'):
                        await tier.aput(key, value)
                    else:
                        tier.put(key, value)
                except Exception as e:
                    logger.warning(f"Error promoting to tier {tier_level.value}: {e}")
    
    async def cleanup_expired(self):
        """Cleanup expired data from all tiers."""
        for tier_level, tier in self.tiers.items():
            try:
                if hasattr(tier, 'cleanup_expired'):
                    await tier.cleanup_expired()
                logger.debug(f"Cleaned up expired data in tier {tier_level.value}")
            except Exception as e:
                logger.warning(f"Error cleaning tier {tier_level.value}: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics.
        
        Returns:
            Storage statistics for all tiers
        """
        stats = {}
        
        for tier_level, tier in self.tiers.items():
            try:
                if hasattr(tier, 'get_stats'):
                    tier_stats = tier.get_stats()
                else:
                    tier_stats = self.tier_stats.get(tier_level, CacheStats())
                
                stats[tier_level.value] = asdict(tier_stats)
                
            except Exception as e:
                logger.warning(f"Error getting stats for tier {tier_level.value}: {e}")
                stats[tier_level.value] = {"error": str(e)}
        
        # Overall statistics
        total_requests = sum(
            s.get("total_requests", 0) for s in stats.values() 
            if isinstance(s, dict) and "error" not in s
        )
        
        total_hits = sum(
            s.get("cache_hits", 0) for s in stats.values()
            if isinstance(s, dict) and "error" not in s
        )
        
        stats["overall"] = {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "overall_hit_rate": total_hits / max(total_requests, 1),
            "active_tiers": len([s for s in stats.values() if isinstance(s, dict) and "error" not in s]),
            "access_patterns_tracked": len(self.access_patterns)
        }
        
        return stats


class LocalSSDCache:
    """Local SSD cache implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize local SSD cache."""
        self.cache_dir = Path(config["cache_dir"])
        self.max_size_mb = config["max_size_mb"]
        self.ttl_seconds = config["ttl_seconds"]
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Index for quick lookups
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        self._load_index()
        
    def _load_index(self):
        """Load cache index from disk."""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}
    
    def _save_index(self):
        """Save cache index to disk."""
        index_file = self.cache_dir / "index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from SSD cache."""
        with self._lock:
            if key not in self._index:
                return None
            
            entry = self._index[key]
            
            # Check expiration
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                self._remove_entry(key)
                return None
            
            # Load from disk
            file_path = self.cache_dir / entry["filename"]
            if not file_path.exists():
                del self._index[key]
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access time
                entry["last_access"] = time.time()
                entry["access_count"] += 1
                
                return data
                
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
                self._remove_entry(key)
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in SSD cache."""
        with self._lock:
            try:
                # Create filename
                filename = hashlib.md5(key.encode()).hexdigest() + ".pkl"
                file_path = self.cache_dir / filename
                
                # Serialize data
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Update index
                self._index[key] = {
                    "filename": filename,
                    "timestamp": time.time(),
                    "last_access": time.time(),
                    "access_count": 0,
                    "size_bytes": file_path.stat().st_size,
                    "ttl": ttl or self.ttl_seconds
                }
                
                self._save_index()
                return True
                
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
                return False
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self._index:
            entry = self._index[key]
            file_path = self.cache_dir / entry["filename"]
            
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cached file: {e}")
            
            del self._index[key]


class ArchiveStorage:
    """Archive storage for long-term carbon metrics retention."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize archive storage."""
        self.archive_dir = Path(config["archive_dir"])
        self.compression = CompressionAlgorithm(config.get("compression", "zstd"))
        self.retention_years = config["retention_years"]
        
        # Create archive directory
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
    async def store_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """Store metrics in archive with compression."""
        try:
            # Group by date
            daily_groups = defaultdict(list)
            
            for metric in metrics:
                timestamp = metric.get("timestamp", time.time())
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                daily_groups[date_str].append(metric)
            
            # Store each day's data
            for date_str, day_metrics in daily_groups.items():
                archive_file = self.archive_dir / f"carbon_metrics_{date_str}.json.zst"
                
                # Compress and store
                data = json.dumps(day_metrics).encode('utf-8')
                
                if self.compression == CompressionAlgorithm.ZSTD:
                    compressor = zstd.ZstdCompressor(level=9)  # High compression
                    compressed_data = compressor.compress(data)
                else:
                    compressed_data = gzip.compress(data, compresslevel=9)
                
                with open(archive_file, 'wb') as f:
                    f.write(compressed_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Archive storage failed: {e}")
            return False


# Global cache instances
_memory_cache: Optional[InMemoryCache] = None
_distributed_cache: Optional[RedisDistributedCache] = None
_hierarchical_storage: Optional[HierarchicalStorageManager] = None


def get_memory_cache(**kwargs) -> InMemoryCache:
    """Get global in-memory cache instance."""
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = InMemoryCache(**kwargs)
    return _memory_cache


def get_distributed_cache(**kwargs) -> RedisDistributedCache:
    """Get global distributed cache instance."""
    global _distributed_cache
    if _distributed_cache is None:
        _distributed_cache = RedisDistributedCache(**kwargs)
    return _distributed_cache


def get_hierarchical_storage(**kwargs) -> HierarchicalStorageManager:
    """Get global hierarchical storage instance."""
    global _hierarchical_storage
    if _hierarchical_storage is None:
        _hierarchical_storage = HierarchicalStorageManager(**kwargs)
    return _hierarchical_storage


async def initialize_advanced_caching():
    """Initialize advanced caching with optimal configuration."""
    # Initialize all cache layers
    memory_cache = get_memory_cache(
        max_size_bytes=1024 * 1024 * 1024,  # 1GB
        eviction_policy=EvictionPolicy.ADAPTIVE
    )
    
    if REDIS_AVAILABLE:
        distributed_cache = get_distributed_cache(
            compression=CompressionAlgorithm.ZSTD
        )
    
    hierarchical_storage = get_hierarchical_storage()
    
    logger.info("Advanced caching system initialized")
    
    return {
        "memory_cache": memory_cache,
        "distributed_cache": _distributed_cache,
        "hierarchical_storage": hierarchical_storage
    }