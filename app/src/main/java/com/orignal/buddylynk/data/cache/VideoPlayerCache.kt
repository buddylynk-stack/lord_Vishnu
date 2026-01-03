package com.orignal.buddylynk.data.cache

import android.content.Context
import androidx.media3.common.util.UnstableApi
import androidx.media3.database.StandaloneDatabaseProvider
import androidx.media3.datasource.DataSource
import androidx.media3.datasource.DefaultDataSource
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.datasource.cache.CacheDataSource
import androidx.media3.datasource.cache.LeastRecentlyUsedCacheEvictor
import androidx.media3.datasource.cache.SimpleCache
import java.io.File

/**
 * VideoPlayerCache - Global video cache manager
 * 
 * Provides disk caching for video playback using ExoPlayer's SimpleCache.
 * Videos are cached to disk (100MB) so they don't re-download when scrolling.
 * Works like Instagram - videos play instantly from cache on second view.
 */
@UnstableApi
object VideoPlayerCache {
    
    private var cache: SimpleCache? = null
    private var cacheDataSourceFactory: CacheDataSource.Factory? = null
    
    // 100MB cache size (increase for more caching)
    private const val CACHE_SIZE_BYTES = 100L * 1024L * 1024L
    
    /**
     * Get or create the SimpleCache instance
     * Uses LRU eviction when cache is full
     */
    @Synchronized
    fun getCache(context: Context): SimpleCache {
        if (cache == null) {
            val cacheDir = File(context.cacheDir, "video_cache")
            if (!cacheDir.exists()) {
                cacheDir.mkdirs()
            }
            
            val evictor = LeastRecentlyUsedCacheEvictor(CACHE_SIZE_BYTES)
            val databaseProvider = StandaloneDatabaseProvider(context)
            
            cache = SimpleCache(cacheDir, evictor, databaseProvider)
        }
        return cache!!
    }
    
    /**
     * Get CacheDataSource.Factory for creating cached media sources
     * This should be used when building ExoPlayer to enable caching
     */
    @Synchronized
    fun getCacheDataSourceFactory(context: Context): DataSource.Factory {
        if (cacheDataSourceFactory == null) {
            val simpleCache = getCache(context)
            
            // Upstream data source (for downloading from network)
            val httpDataSourceFactory = DefaultHttpDataSource.Factory()
                .setAllowCrossProtocolRedirects(true)
                .setConnectTimeoutMs(15000)
                .setReadTimeoutMs(15000)
            
            val defaultDataSourceFactory = DefaultDataSource.Factory(
                context,
                httpDataSourceFactory
            )
            
            // Cache data source that reads from cache first, then network
            cacheDataSourceFactory = CacheDataSource.Factory()
                .setCache(simpleCache)
                .setUpstreamDataSourceFactory(defaultDataSourceFactory)
                .setCacheWriteDataSinkFactory(null) // Use default write sink
                .setFlags(CacheDataSource.FLAG_IGNORE_CACHE_ON_ERROR)
        }
        return cacheDataSourceFactory!!
    }
    
    /**
     * Release the cache (call when app is terminating)
     */
    @Synchronized
    fun release() {
        cache?.release()
        cache = null
        cacheDataSourceFactory = null
    }
    
    /**
     * Clear all cached videos
     */
    @Synchronized
    fun clearCache(context: Context) {
        try {
            val cacheDir = File(context.cacheDir, "video_cache")
            cacheDir.deleteRecursively()
            cache?.release()
            cache = null
            cacheDataSourceFactory = null
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
