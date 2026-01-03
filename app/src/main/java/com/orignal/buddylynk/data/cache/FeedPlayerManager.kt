package com.orignal.buddylynk.data.cache

import android.content.Context
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory

/**
 * FeedPlayerManager - Manages a pool of ExoPlayer instances for feed videos
 * 
 * Uses LRU eviction to reuse players instead of creating new ones on every scroll.
 * When you scroll back to a video, the same player is reused (already buffered).
 * This eliminates re-buffering when scrolling back to previously viewed videos.
 */
@UnstableApi
object FeedPlayerManager {
    
    private var applicationContext: Context? = null
    
    // LRU cache of players - keeps last 8 players (current visible + 7 cached)
    private val players = object : LinkedHashMap<String, ExoPlayer>(12, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, ExoPlayer>?): Boolean {
            if (size > MAX_PLAYERS) {
                eldest?.value?.release()
                return true
            }
            return false
        }
    }
    
    private const val MAX_PLAYERS = 8 // Keep 8 cached players
    
    // Global mute state - persists across all videos when scrolling
    var isMuted: Boolean = true
        private set
    
    /**
     * Toggle global mute state and apply to all cached players
     */
    fun toggleMute() {
        isMuted = !isMuted
        // Apply to all cached players immediately
        players.values.forEach { player ->
            player.volume = if (isMuted) 0f else 1f
        }
    }
    
    /**
     * Set mute state and apply to all cached players
     */
    fun setMuted(muted: Boolean) {
        isMuted = muted
        players.values.forEach { player ->
            player.volume = if (muted) 0f else 1f
        }
    }
    
    /**
     * Initialize with application context (call from MainActivity or Application)
     */
    fun init(context: Context) {
        applicationContext = context.applicationContext
    }
    
    /**
     * Get or create a player for the given URL
     * If player exists, it's reused (no re-buffering!)
     */
    @Synchronized
    fun getPlayer(url: String, onBufferingChange: (Boolean) -> Unit, onError: () -> Unit): ExoPlayer? {
        val context = applicationContext ?: return null
        
        // Check if player already exists for this URL
        players[url]?.let { existingPlayer ->
            // Reset to start if needed
            if (existingPlayer.currentPosition > 0) {
                existingPlayer.seekTo(0)
            }
            existingPlayer.playWhenReady = true
            
            // IMPORTANT: Update buffering state based on current player state
            // This fixes the loading indicator showing when player is already ready
            val isBuffering = existingPlayer.playbackState == Player.STATE_BUFFERING
            onBufferingChange(isBuffering)
            
            return existingPlayer
        }
        
        // Create new player with cached data source
        val cacheFactory = VideoPlayerCache.getCacheDataSourceFactory(context)
        
        val loadControl = androidx.media3.exoplayer.DefaultLoadControl.Builder()
            .setBufferDurationsMs(
                2000,   // Min buffer
                30000,  // Max buffer  
                500,    // Buffer for playback - start quickly
                1000    // Buffer for rebuffer
            )
            .setPrioritizeTimeOverSizeThresholds(true)
            .build()
        
        val player = ExoPlayer.Builder(context)
            .setLoadControl(loadControl)
            .setMediaSourceFactory(DefaultMediaSourceFactory(cacheFactory))
            .build().apply {
                setMediaItem(MediaItem.fromUri(url))
                repeatMode = Player.REPEAT_MODE_ONE
                volume = 0f // Muted by default
                
                addListener(object : Player.Listener {
                    override fun onPlaybackStateChanged(state: Int) {
                        onBufferingChange(state == Player.STATE_BUFFERING)
                    }
                    
                    override fun onPlayerError(error: androidx.media3.common.PlaybackException) {
                        android.util.Log.e("FeedPlayerManager", "Player error: ${error.message}")
                        onError()
                    }
                })
                
                prepare()
                playWhenReady = true
            }
        
        players[url] = player
        return player
    }
    
    /**
     * Pause player for URL (when scrolled away)
     * DON'T release - keep it cached for when user scrolls back
     */
    @Synchronized
    fun pausePlayer(url: String) {
        players[url]?.let { player ->
            player.playWhenReady = false
        }
    }
    
    /**
     * Resume player for URL (when scrolled back)
     */
    @Synchronized
    fun resumePlayer(url: String) {
        players[url]?.let { player ->
            player.playWhenReady = true
        }
    }
    
    /**
     * Release all players (call when leaving feed screen)
     */
    @Synchronized
    fun releaseAll() {
        players.values.forEach { it.release() }
        players.clear()
    }
    
    /**
     * Release single player
     */
    @Synchronized
    fun release(url: String) {
        players[url]?.release()
        players.remove(url)
    }
    
    /**
     * Check if player exists for URL
     */
    fun hasPlayer(url: String): Boolean = players.containsKey(url)
}
