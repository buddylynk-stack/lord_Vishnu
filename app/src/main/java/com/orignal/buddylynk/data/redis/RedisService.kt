package com.orignal.buddylynk.data.redis

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

/**
 * Redis Configuration for Production
 * 
 * HOW TO GET UPSTASH REDIS (FREE TIER):
 * 1. Go to https://upstash.com
 * 2. Sign up / Login
 * 3. Create new Redis database
 * 4. Copy your REST URL and REST Token
 * 5. Paste below
 */
object RedisConfig {
    // ============================================
    // 🔐 YOUR UPSTASH CREDENTIALS (PASTE HERE!)
    // ============================================
    
    // ============================================
    // 🔐 LOCAL SERVER CONFIG (Unified Backend)
    // ============================================
    
    // User's PC IP (for Mobile testing)
    const val UPSTASH_REDIS_URL = "http://192.168.0.102:8080"
    
    // Token is ignored by local server, but kept for compatibility
    const val UPSTASH_REDIS_TOKEN = "dummy_local_token"
    
    // ============================================
    
    // Use local cache as fallback if Upstash fails
    const val USE_LOCAL_FALLBACK = true
    
    // Check if Upstash is configured
    fun isUpstashConfigured(): Boolean {
        return !UPSTASH_REDIS_URL.contains("YOUR-UPSTASH") &&
               !UPSTASH_REDIS_TOKEN.contains("YOUR-UPSTASH")
    }
    
    // Redis keys prefixes
    const val PREFIX_ONLINE = "online:"
    const val PREFIX_VIEWS = "views:"
    const val PREFIX_LIKES = "likes:"
    const val PREFIX_SHARES = "shares:"
    const val PREFIX_NOTIFICATIONS = "notif:"
    const val PREFIX_CHAT = "chat:"
    const val PREFIX_FEED = "feed:"
    const val PREFIX_TRENDING = "trending"
    const val PREFIX_RATE_LIMIT = "rate:"
    const val PREFIX_SUGGESTIONS = "suggest:"
    
    // TTL values (in seconds)
    const val TTL_ONLINE = 300
    const val TTL_CHAT_CACHE = 3600
    const val TTL_RATE_LIMIT = 60
    const val TTL_SUGGESTIONS = 86400
}

/**
 * Local In-Memory Cache - Fallback when Redis unavailable
 */
object LocalCache {
    private val store = ConcurrentHashMap<String, CacheEntry>()
    private val sortedSets = ConcurrentHashMap<String, MutableMap<String, Double>>()
    private val lists = ConcurrentHashMap<String, MutableList<String>>()
    
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    init {
        scope.launch {
            while (true) {
                delay(60_000)
                cleanupExpired()
            }
        }
    }
    
    data class CacheEntry(val value: String, val expiresAt: Long? = null) {
        fun isExpired() = expiresAt != null && System.currentTimeMillis() > expiresAt
    }
    
    private fun cleanupExpired() {
        store.entries.removeIf { it.value.isExpired() }
    }
    
    fun set(key: String, value: String, ttlSeconds: Int? = null) {
        val expiresAt = ttlSeconds?.let { System.currentTimeMillis() + (it * 1000L) }
        store[key] = CacheEntry(value, expiresAt)
    }
    
    fun get(key: String): String? {
        val entry = store[key] ?: return null
        return if (entry.isExpired()) { store.remove(key); null } else entry.value
    }
    
    fun delete(key: String) = store.remove(key) != null
    
    fun keys(pattern: String): List<String> {
        val regex = pattern.replace("*", ".*").toRegex()
        return store.keys.filter { regex.matches(it) && !store[it]!!.isExpired() }
    }
    
    fun incr(key: String): Long {
        val newValue = (get(key)?.toLongOrNull() ?: 0L) + 1
        set(key, newValue.toString())
        return newValue
    }
    
    fun decr(key: String): Long {
        val newValue = maxOf(0L, (get(key)?.toLongOrNull() ?: 0L) - 1)
        set(key, newValue.toString())
        return newValue
    }
    
    fun expire(key: String, ttl: Int) {
        store[key]?.let { store[key] = it.copy(expiresAt = System.currentTimeMillis() + (ttl * 1000L)) }
    }
    
    fun lpush(key: String, value: String): Long {
        val list = lists.getOrPut(key) { mutableListOf() }
        list.add(0, value)
        if (list.size > 100) list.removeAt(list.lastIndex)
        return list.size.toLong()
    }
    
    fun rpush(key: String, value: String): Long {
        val list = lists.getOrPut(key) { mutableListOf() }
        list.add(value)
        if (list.size > 100) list.removeAt(0)
        return list.size.toLong()
    }
    
    fun lrange(key: String, start: Int, stop: Int): List<String> {
        val list = lists[key] ?: return emptyList()
        val actualStop = if (stop < 0) list.size + stop + 1 else minOf(stop + 1, list.size)
        val actualStart = if (start < 0) maxOf(0, list.size + start) else minOf(start, list.size)
        return if (actualStart < actualStop) list.subList(actualStart, actualStop) else emptyList()
    }
    
    fun zadd(key: String, score: Double, member: String) {
        sortedSets.getOrPut(key) { mutableMapOf() }[member] = score
    }
    
    fun zincrby(key: String, increment: Double, member: String): Double {
        val set = sortedSets.getOrPut(key) { mutableMapOf() }
        val newScore = (set[member] ?: 0.0) + increment
        set[member] = newScore
        return newScore
    }
    
    fun zrevrange(key: String, start: Int, stop: Int): List<String> {
        val set = sortedSets[key] ?: return emptyList()
        val sorted = set.entries.sortedByDescending { it.value }
        val actualStop = minOf(stop + 1, sorted.size)
        return if (start < actualStop) sorted.subList(start, actualStop).map { it.key } else emptyList()
    }
    
    fun getSortedSet(key: String): MutableMap<String, Double>? = sortedSets[key]
}

/**
 * Production Redis Service
 * Uses Upstash REST API for real Redis, falls back to local cache
 */
object RedisService {
    
    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .build()
    
    private val _onlineUsers = MutableStateFlow<Set<String>>(emptySet())
    val onlineUsers: StateFlow<Set<String>> = _onlineUsers.asStateFlow()
    
    /**
     * Execute Upstash Redis command via REST API
     */
    private suspend fun upstashCommand(vararg args: String): String? = withContext(Dispatchers.IO) {
        if (!RedisConfig.isUpstashConfigured()) return@withContext null
        
        try {
            val jsonArray = JSONArray(args.toList())
            val request = Request.Builder()
                .url(RedisConfig.UPSTASH_REDIS_URL)
                .addHeader("Authorization", "Bearer ${RedisConfig.UPSTASH_REDIS_TOKEN}")
                .post(jsonArray.toString().toRequestBody("application/json".toMediaType()))
                .build()
            
            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                val body = response.body?.string() ?: "{}"
                // Extract result from {"result": value}
                val resultRegex = """"result"\s*:\s*(?:"([^"]*)"|([\d.]+)|(\[.*\]))""".toRegex()
                val match = resultRegex.find(body)
                match?.groupValues?.drop(1)?.firstOrNull { it.isNotEmpty() } ?: body
            } else null
        } catch (e: Exception) {
            null
        }
    }
    
    // ==========================================================================
    // 🟢 ONLINE STATUS
    // ==========================================================================
    
    suspend fun setUserOnline(userId: String): Boolean = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_ONLINE}$userId"
        val result = upstashCommand("SETEX", key, RedisConfig.TTL_ONLINE.toString(), System.currentTimeMillis().toString())
        
        if (result == null && RedisConfig.USE_LOCAL_FALLBACK) {
            LocalCache.set(key, System.currentTimeMillis().toString(), RedisConfig.TTL_ONLINE)
        }
        
        _onlineUsers.value = _onlineUsers.value + userId
        true
    }
    
    suspend fun setUserOffline(userId: String): Boolean = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_ONLINE}$userId"
        upstashCommand("DEL", key)
        LocalCache.delete(key)
        _onlineUsers.value = _onlineUsers.value - userId
        true
    }
    
    suspend fun isUserOnline(userId: String): Boolean = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_ONLINE}$userId"
        val result = upstashCommand("GET", key)
        result?.isNotBlank() == true || LocalCache.get(key) != null
    }
    
    suspend fun getOnlineUsers(): Set<String> = withContext(Dispatchers.IO) {
        val result = upstashCommand("KEYS", "${RedisConfig.PREFIX_ONLINE}*")
        val users = if (result != null) {
            result.removeSurrounding("[", "]")
                .split(",")
                .map { it.trim().removeSurrounding("\"").removePrefix(RedisConfig.PREFIX_ONLINE) }
                .filter { it.isNotBlank() }
                .toSet()
        } else {
            LocalCache.keys("${RedisConfig.PREFIX_ONLINE}*")
                .map { it.removePrefix(RedisConfig.PREFIX_ONLINE) }
                .toSet()
        }
        _onlineUsers.value = users
        users
    }
    
    // ==========================================================================
    // ❤️ LIKE / 👁️ VIEW COUNTERS
    // ==========================================================================
    
    suspend fun incrementViews(postId: String): Long = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_VIEWS}$postId"
        val result = upstashCommand("INCR", key)?.toLongOrNull()
        result ?: LocalCache.incr(key)
    }
    
    suspend fun getViews(postId: String): Long = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_VIEWS}$postId"
        val result = upstashCommand("GET", key)?.toLongOrNull()
        result ?: LocalCache.get(key)?.toLongOrNull() ?: 0L
    }
    
    suspend fun incrementLikes(postId: String): Long = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_LIKES}$postId"
        val result = upstashCommand("INCR", key)?.toLongOrNull()
        result ?: LocalCache.incr(key)
    }
    
    suspend fun decrementLikes(postId: String): Long = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_LIKES}$postId"
        val result = upstashCommand("DECR", key)?.toLongOrNull()
        result ?: LocalCache.decr(key)
    }
    
    suspend fun incrementShares(postId: String): Long = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_SHARES}$postId"
        val result = upstashCommand("INCR", key)?.toLongOrNull()
        result ?: LocalCache.incr(key)
    }
    
    // ==========================================================================
    // 🔔 NOTIFICATIONS
    // ==========================================================================
    
    data class Notification(
        val id: String,
        val type: String,
        val title: String,
        val message: String,
        val fromUserId: String,
        val timestamp: Long,
        val isRead: Boolean = false
    )
    
    suspend fun pushNotification(userId: String, notification: Notification): Boolean = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_NOTIFICATIONS}$userId"
        val json = """{"id":"${notification.id}","type":"${notification.type}","title":"${notification.title}","message":"${notification.message}","fromUserId":"${notification.fromUserId}","timestamp":${notification.timestamp},"isRead":${notification.isRead}}"""
        
        val result = upstashCommand("LPUSH", key, json)
        if (result == null && RedisConfig.USE_LOCAL_FALLBACK) {
            LocalCache.lpush(key, json)
        }
        true
    }
    
    suspend fun getNotifications(userId: String, limit: Int = 50): List<Notification> = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_NOTIFICATIONS}$userId"
        val result = upstashCommand("LRANGE", key, "0", limit.toString())
        
        val items = if (result != null && result.startsWith("[")) {
            try {
                JSONArray(result).let { arr ->
                    (0 until arr.length()).map { arr.getString(it) }
                }
            } catch (e: Exception) { emptyList() }
        } else {
            LocalCache.lrange(key, 0, limit)
        }
        
        items.mapNotNull { parseNotification(it) }
    }
    
    private fun parseNotification(json: String): Notification? = try {
        Notification(
            id = Regex(""""id":"([^"]+)"""").find(json)?.groupValues?.get(1) ?: "",
            type = Regex(""""type":"([^"]+)"""").find(json)?.groupValues?.get(1) ?: "",
            title = Regex(""""title":"([^"]+)"""").find(json)?.groupValues?.get(1) ?: "",
            message = Regex(""""message":"([^"]+)"""").find(json)?.groupValues?.get(1) ?: "",
            fromUserId = Regex(""""fromUserId":"([^"]+)"""").find(json)?.groupValues?.get(1) ?: "",
            timestamp = Regex(""""timestamp":(\d+)""").find(json)?.groupValues?.get(1)?.toLongOrNull() ?: 0L,
            isRead = json.contains(""""isRead":true""")
        )
    } catch (e: Exception) { null }
    
    // ==========================================================================
    // 💬 CHAT CACHE
    // ==========================================================================
    
    data class ChatMessage(
        val id: String,
        val senderId: String,
        val content: String,
        val timestamp: Long,
        val isDelivered: Boolean = false,
        val isRead: Boolean = false
    )
    
    suspend fun cacheMessage(conversationId: String, message: ChatMessage): Boolean = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_CHAT}$conversationId"
        val json = """{"id":"${message.id}","senderId":"${message.senderId}","content":"${message.content}","timestamp":${message.timestamp},"isDelivered":${message.isDelivered},"isRead":${message.isRead}}"""
        
        val result = upstashCommand("RPUSH", key, json)
        if (result == null && RedisConfig.USE_LOCAL_FALLBACK) {
            LocalCache.rpush(key, json)
        }
        upstashCommand("EXPIRE", key, RedisConfig.TTL_CHAT_CACHE.toString())
        true
    }
    
    // Overload for Message model
    suspend fun cacheMessage(conversationId: String, message: com.orignal.buddylynk.data.model.Message): Boolean = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_CHAT}$conversationId"
        val json = """{"id":"${message.messageId}","senderId":"${message.senderId}","content":"${message.content}","timestamp":${message.createdAt},"isDelivered":true,"isRead":${message.isRead}}"""
        
        val result = upstashCommand("RPUSH", key, json)
        if (result == null && RedisConfig.USE_LOCAL_FALLBACK) {
            LocalCache.rpush(key, json)
        }
        upstashCommand("EXPIRE", key, RedisConfig.TTL_CHAT_CACHE.toString())
        true
    }
    
    suspend fun getCachedMessages(conversationId: String, limit: Int = 50): List<ChatMessage> = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_CHAT}$conversationId"
        val result = upstashCommand("LRANGE", key, (-limit).toString(), "-1")
        
        val items = if (result != null && result.startsWith("[")) {
            try {
                JSONArray(result).let { arr -> (0 until arr.length()).map { arr.getString(it) } }
            } catch (e: Exception) { emptyList() }
        } else {
            LocalCache.lrange(key, -limit, -1)
        }
        
        items.mapNotNull { parseChatMessage(it) }
    }
    
    private fun parseChatMessage(json: String): ChatMessage? = try {
        ChatMessage(
            id = Regex(""""id":"([^"]+)"""").find(json)?.groupValues?.get(1) ?: "",
            senderId = Regex(""""senderId":"([^"]+)"""").find(json)?.groupValues?.get(1) ?: "",
            content = Regex(""""content":"([^"]+)"""").find(json)?.groupValues?.get(1) ?: "",
            timestamp = Regex(""""timestamp":(\d+)""").find(json)?.groupValues?.get(1)?.toLongOrNull() ?: 0L,
            isDelivered = json.contains(""""isDelivered":true"""),
            isRead = json.contains(""""isRead":true""")
        )
    } catch (e: Exception) { null }
    
    // ==========================================================================
    // 🔥 TRENDING FEED (Sorted Sets)
    // ==========================================================================
    
    suspend fun addToTrending(postId: String, score: Double): Boolean = withContext(Dispatchers.IO) {
        val result = upstashCommand("ZADD", RedisConfig.PREFIX_TRENDING, score.toString(), postId)
        if (result == null && RedisConfig.USE_LOCAL_FALLBACK) {
            LocalCache.zadd(RedisConfig.PREFIX_TRENDING, score, postId)
        }
        true
    }
    
    suspend fun getTrendingPosts(limit: Int = 20): List<String> = withContext(Dispatchers.IO) {
        val result = upstashCommand("ZREVRANGE", RedisConfig.PREFIX_TRENDING, "0", limit.toString())
        
        if (result != null && result.startsWith("[")) {
            try {
                JSONArray(result).let { arr -> (0 until arr.length()).map { arr.getString(it) } }
            } catch (e: Exception) { emptyList() }
        } else {
            LocalCache.zrevrange(RedisConfig.PREFIX_TRENDING, 0, limit)
        }
    }
    
    suspend fun incrementPostScore(postId: String, incrementBy: Double = 1.0): Boolean = withContext(Dispatchers.IO) {
        val result = upstashCommand("ZINCRBY", RedisConfig.PREFIX_TRENDING, incrementBy.toString(), postId)
        if (result == null && RedisConfig.USE_LOCAL_FALLBACK) {
            LocalCache.zincrby(RedisConfig.PREFIX_TRENDING, incrementBy, postId)
        }
        true
    }
    
    // ==========================================================================
    // 🚦 RATE LIMITING
    // ==========================================================================
    
    suspend fun checkRateLimit(userId: String, action: String, maxRequests: Int = 60): Boolean = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_RATE_LIMIT}$userId:$action"
        
        val result = upstashCommand("INCR", key)?.toLongOrNull()
        val count = if (result != null) {
            if (result == 1L) {
                upstashCommand("EXPIRE", key, RedisConfig.TTL_RATE_LIMIT.toString())
            }
            result
        } else {
            val localCount = LocalCache.incr(key)
            if (localCount == 1L) LocalCache.expire(key, RedisConfig.TTL_RATE_LIMIT)
            localCount
        }
        
        count <= maxRequests
    }
    
    // ==========================================================================
    // 👥 FRIEND SUGGESTIONS
    // ==========================================================================
    
    suspend fun cacheSuggestions(userId: String, suggestions: List<String>): Boolean = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_SUGGESTIONS}$userId"
        val value = suggestions.joinToString(",")
        
        val result = upstashCommand("SETEX", key, RedisConfig.TTL_SUGGESTIONS.toString(), value)
        if (result == null && RedisConfig.USE_LOCAL_FALLBACK) {
            LocalCache.set(key, value, RedisConfig.TTL_SUGGESTIONS)
        }
        true
    }
    
    suspend fun getSuggestions(userId: String): List<String> = withContext(Dispatchers.IO) {
        val key = "${RedisConfig.PREFIX_SUGGESTIONS}$userId"
        val result = upstashCommand("GET", key)
        
        (result ?: LocalCache.get(key))?.split(",")?.filter { it.isNotBlank() } ?: emptyList()
    }
    
    // ==========================================================================
    // # HASHTAGS
    // ==========================================================================
    
    suspend fun getTrendingHashtags(limit: Int = 10): List<String> = withContext(Dispatchers.IO) {
        val key = "hashtags:trending"
        val result = upstashCommand("ZREVRANGE", key, "0", (limit - 1).toString())
        
        if (result != null) {
            result.split(",").filter { it.isNotBlank() }
        } else {
            LocalCache.getSortedSet("hashtags:trending")
                ?.entries
                ?.sortedByDescending { it.value }
                ?.take(limit)
                ?.map { it.key }
                ?: emptyList()
        }
    }
    
    suspend fun getHashtagPostCount(hashtag: String): Long = withContext(Dispatchers.IO) {
        val key = "hashtag:count:${hashtag.lowercase()}"
        upstashCommand("GET", key)?.toLongOrNull() ?: LocalCache.get(key)?.toLongOrNull() ?: 0L
    }
    
    suspend fun incrementHashtagCount(hashtag: String) = withContext(Dispatchers.IO) {
        val key = "hashtag:count:${hashtag.lowercase()}"
        upstashCommand("INCR", key) ?: LocalCache.incr(key)
    }
    
    suspend fun searchHashtags(prefix: String, limit: Int = 20): List<String> = withContext(Dispatchers.IO) {
        // In production, use Redis SCAN or secondary index
        // For now, return from trending that match prefix
        getTrendingHashtags(50).filter { it.startsWith(prefix) }.take(limit)
    }
    
    suspend fun addToTrending(hashtag: String) = withContext(Dispatchers.IO) {
        val key = "hashtags:trending"
        val tag = hashtag.lowercase()
        val score = System.currentTimeMillis().toDouble()
        
        upstashCommand("ZINCRBY", key, "1", tag) ?: LocalCache.zadd(key, score, tag)
    }
    
    suspend fun linkPostToHashtag(postId: String, hashtag: String) = withContext(Dispatchers.IO) {
        val key = "hashtag:posts:${hashtag.lowercase()}"
        upstashCommand("LPUSH", key, postId) ?: LocalCache.lpush(key, postId)
    }
    
    suspend fun getPostsForHashtag(hashtag: String, limit: Int = 50): List<String> = withContext(Dispatchers.IO) {
        val key = "hashtag:posts:${hashtag.lowercase()}"
        val result = upstashCommand("LRANGE", key, "0", (limit - 1).toString())
        
        result?.split(",")?.filter { it.isNotBlank() } 
            ?: LocalCache.lrange(key, 0, limit) 
            ?: emptyList()
    }
}

