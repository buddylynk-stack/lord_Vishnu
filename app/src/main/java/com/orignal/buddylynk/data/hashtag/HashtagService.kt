package com.orignal.buddylynk.data.hashtag

import com.orignal.buddylynk.data.aws.DynamoDbService
import com.orignal.buddylynk.data.redis.RedisService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Hashtag Service - Parse, extract, and manage hashtags
 */
object HashtagService {
    
    // Hashtag regex pattern
    private val HASHTAG_REGEX = Regex("#[a-zA-Z0-9_]+")
    
    /**
     * Extract hashtags from text
     */
    fun extractHashtags(text: String): List<String> {
        return HASHTAG_REGEX.findAll(text)
            .map { it.value.lowercase() }
            .distinct()
            .toList()
    }
    
    /**
     * Make text clickable with hashtags
     * Returns pairs of (text, isHashtag)
     */
    fun parseTextWithHashtags(text: String): List<TextSegment> {
        val segments = mutableListOf<TextSegment>()
        var lastEnd = 0
        
        HASHTAG_REGEX.findAll(text).forEach { match ->
            // Add text before hashtag
            if (match.range.first > lastEnd) {
                segments.add(TextSegment(
                    text = text.substring(lastEnd, match.range.first),
                    isHashtag = false
                ))
            }
            
            // Add hashtag
            segments.add(TextSegment(
                text = match.value,
                isHashtag = true,
                hashtag = match.value.lowercase()
            ))
            
            lastEnd = match.range.last + 1
        }
        
        // Add remaining text
        if (lastEnd < text.length) {
            segments.add(TextSegment(
                text = text.substring(lastEnd),
                isHashtag = false
            ))
        }
        
        return segments
    }
    
    /**
     * Get trending hashtags
     */
    suspend fun getTrendingHashtags(limit: Int = 10): List<TrendingHashtag> = withContext(Dispatchers.IO) {
        try {
            // Get from Redis trending set
            val trending = RedisService.getTrendingHashtags(limit)
            trending.mapIndexed { index, tag ->
                TrendingHashtag(
                    hashtag = tag,
                    postsCount = RedisService.getHashtagPostCount(tag).toInt(),
                    rank = index + 1,
                    trend = if (index < 3) TrendDirection.UP else TrendDirection.STABLE
                )
            }
        } catch (e: Exception) {
            // Return popular defaults
            getDefaultTrendingHashtags()
        }
    }
    
    /**
     * Search hashtags by prefix
     */
    suspend fun searchHashtags(query: String, limit: Int = 20): List<String> = withContext(Dispatchers.IO) {
        try {
            val searchTerm = if (query.startsWith("#")) query else "#$query"
            RedisService.searchHashtags(searchTerm.lowercase(), limit)
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    /**
     * Record hashtag usage (called when post is created)
     */
    suspend fun recordHashtagUsage(hashtags: List<String>, postId: String) = withContext(Dispatchers.IO) {
        hashtags.forEach { tag ->
            try {
                // Increment usage count
                RedisService.incrementHashtagCount(tag)
                // Add to trending
                RedisService.addToTrending(tag)
                // Link post to hashtag
                RedisService.linkPostToHashtag(postId, tag)
            } catch (e: Exception) {
                // Log error
            }
        }
    }
    
    /**
     * Get posts for a hashtag
     */
    suspend fun getPostsForHashtag(hashtag: String, limit: Int = 50): List<String> = withContext(Dispatchers.IO) {
        try {
            val tag = if (hashtag.startsWith("#")) hashtag else "#$hashtag"
            RedisService.getPostsForHashtag(tag.lowercase(), limit)
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    /**
     * Get suggested hashtags based on content
     */
    fun getSuggestedHashtags(content: String): List<String> {
        val words = content.lowercase().split(" ", ",", ".", "!", "?")
        val suggestions = mutableListOf<String>()
        
        // Common topic mappings
        val topicHashtags = mapOf(
            "gaming" to listOf("#gaming", "#gamer", "#games"),
            "music" to listOf("#music", "#songs", "#beats"),
            "travel" to listOf("#travel", "#wanderlust", "#adventure"),
            "food" to listOf("#food", "#foodie", "#yummy"),
            "fitness" to listOf("#fitness", "#workout", "#gym"),
            "art" to listOf("#art", "#creative", "#artist"),
            "tech" to listOf("#tech", "#technology", "#coding"),
            "fashion" to listOf("#fashion", "#style", "#ootd"),
            "nature" to listOf("#nature", "#outdoors", "#beautiful"),
            "love" to listOf("#love", "#couple", "#relationship")
        )
        
        words.forEach { word ->
            topicHashtags[word]?.let { suggestions.addAll(it) }
        }
        
        return suggestions.distinct().take(5)
    }
    
    /**
     * Default trending hashtags
     */
    private fun getDefaultTrendingHashtags() = listOf(
        TrendingHashtag("#viral", 15420, 1, TrendDirection.UP),
        TrendingHashtag("#trending", 12300, 2, TrendDirection.UP),
        TrendingHashtag("#fyp", 10500, 3, TrendDirection.UP),
        TrendingHashtag("#buddylynk", 8900, 4, TrendDirection.STABLE),
        TrendingHashtag("#explore", 7200, 5, TrendDirection.DOWN),
        TrendingHashtag("#photography", 6100, 6, TrendDirection.STABLE),
        TrendingHashtag("#lifestyle", 5400, 7, TrendDirection.STABLE),
        TrendingHashtag("#gaming", 4800, 8, TrendDirection.UP),
        TrendingHashtag("#music", 4200, 9, TrendDirection.STABLE),
        TrendingHashtag("#art", 3600, 10, TrendDirection.DOWN)
    )
}

/**
 * Text segment for hashtag parsing
 */
data class TextSegment(
    val text: String,
    val isHashtag: Boolean,
    val hashtag: String? = null
)

/**
 * Trending hashtag with stats
 */
data class TrendingHashtag(
    val hashtag: String,
    val postsCount: Int,
    val rank: Int,
    val trend: TrendDirection
)

/**
 * Trend direction
 */
enum class TrendDirection {
    UP, DOWN, STABLE
}
