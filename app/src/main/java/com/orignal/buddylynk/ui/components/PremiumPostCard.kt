package com.orignal.buddylynk.ui.components

import android.graphics.drawable.ColorDrawable
import android.net.Uri
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.painter.ColorPainter
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.ui.AspectRatioFrameLayout
import androidx.media3.ui.PlayerView
import coil.compose.AsyncImage
import coil.compose.SubcomposeAsyncImage
import coil.request.ImageRequest
import com.orignal.buddylynk.data.cache.VideoPlayerCache
import com.orignal.buddylynk.data.cache.FeedPlayerManager
import com.orignal.buddylynk.data.model.Post
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

// Premium Colors matching React
private val DarkBg = Color(0xFF050505)
private val CardBg = Color(0xFF1A1A1A)
private val BorderWhite10 = Color.White.copy(alpha = 0.1f)
private val CyanGlow = Color(0xFF22D3EE)
private val IndigoAccent = Color(0xFF6366F1)

/**
 * Format timestamp to relative time (e.g., "1 min ago", "2 hours ago")
 */
private fun formatTimeAgo(createdAt: String): String {
    if (createdAt.isBlank()) return ""
    
    return try {
        // Parse ISO8601 timestamp
        val timestamp = try {
            java.time.Instant.parse(createdAt).toEpochMilli()
        } catch (e: Exception) {
            // Try parsing as epoch millis string
            createdAt.toLongOrNull() ?: return ""
        }
        
        val now = System.currentTimeMillis()
        val diff = now - timestamp
        
        val seconds = diff / 1000
        val minutes = seconds / 60
        val hours = minutes / 60
        val days = hours / 24
        val weeks = days / 7
        val months = days / 30
        
        when {
            seconds < 60 -> "Just now"
            minutes < 60 -> "${minutes}m ago"
            hours < 24 -> "${hours}h ago"
            days == 1L -> "Yesterday"
            days < 7 -> "${days}d ago"
            weeks < 4 -> "${weeks}w ago"
            months < 12 -> "${months}mo ago"
            else -> "${days / 365}y ago"
        }
    } catch (e: Exception) {
        ""
    }
}

/**
 * Premium Post Card with "Action Capsule" design from React
 * Features:
 * - Double-tap to like with shockwave animation
 * - Spinning neon ring for users with stories
 * - Capsule-style action buttons (like React design)
 * - Bookmark button with active state
 */
@Composable
fun PremiumPostCard(
    post: Post,
    isLiked: Boolean,
    isSaved: Boolean,
    hasStatus: Boolean = false,
    isOwner: Boolean = false,
    isVisible: Boolean = true, // Only play video when visible on screen
    onLike: () -> Unit,
    onSave: () -> Unit,
    onComment: () -> Unit,
    onShare: () -> Unit,
    onUserClick: (String) -> Unit = {},
    onMoreClick: () -> Unit = {},
    onEdit: () -> Unit = {},
    onDelete: () -> Unit = {},
    onReport: () -> Unit = {},
    onBlock: () -> Unit = {},
    onAvatarLongPress: () -> Unit = {},
    onNavigateToShorts: () -> Unit = {}, // Navigate to Shorts when video is long pressed
    modifier: Modifier = Modifier
) {
    var showLikeAnimation by remember { mutableStateOf(false) }
    var cardScale by remember { mutableStateOf(1f) }
    var isFullscreen by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    
    // NSFW content handling - check admin flags and user settings
    val sensitiveMode by com.orignal.buddylynk.data.settings.SensitiveContentManager.contentMode.collectAsState()
    val isNSFWContent = post.isNSFW || post.isSensitive
    var showNSFWContent by remember(post.postId) { mutableStateOf(false) } // User clicked "Show" - keyed by postId to persist
    
    // Determine if we should hide/blur this post
    val shouldHidePost = isNSFWContent && sensitiveMode == com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.HIDE && !showNSFWContent
    val shouldBlurPost = isNSFWContent && sensitiveMode == com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.BLUR && !showNSFWContent
    
    // Debug logging for NSFW
    LaunchedEffect(post.postId, isNSFWContent, sensitiveMode) {
        if (post.isNSFW || post.isSensitive) {
            android.util.Log.d("NSFW_DEBUG", "Post ${post.postId.take(8)}: isNSFW=${post.isNSFW}, isSensitive=${post.isSensitive}, mode=$sensitiveMode, shouldBlur=$shouldBlurPost, shouldHide=$shouldHidePost")
        }
    }

    // Animate card scale
    val animatedScale by animateFloatAsState(
        targetValue = cardScale,
        animationSpec = spring(dampingRatio = 0.5f, stiffness = 300f),
        label = "cardScale"
    )

    // Double tap handler
    val handleDoubleTap = {
        if (!isLiked) {
            onLike()
        }
        showLikeAnimation = true
        cardScale = 0.98f
        scope.launch {
            delay(600)
            showLikeAnimation = false
            cardScale = 1f
        }
    }
    
    // Production-ready fullscreen overlay using Dialog
    if (isFullscreen) {
        androidx.compose.ui.window.Dialog(
            onDismissRequest = { isFullscreen = false },
            properties = androidx.compose.ui.window.DialogProperties(
                dismissOnBackPress = true,
                dismissOnClickOutside = false,
                usePlatformDefaultWidth = false // Full width
            )
        ) {
            // Immersive fullscreen container
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black)
                    .systemBarsPadding(),
                contentAlignment = Alignment.Center
            ) {
                // Get all media URLs
                val allMediaUrls = if (post.mediaUrls.isNotEmpty()) {
                    post.mediaUrls
                } else if (!post.mediaUrl.isNullOrBlank()) {
                    listOf(post.mediaUrl)
                } else {
                    emptyList()
                }
                
                val firstMediaUrl = allMediaUrls.firstOrNull()
                val isVideo = firstMediaUrl?.let { url ->
                    url.endsWith(".mp4", ignoreCase = true) ||
                    url.endsWith(".webm", ignoreCase = true) ||
                    url.endsWith(".mov", ignoreCase = true) ||
                    post.mediaType == "video"
                } ?: false
                
                if (allMediaUrls.isNotEmpty()) {
                    if (isVideo && firstMediaUrl != null) {
                        // Fullscreen video player
                        val context = LocalContext.current
                        FeedPlayerManager.init(context)
                        
                        @androidx.annotation.OptIn(UnstableApi::class)
                        val fullscreenPlayer = remember(firstMediaUrl) {
                            FeedPlayerManager.getPlayer(
                                url = firstMediaUrl,
                                onBufferingChange = { },
                                onError = { }
                            )
                        }
                        
                        fullscreenPlayer?.let { player ->
                            player.volume = 1f // Unmute in fullscreen
                            player.playWhenReady = true
                            
                            AndroidView(
                                factory = { ctx ->
                                    PlayerView(ctx).apply {
                                        this.player = player
                                        useController = true // Show controls in fullscreen
                                        controllerShowTimeoutMs = 3000
                                        controllerAutoShow = true
                                        resizeMode = AspectRatioFrameLayout.RESIZE_MODE_FIT
                                        setBackgroundColor(android.graphics.Color.BLACK)
                                        layoutParams = FrameLayout.LayoutParams(
                                            ViewGroup.LayoutParams.MATCH_PARENT,
                                            ViewGroup.LayoutParams.MATCH_PARENT
                                        )
                                    }
                                },
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                    } else {
                        // Fullscreen image gallery with HorizontalPager
                        val fullscreenPagerState = androidx.compose.foundation.pager.rememberPagerState { allMediaUrls.size }
                        
                        Box(modifier = Modifier.fillMaxSize()) {
                            androidx.compose.foundation.pager.HorizontalPager(
                                state = fullscreenPagerState,
                                modifier = Modifier.fillMaxSize()
                            ) { page ->
                                val imageUrl = allMediaUrls[page]
                                AsyncImage(
                                    model = ImageRequest.Builder(LocalContext.current)
                                        .data(imageUrl)
                                        .crossfade(true)
                                        .build(),
                                    contentDescription = "Fullscreen image ${page + 1}",
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .pointerInput(Unit) {
                                            detectTapGestures(onTap = { isFullscreen = false })
                                        },
                                    contentScale = ContentScale.Fit
                                )
                            }
                            
                            // Page indicator for multiple images
                            if (allMediaUrls.size > 1) {
                                Row(
                                    modifier = Modifier
                                        .align(Alignment.TopCenter)
                                        .padding(top = 40.dp)
                                        .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(20.dp))
                                        .padding(horizontal = 12.dp, vertical = 6.dp),
                                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    repeat(allMediaUrls.size) { index ->
                                        Box(
                                            modifier = Modifier
                                                .size(if (fullscreenPagerState.currentPage == index) 8.dp else 6.dp)
                                                .clip(CircleShape)
                                                .background(
                                                    if (fullscreenPagerState.currentPage == index) 
                                                        CyanGlow else Color.White.copy(alpha = 0.5f)
                                                )
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Premium close button with glassmorphism
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(top = 24.dp, end = 16.dp)
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(
                            Brush.radialGradient(
                                colors = listOf(
                                    Color.White.copy(alpha = 0.2f),
                                    Color.White.copy(alpha = 0.1f)
                                )
                            )
                        )
                        .border(1.dp, Color.White.copy(alpha = 0.3f), CircleShape)
                        .clickable { isFullscreen = false },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Filled.Close,
                        contentDescription = "Close",
                        tint = Color.White,
                        modifier = Modifier.size(24.dp)
                    )
                }
                
                // User info at bottom
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .fillMaxWidth()
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(Color.Transparent, Color.Black.copy(alpha = 0.8f))
                            )
                        )
                        .padding(16.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // User avatar
                        AsyncImage(
                            model = post.userAvatar,
                            contentDescription = "User",
                            modifier = Modifier
                                .size(40.dp)
                                .clip(CircleShape)
                                .border(2.dp, CyanGlow, CircleShape),
                            contentScale = ContentScale.Crop
                        )
                        
                        Column {
                            Text(
                                text = post.username ?: "Unknown",
                                color = Color.White,
                                fontSize = 15.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                            if (post.content.isNotBlank()) {
                                Text(
                                    text = post.content,
                                    color = Color.White.copy(alpha = 0.7f),
                                    fontSize = 13.sp,
                                    maxLines = 2,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    Column(modifier = modifier) {
        // HIDDEN POST: Show placeholder if user selected HIDE mode for NSFW content
        if (shouldHidePost) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(4f / 5f)
                    .clip(RoundedCornerShape(32.dp))
                    .background(
                        Brush.verticalGradient(
                            colors = listOf(
                                Color(0xFF1A1A2E),
                                Color(0xFF16162A),
                                Color(0xFF1A1A2E)
                            )
                        )
                    )
                    .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(32.dp)),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    // Warning Icon with glow
                    Box(
                        modifier = Modifier
                            .size(80.dp)
                            .clip(CircleShape)
                            .background(
                                Brush.radialGradient(
                                    colors = listOf(
                                        Color(0xFFF97316).copy(alpha = 0.2f),
                                        Color.Transparent
                                    )
                                )
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Filled.VisibilityOff,
                            contentDescription = "Hidden Content",
                            tint = Color(0xFFF97316),
                            modifier = Modifier.size(40.dp)
                        )
                    }
                    
                    Text(
                        text = "Sensitive Content",
                        color = Color.White,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "This post is hidden because it contains\n18+ adult content",
                        color = Color.White.copy(alpha = 0.6f),
                        fontSize = 14.sp,
                        textAlign = TextAlign.Center
                    )
                    // NO reveal button - content is completely hidden
                }
            }
        } else {
        // Image Container
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(4f / 5f) // Instagram-style 4:5 ratio - content fills better
                .scale(animatedScale)
                .clip(RoundedCornerShape(32.dp))
                .background(CardBg)
                .border(1.dp, BorderWhite10, RoundedCornerShape(32.dp))
                // NOTE: Blur is now applied ONLY to the thumbnail inside the overlay, not here
                .pointerInput(Unit) {
                    detectTapGestures(
                        onDoubleTap = { handleDoubleTap() },
                        onLongPress = { 
                            // Check if it's a video - navigate to Shorts, otherwise fullscreen for images
                            val mediaUrl = post.mediaUrl ?: post.mediaUrls.firstOrNull()
                            val isVideoPost = mediaUrl?.let { url ->
                                url.endsWith(".mp4", ignoreCase = true) ||
                                url.endsWith(".webm", ignoreCase = true) ||
                                url.endsWith(".mov", ignoreCase = true) ||
                                post.mediaType == "video"
                            } ?: false
                            
                            if (isVideoPost) {
                                onNavigateToShorts() // Go to Shorts for videos
                            } else {
                                isFullscreen = true // Fullscreen gallery for images
                            }
                        }
                    )
                }
        ) {
            // Post Image/Video - Support for multiple media
            val allMediaUrls = if (post.mediaUrls.isNotEmpty()) {
                post.mediaUrls
            } else if (!post.mediaUrl.isNullOrBlank()) {
                listOf(post.mediaUrl)
            } else {
                emptyList()
            }
            
            val context = LocalContext.current
            
            if (allMediaUrls.isNotEmpty()) {
                var currentPage by remember { mutableIntStateOf(0) }
                
                Box(modifier = Modifier.fillMaxSize()) {
                    // CRITICAL: Only render actual media when NOT blurred
                    // When shouldBlurPost is true, we skip rendering actual content entirely
                    if (!shouldBlurPost) {
                        // Media display with swipe support - ONLY when not blurred
                        androidx.compose.foundation.pager.HorizontalPager(
                            state = androidx.compose.foundation.pager.rememberPagerState { allMediaUrls.size }.also { 
                                currentPage = it.currentPage 
                            },
                            modifier = Modifier.fillMaxSize()
                        ) { page ->
                        val mediaUrl = allMediaUrls[page]
                        
                        // Better video detection - check multiple formats
                        val isVideo = mediaUrl.endsWith(".mp4", ignoreCase = true) || 
                                       mediaUrl.endsWith(".webm", ignoreCase = true) ||
                                       mediaUrl.endsWith(".mov", ignoreCase = true) ||
                                       mediaUrl.endsWith(".avi", ignoreCase = true) ||
                                       mediaUrl.endsWith(".mkv", ignoreCase = true) ||
                                       mediaUrl.contains("/video/", ignoreCase = true) ||
                                       post.mediaType == "video"
                        
                        android.util.Log.d("PremiumPostCard", "Media URL: $mediaUrl, isVideo: $isVideo, mediaType: ${post.mediaType}")
                        
                        if (isVideo) {
                            // Video Player with pooled ExoPlayer (no re-buffering on scroll!)
                            // Start with FALSE - assume player is ready (for pooled players)
                            var isBuffering by remember { mutableStateOf(false) }
                            var hasError by remember { mutableStateOf(false) }
                            // Use global mute state from FeedPlayerManager (persists across scroll)
                            var isMutedState by remember { mutableStateOf(FeedPlayerManager.isMuted) }
                            
                            // Initialize player manager with context
                            FeedPlayerManager.init(context)
                            
                            @androidx.annotation.OptIn(UnstableApi::class)
                            val exoPlayer = remember(mediaUrl) {
                                FeedPlayerManager.getPlayer(
                                    url = mediaUrl,
                                    onBufferingChange = { buffering -> isBuffering = buffering },
                                    onError = { hasError = true }
                                )
                            }
                            
                            // Continuously poll player state to update buffering status
                            // This ensures we catch when player starts playing
                            // Only poll when visible (not off-screen) - major performance optimization
                            LaunchedEffect(exoPlayer, isVisible) {
                                if (!isVisible) return@LaunchedEffect // Skip polling for off-screen videos
                                
                                exoPlayer?.let { player ->
                                    while (isVisible) {
                                        // Update buffering based on actual player state
                                        val shouldBuffer = player.playbackState == Player.STATE_BUFFERING
                                        val isPlaying = player.isPlaying
                                        
                                        // Only show loading if truly buffering AND not playing
                                        isBuffering = shouldBuffer && !isPlaying
                                        
                                        // Sync with global mute state
                                        isMutedState = FeedPlayerManager.isMuted
                                        player.volume = if (FeedPlayerManager.isMuted) 0f else 1f
                                        
                                        delay(500) // Reduced from 100ms to 500ms for smoother scrolling
                                    }
                                }
                            }
                            
                            // Pause when scrolled away, DON'T release (keeps it cached)
                            DisposableEffect(mediaUrl) {
                                onDispose {
                                    // Just pause - FeedPlayerManager keeps player cached
                                    FeedPlayerManager.pausePlayer(mediaUrl)
                                }
                            }
                            
                            // Play/pause based on visibility AND NSFW status (like Instagram)
                            // CRITICAL: Stop video when content should be hidden/blurred for NSFW
                            LaunchedEffect(isVisible, exoPlayer, shouldBlurPost, shouldHidePost) {
                                exoPlayer?.let { player ->
                                    // Only play if visible AND not blocked by NSFW overlay
                                    val canPlay = isVisible && !shouldBlurPost && !shouldHidePost
                                    if (canPlay) {
                                        player.playWhenReady = true
                                    } else {
                                        player.playWhenReady = false
                                    }
                                }
                            }
                            
                            Box(modifier = Modifier.fillMaxSize()) {
                                if (!hasError && exoPlayer != null) {
                                    AndroidView(
                                        factory = { ctx ->
                                            PlayerView(ctx).apply {
                                                player = exoPlayer
                                                useController = false
                                                resizeMode = AspectRatioFrameLayout.RESIZE_MODE_ZOOM
                                                layoutParams = FrameLayout.LayoutParams(
                                                    ViewGroup.LayoutParams.MATCH_PARENT,
                                                    ViewGroup.LayoutParams.MATCH_PARENT
                                                )
                                            }
                                        },
                                        modifier = Modifier.fillMaxSize()
                                    )
                                }
                                
                                // Show loading indicator only when truly buffering
                                if (isBuffering && !hasError) {
                                    Box(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .background(Color.Black.copy(alpha = 0.7f)),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Column(
                                            horizontalAlignment = Alignment.CenterHorizontally,
                                            verticalArrangement = Arrangement.Center
                                        ) {
                                            CircularProgressIndicator(
                                                color = CyanGlow,
                                                strokeWidth = 4.dp,
                                                modifier = Modifier.size(56.dp)
                                            )
                                            Spacer(Modifier.height(12.dp))
                                            Text(
                                                text = "Loading...",
                                                color = Color.White.copy(alpha = 0.8f),
                                                fontSize = 12.sp,
                                                fontWeight = FontWeight.Medium
                                            )
                                        }
                                    }
                                }
                                
                                // Show error state
                                if (hasError) {
                                    Box(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .background(Color(0xFF1A1A2E)),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                            Icon(
                                                imageVector = Icons.Filled.VideocamOff,
                                                contentDescription = "Video error",
                                                tint = Color.White.copy(alpha = 0.5f),
                                                modifier = Modifier.size(48.dp)
                                            )
                                            Spacer(Modifier.height(8.dp))
                                            Text(
                                                "Video unavailable",
                                                color = Color.White.copy(alpha = 0.5f),
                                                fontSize = 12.sp
                                            )
                                        }
                                    }
                                }
                                
                                // Mute/Unmute button - bottom right corner
                                Box(
                                    modifier = Modifier
                                        .align(Alignment.BottomEnd)
                                        .padding(12.dp)
                                        .size(40.dp)
                                        .clip(CircleShape)
                                        .background(Color.Black.copy(alpha = 0.6f))
                                        .border(1.dp, Color.White.copy(alpha = 0.3f), CircleShape)
                                        .clickable { 
                                            // Use global mute toggle - persists across all videos
                                            FeedPlayerManager.toggleMute()
                                            isMutedState = FeedPlayerManager.isMuted
                                        },
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = if (isMutedState) 
                                            Icons.Filled.VolumeOff 
                                        else 
                                            Icons.Filled.VolumeUp,
                                        contentDescription = if (isMutedState) "Unmute" else "Mute",
                                        tint = Color.White,
                                        modifier = Modifier.size(20.dp)
                                    )
                                }
                            }
                        } else {
                            // Optimized Image display with shimmer loading
                            // Add logging for debugging
                            android.util.Log.d("PremiumPostCard", "Loading image: $mediaUrl")
                            
                            SubcomposeAsyncImage(
                                model = ImageRequest.Builder(context)
                                    .data(mediaUrl)
                                    .crossfade(300)
                                    .memoryCachePolicy(coil.request.CachePolicy.ENABLED)
                                    .diskCachePolicy(coil.request.CachePolicy.ENABLED)
                                    .networkCachePolicy(coil.request.CachePolicy.ENABLED)
                                    .placeholder(ColorDrawable(android.graphics.Color.parseColor("#1A1A2E")))
                                    .error(ColorDrawable(android.graphics.Color.parseColor("#1A1A2E")))
                                    .build(),
                                contentDescription = "Post media ${page + 1}",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Crop,
                                loading = {
                                    // Shimmer loading placeholder
                                    Box(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .background(CardBg)
                                    ) {
                                        ShimmerEffect(modifier = Modifier.fillMaxSize())
                                    }
                                },
                                error = {
                                    // Show error with URL for debugging
                                    android.util.Log.e("PremiumPostCard", "Failed to load: $mediaUrl")
                                    Box(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .background(Color(0xFF1A1A2E)),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                            Icon(
                                                imageVector = Icons.Filled.BrokenImage,
                                                contentDescription = "Error loading image",
                                                tint = Color.White.copy(alpha = 0.5f),
                                                modifier = Modifier.size(48.dp)
                                            )
                                            Spacer(Modifier.height(8.dp))
                                            Text(
                                                "Tap to retry",
                                                color = Color.White.copy(alpha = 0.5f),
                                                fontSize = 12.sp
                                            )
                                        }
                                    }
                                }
                            )
                        }
                    }
                    
                    // Pagination dots (only show if multiple media)
                    if (allMediaUrls.size > 1) {
                        Row(
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .padding(bottom = 16.dp)
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color.Black.copy(alpha = 0.5f))
                                .padding(horizontal = 10.dp, vertical = 6.dp),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            repeat(allMediaUrls.size) { index ->
                                Box(
                                    modifier = Modifier
                                        .size(if (index == currentPage) 8.dp else 6.dp)
                                        .clip(CircleShape)
                                        .background(
                                            if (index == currentPage) 
                                                Color.White 
                                            else 
                                                Color.White.copy(alpha = 0.4f)
                                        )
                                )
                            }
                        }
                        
                        // Page counter badge (top right)
                        Box(
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .padding(16.dp)
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color.Black.copy(alpha = 0.6f))
                                .padding(horizontal = 10.dp, vertical = 4.dp)
                        ) {
                            Text(
                                text = "${currentPage + 1}/${allMediaUrls.size}",
                                color = Color.White,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                    } // end if (!shouldBlurPost)
                } // end Box(modifier = Modifier.fillMaxSize())
            } else {
                // No media - show placeholder/content background
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            Brush.linearGradient(
                                colors = listOf(
                                    Color(0xFF1A1A2E),
                                    Color(0xFF252547)
                                )
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = post.content.take(100),
                        color = Color.White.copy(alpha = 0.8f),
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Medium,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(24.dp)
                    )
                }
            }

            // Shockwave Like Animation
            if (showLikeAnimation) {
                LikeShockwaveAnimation()
            }

            // Top User Info Bar
            UserInfoBar(
                username = post.username ?: "User",
                avatar = post.userAvatar,
                createdAt = post.createdAt,
                hasStatus = hasStatus,
                isOwner = isOwner,
                onUserClick = { onUserClick(post.userId) },
                onMoreClick = onMoreClick,
                onEdit = onEdit,
                onDelete = onDelete,
                onReport = onReport,
                onBlock = onBlock,
                onAvatarLongPress = onAvatarLongPress,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(16.dp)
            )
            // NSFW Blur overlay - clean design with gradient background
            if (shouldBlurPost) {
                val thumbnailUrl = post.mediaUrl ?: post.mediaUrls.firstOrNull()
                
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .clickable { showNSFWContent = true },
                    contentAlignment = Alignment.Center
                ) {
                    // Layer 1: Blurred/gradient background
                    if (thumbnailUrl != null) {
                        // Try to show blurred thumbnail
                        AsyncImage(
                            model = coil.request.ImageRequest.Builder(LocalContext.current)
                                .data(thumbnailUrl)
                                .size(60, 75) // Small size for blur effect
                                .crossfade(true)
                                .build(),
                            contentDescription = null,
                            modifier = Modifier
                                .fillMaxSize()
                                .blur(30.dp),
                            contentScale = ContentScale.Crop
                        )
                    }
                    
                    // Layer 2: Dark gradient overlay (always visible)
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                Brush.verticalGradient(
                                    colors = listOf(
                                        Color(0xFF1A1A2E).copy(alpha = 0.9f),
                                        Color(0xFF16162A),
                                        Color(0xFF1A1A2E).copy(alpha = 0.9f)
                                    )
                                )
                            )
                    )
                    
                    // Layer 3: Centered 18+ Content text ONLY (no header/menu)
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // Warning Icon with glow effect
                        Box(
                            modifier = Modifier
                                .size(80.dp)
                                .clip(CircleShape)
                                .background(
                                    Brush.radialGradient(
                                        colors = listOf(
                                            Color(0xFF6366F1).copy(alpha = 0.3f),
                                            Color.Transparent
                                        )
                                    )
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Filled.VisibilityOff,
                                contentDescription = null,
                                tint = Color.White,
                                modifier = Modifier.size(44.dp)
                            )
                        }
                        
                        Text(
                            text = "Sensitive Content",
                            color = Color.White,
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Click to See",
                            color = Color(0xFF6366F1),
                            fontSize = 16.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }
            }
        }
        } // End else block for shouldHidePost

        Spacer(modifier = Modifier.height(16.dp))

        // Action Capsule + Bookmark
        ActionCapsuleRow(
            post = post,
            isLiked = isLiked,
            isSaved = isSaved,
            onLike = onLike,
            onComment = onComment,
            onShare = onShare,
            onSave = onSave
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Caption
        CaptionSection(
            username = post.username ?: "User",
            caption = post.content,
            timeAgo = formatTimeAgo(post.createdAt)
        )
    }
}

@Composable
private fun LikeShockwaveAnimation() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        // Outer ring 1
        val ring1Scale by rememberInfiniteTransition(label = "ring1").animateFloat(
            initialValue = 0f,
            targetValue = 4f,
            animationSpec = infiniteRepeatable(
                animation = tween(700, easing = FastOutSlowInEasing),
                repeatMode = RepeatMode.Restart
            ),
            label = "ring1Scale"
        )
        val ring1Alpha by rememberInfiniteTransition(label = "ring1Alpha").animateFloat(
            initialValue = 0.6f,
            targetValue = 0f,
            animationSpec = infiniteRepeatable(
                animation = tween(700, easing = FastOutSlowInEasing),
                repeatMode = RepeatMode.Restart
            ),
            label = "ring1Alpha"
        )
        
        Box(
            modifier = Modifier
                .size(48.dp)
                .scale(ring1Scale)
                .border(2.dp, Color.White.copy(alpha = ring1Alpha), CircleShape)
        )

        // Center Heart
        Icon(
            imageVector = Icons.Filled.Favorite,
            contentDescription = null,
            tint = Color.White,
            modifier = Modifier
                .size(48.dp)
                .scale(1.2f)
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun UserInfoBar(
    username: String,
    avatar: String?,
    createdAt: String,
    hasStatus: Boolean,
    isOwner: Boolean,
    onUserClick: () -> Unit,
    onMoreClick: () -> Unit,
    onEdit: () -> Unit,
    onDelete: () -> Unit,
    onReport: () -> Unit,
    onBlock: () -> Unit,
    onAvatarLongPress: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // User Avatar + Name Pill - CLICKABLE to navigate to profile
        Row(
            modifier = Modifier
                .clip(RoundedCornerShape(24.dp))
                .background(Color.Black.copy(alpha = 0.3f))
                .border(1.dp, BorderWhite10, RoundedCornerShape(24.dp))
                .combinedClickable(
                    onClick = { onUserClick() },
                    onLongClick = { onAvatarLongPress() }
                )
                .padding(start = 6.dp, end = 12.dp, top = 6.dp, bottom = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Avatar with Spinning Neon Ring (if has status)
            Box {
                if (hasStatus) {
                    SpinningNeonRing()
                }
                
                AsyncImage(
                    model = ImageRequest.Builder(LocalContext.current)
                        .data(avatar)
                        .crossfade(200)
                        .build(),
                    contentDescription = null,
                    modifier = Modifier
                        .size(32.dp)
                        .clip(CircleShape)
                        .border(2.dp, Color.Black, CircleShape),
                    contentScale = ContentScale.Crop
                )
                
                if (hasStatus) {
                    // Pulsing status dot
                    Box(
                        modifier = Modifier
                            .align(Alignment.BottomEnd)
                            .size(10.dp)
                            .clip(CircleShape)
                            .background(CyanGlow)
                            .border(2.dp, Color.Black, CircleShape)
                    )
                }
            }
            
            // Username only (no time)
            Text(
                text = username,
                color = Color.White,
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold
            )
        }

        // More Button with Animated Popup Menu
        var showMenu by remember { mutableStateOf(false) }
        var showReportDialog by remember { mutableStateOf(false) }
        var showBlockDialog by remember { mutableStateOf(false) }
        val menuScale by animateFloatAsState(
            targetValue = if (showMenu) 1f else 0.8f,
            animationSpec = spring(dampingRatio = 0.6f, stiffness = 400f),
            label = "menuScale"
        )
        
        Box {
            // 3-Dot Button with press animation
            var isPressed by remember { mutableStateOf(false) }
            val buttonScale by animateFloatAsState(
                targetValue = if (isPressed) 0.9f else 1f,
                animationSpec = spring(dampingRatio = 0.5f, stiffness = 500f),
                label = "buttonScale"
            )
            
            IconButton(
                onClick = { showMenu = true },
                modifier = Modifier
                    .size(36.dp)
                    .scale(buttonScale)
                    .clip(CircleShape)
                    .background(Color.Black.copy(alpha = 0.3f)) // Glassy transparent
                    .border(1.dp, Color.White.copy(alpha = 0.15f), CircleShape)
                    .pointerInput(Unit) {
                        detectTapGestures(
                            onPress = {
                                isPressed = true
                                tryAwaitRelease()
                                isPressed = false
                            }
                        )
                    }
            ) {
                Icon(
                    imageVector = Icons.Filled.MoreVert,
                    contentDescription = "More options",
                    tint = Color.White.copy(alpha = 0.9f),
                    modifier = Modifier.size(18.dp)
                )
            }
            
            MaterialTheme(
                colorScheme = MaterialTheme.colorScheme.copy(
                    surface = Color.Transparent,
                    surfaceContainer = Color(0xFF0A0A12)
                ),
                shapes = MaterialTheme.shapes.copy(
                    extraSmall = RoundedCornerShape(16.dp)
                )
            ) {
                DropdownMenu(
                    expanded = showMenu,
                    onDismissRequest = { showMenu = false },
                    modifier = Modifier
                        .scale(menuScale)
                        .widthIn(min = 200.dp, max = 240.dp)
                        .background(
                            color = Color(0xFF0A0A12),
                            shape = RoundedCornerShape(16.dp)
                        )
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(
                                    Color(0xFF1A1A35),
                                    Color(0xFF121225),
                                    Color(0xFF0A0A12)
                                )
                            ),
                            shape = RoundedCornerShape(16.dp)
                        )
                        .border(
                            width = 1.dp,
                            brush = Brush.linearGradient(
                                colors = listOf(
                                    Color(0xFF6366F1).copy(alpha = 0.4f),
                                    Color(0xFFEC4899).copy(alpha = 0.25f),
                                    Color(0xFF8B5CF6).copy(alpha = 0.35f)
                                )
                            ),
                            shape = RoundedCornerShape(16.dp)
                        )
                        .padding(vertical = 6.dp, horizontal = 4.dp)
                ) {
                if (isOwner) {
                    // Owner options - Edit and Delete with hover effect
                    DropdownMenuItem(
                        text = {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.padding(vertical = 4.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(32.dp)
                                        .clip(CircleShape)
                                        .background(Color(0xFF6366F1).copy(alpha = 0.2f)),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.Outlined.Edit,
                                        contentDescription = null,
                                        tint = Color(0xFF818CF8),
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                                Spacer(Modifier.width(12.dp))
                                Column {
                                    Text(
                                        "Edit Post",
                                        color = Color.White,
                                        fontWeight = FontWeight.Medium,
                                        fontSize = 14.sp
                                    )
                                    Text(
                                        "Modify your post",
                                        color = Color.White.copy(alpha = 0.5f),
                                        fontSize = 11.sp
                                    )
                                }
                            }
                        },
                        onClick = {
                            showMenu = false
                            onEdit()
                        }
                    )
                    
                    // Divider
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(1.dp)
                            .padding(horizontal = 16.dp)
                            .background(Color.White.copy(alpha = 0.08f))
                    )
                    
                    DropdownMenuItem(
                        text = {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.padding(vertical = 4.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(32.dp)
                                        .clip(CircleShape)
                                        .background(Color(0xFFEF4444).copy(alpha = 0.2f)),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.Outlined.Delete,
                                        contentDescription = null,
                                        tint = Color(0xFFF87171),
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                                Spacer(Modifier.width(12.dp))
                                Column {
                                    Text(
                                        "Delete Post",
                                        color = Color(0xFFF87171),
                                        fontWeight = FontWeight.Medium,
                                        fontSize = 14.sp
                                    )
                                    Text(
                                        "Remove permanently",
                                        color = Color(0xFFEF4444).copy(alpha = 0.6f),
                                        fontSize = 11.sp
                                    )
                                }
                            }
                        },
                        onClick = {
                            showMenu = false
                            onDelete()
                        }
                    )
                } else {
                    // Non-owner options - Report compact
                    DropdownMenuItem(
                        text = {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.padding(vertical = 4.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(34.dp)
                                        .clip(RoundedCornerShape(10.dp))
                                        .background(
                                            Brush.radialGradient(
                                                colors = listOf(
                                                    Color(0xFFFBBF24).copy(alpha = 0.3f),
                                                    Color(0xFFF59E0B).copy(alpha = 0.15f)
                                                )
                                            )
                                        ),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.Outlined.Flag,
                                        contentDescription = null,
                                        tint = Color(0xFFFCD34D),
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                                Spacer(Modifier.width(12.dp))
                                Column {
                                    Text(
                                        "Report Post",
                                        color = Color.White,
                                        fontWeight = FontWeight.Medium,
                                        fontSize = 14.sp
                                    )
                                    Text(
                                        "Flag content",
                                        color = Color.White.copy(alpha = 0.45f),
                                        fontSize = 11.sp
                                    )
                                }
                            }
                        },
                        onClick = {
                            showMenu = false
                            showReportDialog = true
                        }
                    )
                    
                    // Divider line
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp, vertical = 4.dp)
                            .height(1.dp)
                            .background(
                                Brush.horizontalGradient(
                                    colors = listOf(
                                        Color.Transparent,
                                        Color.White.copy(alpha = 0.1f),
                                        Color.Transparent
                                    )
                                )
                            )
                    )
                    
                    // Block User - compact
                    DropdownMenuItem(
                        text = {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.padding(vertical = 4.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(34.dp)
                                        .clip(RoundedCornerShape(10.dp))
                                        .background(
                                            Brush.radialGradient(
                                                colors = listOf(
                                                    Color(0xFFEF4444).copy(alpha = 0.3f),
                                                    Color(0xFFDC2626).copy(alpha = 0.15f)
                                                )
                                            )
                                        ),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.Outlined.Block,
                                        contentDescription = null,
                                        tint = Color(0xFFF87171),
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                                Spacer(Modifier.width(12.dp))
                                Column {
                                    Text(
                                        "Block User",
                                        color = Color(0xFFF87171),
                                        fontWeight = FontWeight.Medium,
                                        fontSize = 14.sp
                                    )
                                    Text(
                                        "Hide content",
                                        color = Color(0xFFEF4444).copy(alpha = 0.45f),
                                        fontSize = 11.sp
                                    )
                                }
                            }
                        },
                        onClick = {
                            showMenu = false
                            showBlockDialog = true
                        }
                    )
                }
            }
            } // Close MaterialTheme
        }
        
        // Report Confirmation Dialog
        if (showReportDialog) {
            AlertDialog(
                onDismissRequest = { showReportDialog = false },
                containerColor = Color(0xFF1A1A2E),
                titleContentColor = Color.White,
                textContentColor = Color.White.copy(alpha = 0.7f),
                shape = RoundedCornerShape(20.dp),
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color(0xFFFBBF24).copy(alpha = 0.2f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                Icons.Outlined.Flag,
                                contentDescription = null,
                                tint = Color(0xFFFCD34D),
                                modifier = Modifier.size(22.dp)
                            )
                        }
                        Spacer(Modifier.width(12.dp))
                        Text("Report Post", fontWeight = FontWeight.Bold)
                    }
                },
                text = {
                    Text("Are you sure you want to report this post? Our team will review it for any violations.")
                },
                confirmButton = {
                    Button(
                        onClick = {
                            showReportDialog = false
                            onReport()
                        },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFFFBBF24)
                        ),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("Report", color = Color.Black, fontWeight = FontWeight.SemiBold)
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showReportDialog = false }) {
                        Text("Cancel", color = Color.White.copy(alpha = 0.7f))
                    }
                }
            )
        }
        
        // Block Confirmation Dialog
        if (showBlockDialog) {
            AlertDialog(
                onDismissRequest = { showBlockDialog = false },
                containerColor = Color(0xFF1A1A2E),
                titleContentColor = Color.White,
                textContentColor = Color.White.copy(alpha = 0.7f),
                shape = RoundedCornerShape(20.dp),
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color(0xFFEF4444).copy(alpha = 0.2f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                Icons.Outlined.Block,
                                contentDescription = null,
                                tint = Color(0xFFF87171),
                                modifier = Modifier.size(22.dp)
                            )
                        }
                        Spacer(Modifier.width(12.dp))
                        Text("Block User", fontWeight = FontWeight.Bold)
                    }
                },
                text = {
                    Text("Are you sure you want to block this user? You won't see their posts or messages anymore.")
                },
                confirmButton = {
                    Button(
                        onClick = {
                            showBlockDialog = false
                            onBlock()
                        },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFFEF4444)
                        ),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("Block", color = Color.White, fontWeight = FontWeight.SemiBold)
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showBlockDialog = false }) {
                        Text("Cancel", color = Color.White.copy(alpha = 0.7f))
                    }
                }
            )
        }
    }
}

@Composable
private fun SpinningNeonRing() {
    val infiniteTransition = rememberInfiniteTransition(label = "spin")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(3000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "rotation"
    )

    Box(
        modifier = Modifier
            .size(38.dp)
            .offset(x = (-3).dp, y = (-3).dp)
            .rotate(rotation)
            .clip(CircleShape)
            .background(
                Brush.sweepGradient(
                    colors = listOf(
                        Color.Transparent,
                        CyanGlow,
                        Color(0xFF3B82F6) // Blue
                    )
                )
            )
    )
}

@Composable
private fun ActionCapsuleRow(
    post: Post,
    isLiked: Boolean,
    isSaved: Boolean,
    onLike: () -> Unit,
    onComment: () -> Unit,
    onShare: () -> Unit,
    onSave: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Action Capsule
        Row(
            modifier = Modifier
                .clip(RoundedCornerShape(28.dp))
                .background(CardBg)
                .border(1.dp, BorderWhite10, RoundedCornerShape(28.dp))
                .padding(6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Like Button + Count
            Row(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(if (isLiked) Color.White.copy(alpha = 0.1f) else Color.Transparent)
                    .clickable { onLike() }
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Icon(
                    imageVector = if (isLiked) Icons.Filled.Favorite else Icons.Outlined.FavoriteBorder,
                    contentDescription = "Like",
                    tint = if (isLiked) Color(0xFFEF4444) else Color(0xFFE4E4E7),
                    modifier = Modifier.size(22.dp)
                )
                Text(
                    text = formatCount(post.likesCount),
                    color = if (isLiked) Color.White else Color(0xFFA1A1AA),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            // Divider
            Box(
                modifier = Modifier
                    .width(1.5.dp)
                    .height(24.dp)
                    .background(Color(0xFF3F3F46).copy(alpha = 0.8f))
            )

            Spacer(modifier = Modifier.width(8.dp))

            // Comment
            Row(
                modifier = Modifier
                    .clickable { onComment() }
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Icon(
                    imageVector = Icons.Outlined.ChatBubbleOutline,
                    contentDescription = "Comment",
                    tint = Color(0xFFA1A1AA),
                    modifier = Modifier.size(22.dp)
                )
                Text(
                    text = formatCount(post.commentsCount),
                    color = Color(0xFFA1A1AA),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            // Share
            Row(
                modifier = Modifier
                    .clickable { onShare() }
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Icon(
                    imageVector = Icons.Outlined.Share,
                    contentDescription = "Share",
                    tint = Color(0xFFA1A1AA),
                    modifier = Modifier.size(22.dp)
                )
                Text(
                    text = formatCount(post.sharesCount),
                    color = Color(0xFFA1A1AA),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            // Views
            Row(
                modifier = Modifier
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Icon(
                    imageVector = Icons.Outlined.Visibility,
                    contentDescription = "Views",
                    tint = Color(0xFFA1A1AA),
                    modifier = Modifier.size(22.dp)
                )
                Text(
                    text = formatCount(post.viewsCount),
                    color = Color(0xFFA1A1AA),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        // Detached Bookmark Circle (Save)
        Box(
            modifier = Modifier
                .size(44.dp)
                .clip(CircleShape)
                .background(if (isSaved) IndigoAccent.copy(alpha = 0.1f) else CardBg)
                .border(
                    1.dp,
                    if (isSaved) IndigoAccent.copy(alpha = 0.3f) else BorderWhite10,
                    CircleShape
                )
                .clickable { onSave() },
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = if (isSaved) Icons.Filled.Bookmark else Icons.Outlined.BookmarkBorder,
                contentDescription = "Save",
                tint = if (isSaved) IndigoAccent else Color(0xFFA1A1AA),
                modifier = Modifier.size(20.dp)
            )
        }
    }
}

@Composable
private fun CaptionSection(
    username: String,
    caption: String,
    timeAgo: String
) {
    Column(
        modifier = Modifier.padding(horizontal = 4.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        // Caption Text
        Text(
            text = buildString {
                append(username)
                append("  ")
                append(caption)
            },
            color = Color.White.copy(alpha = 0.9f),
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium,
            lineHeight = 20.sp,
            maxLines = 3,
            overflow = TextOverflow.Ellipsis
        )

        // Time
        Text(
            text = timeAgo.uppercase(),
            color = Color(0xFF71717A),
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            letterSpacing = 1.sp
        )
    }
}

// Helper functions
private fun formatCount(count: Int): String {
    return when {
        count >= 1_000_000 -> "${count / 1_000_000}M"
        count >= 1_000 -> "${count / 1_000}k"
        else -> count.toString()
    }
}
