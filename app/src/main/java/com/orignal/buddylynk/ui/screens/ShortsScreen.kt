package com.orignal.buddylynk.ui.screens

import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.pager.*
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
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.datasource.cache.CacheDataSource
import androidx.media3.datasource.cache.SimpleCache
import androidx.media3.datasource.cache.LeastRecentlyUsedCacheEvictor
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.database.StandaloneDatabaseProvider
import androidx.media3.ui.PlayerView
import coil.compose.AsyncImage
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.data.cache.VideoPlayerCache
import com.orignal.buddylynk.ui.viewmodel.HomeViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File

// Premium Gradient Colors
private val IndigoAccent = Color(0xFF6366F1)
private val PurpleAccent = Color(0xFF8B5CF6)
private val PinkAccent = Color(0xFFEC4899)
private val RedAccent = Color(0xFFEF4444)
private val OrangeAccent = Color(0xFFF97316)
private val EmeraldAccent = Color(0xFF10B981)
private val TealAccent = Color(0xFF14B8A6)
private val CyanAccent = Color(0xFF22D3EE)

// Gradient Presets for Different Moods
private val gradientPresets = listOf(
    listOf(Color(0xFF6366F1), Color(0xFF8B5CF6)),
    listOf(Color(0xFFEF4444), Color(0xFFF97316)),
    listOf(Color(0xFF10B981), Color(0xFF14B8A6)),
    listOf(Color(0xFFEC4899), Color(0xFFF43F5E)),
    listOf(Color(0xFF3B82F6), Color(0xFF6366F1))
)

/**
 * Production-level video player manager with pre-loading
 * - Pre-loads next 2 videos while current plays
 * - Reuses ExoPlayer instances with LRU eviction
 * - Aggressive buffer settings for instant playback
 */
@androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
class ShortsPlayerManager(private val context: android.content.Context) {
    // LinkedHashMap with access order for LRU eviction
    private val players = object : LinkedHashMap<String, ExoPlayer>(8, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, ExoPlayer>?): Boolean {
            if (size > MAX_PLAYERS) {
                eldest?.value?.release()
                return true
            }
            return false
        }
    }
    
    companion object {
        const val MAX_PLAYERS = 5 // Keep 5 players (current + 2 next + 2 previous)
    }
    
    private fun createPlayer(): ExoPlayer {
        val loadControl = androidx.media3.exoplayer.DefaultLoadControl.Builder()
            .setBufferDurationsMs(
                1500,   // Min buffer - very short for instant start
                30000,  // Max buffer - larger for smooth playback
                300,    // Buffer for playback - start almost immediately
                800     // Buffer for rebuffer
            )
            .setPrioritizeTimeOverSizeThresholds(true)
            .build()
        
        // Use cached data source for disk caching (videos won't re-download)
        val cacheDataSourceFactory = VideoPlayerCache.getCacheDataSourceFactory(context)
        
        return ExoPlayer.Builder(context)
            .setLoadControl(loadControl)
            .setMediaSourceFactory(DefaultMediaSourceFactory(cacheDataSourceFactory))
            .build()
    }
    
    fun getPlayer(url: String): ExoPlayer {
        // Reuse existing player if available (this also updates LRU order)
        players[url]?.let { 
            // Reset to start if it was previously played
            if (it.currentPosition > 0) {
                it.seekTo(0)
            }
            return it 
        }
        
        // Create new player and prepare
        val player = createPlayer()
        player.setMediaItem(MediaItem.fromUri(url))
        player.prepare()
        players[url] = player
        return player
    }
    
    // Pre-load videos in background (call this for next/previous videos)
    fun preloadVideo(url: String) {
        if (players.containsKey(url)) return
        
        val player = createPlayer()
        player.setMediaItem(MediaItem.fromUri(url))
        player.prepare()
        player.playWhenReady = false // Don't auto-play, just buffer
        players[url] = player
    }
    
    fun releaseAll() {
        players.values.forEach { it.release() }
        players.clear()
    }
    
    fun release(url: String) {
        players[url]?.release()
        players.remove(url)
    }
}

/**
 * Premium Shorts Screen - TikTok/Instagram Reels style
 * Converted from React design with holographic card UI
 */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun ShortsScreen(
    viewModel: HomeViewModel = viewModel(),
    onNavigateBack: () -> Unit = {},
    onNavigateToProfile: (String) -> Unit = {},
    onNavigateToComments: (String) -> Unit = {}
) {
    val posts by viewModel.posts.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    // Production-level player manager with pre-loading
    val playerManager = remember { ShortsPlayerManager(context) }
    
    // Clean up all players when leaving screen
    DisposableEffect(Unit) {
        onDispose {
            playerManager.releaseAll()
        }
    }
    
    // Keep screen awake while watching shorts (like YouTube/TikTok)
    val activity = context as? android.app.Activity
    DisposableEffect(Unit) {
        activity?.window?.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            activity?.window?.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }
    
    // Filter posts to only show VIDEOS (not images) for shorts
    val videoPosts = posts.filter { post ->
        post.mediaUrl != null && (
            post.mediaType == "video" ||
            post.mediaUrl!!.contains(".mp4", ignoreCase = true) ||
            post.mediaUrl!!.contains(".webm", ignoreCase = true) ||
            post.mediaUrl!!.contains(".mov", ignoreCase = true)
        )
    }
    val pagerState = rememberPagerState(pageCount = { if (videoPosts.isEmpty()) 1 else videoPosts.size })
    
    // States
    var isMuted by remember { mutableStateOf(false) }
    var showVolumeIndicator by remember { mutableStateOf(false) }
    var isAnimating by remember { mutableStateOf(false) }
    val likedPosts = remember { mutableStateListOf<String>() }
    val savedPosts = remember { mutableStateListOf<String>() }
    
    // Progress bar animation
    var progress by remember { mutableStateOf(0f) }
    
    // Pre-load next videos when page changes
    LaunchedEffect(pagerState.currentPage, videoPosts) {
        if (videoPosts.isEmpty()) return@LaunchedEffect
        
        progress = 0f
        
        // Pre-load next 2 videos in background
        val currentIndex = pagerState.currentPage
        listOf(currentIndex + 1, currentIndex + 2).forEach { nextIndex ->
            if (nextIndex < videoPosts.size) {
                videoPosts[nextIndex].mediaUrl?.let { url ->
                    playerManager.preloadVideo(url)
                }
            }
        }
        
        // Also pre-load previous video for smooth back-swipe
        if (currentIndex > 0) {
            videoPosts[currentIndex - 1].mediaUrl?.let { url ->
                playerManager.preloadVideo(url)
            }
        }
        
        // Progress animation
        while (true) {
            delay(50)
            progress = if (progress >= 100f) 0f else progress + 0.5f
        }
    }
    
    LaunchedEffect(Unit) {
        viewModel.loadPosts()
    }
    
    // Current gradient based on page
    val currentGradient = gradientPresets[pagerState.currentPage % gradientPresets.size]
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF050505))
    ) {
        // --- Dynamic Ambient Background ---
        DynamicAmbientBackground(
            colors = currentGradient,
            isAnimating = isAnimating
        )
        
        // Main Content
        if (isLoading && videoPosts.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(color = IndigoAccent)
            }
        } else if (videoPosts.isEmpty()) {
            EmptyShortsStat()
        } else {
            // Fullscreen Shorts Container - no nav bar visible
            Box(
                modifier = Modifier
                    .fillMaxSize()
                // No bottom padding needed - nav bar is hidden
            ) {
                VerticalPager(
                    state = pagerState,
                    modifier = Modifier.fillMaxSize()
                ) { page ->
                    val post = videoPosts[page]
                    val isCurrentPage = pagerState.currentPage == page
                    
                    HolographicShortCard(
                        post = post,
                        playerManager = playerManager,
                        isCurrentPage = isCurrentPage,
                        isMuted = isMuted,
                        isLiked = likedPosts.contains(post.postId),
                        isSaved = savedPosts.contains(post.postId),
                        progress = if (isCurrentPage) progress else 0f,
                        isAnimating = isAnimating,
                        onToggleMute = { 
                            isMuted = !isMuted
                            showVolumeIndicator = true
                            scope.launch {
                                delay(800)
                                showVolumeIndicator = false
                            }
                        },
                        onLike = {
                            if (likedPosts.contains(post.postId)) {
                                likedPosts.remove(post.postId)
                            } else {
                                likedPosts.add(post.postId)
                            }
                            viewModel.likePost(post.postId)
                        },
                        onSave = {
                            if (savedPosts.contains(post.postId)) {
                                savedPosts.remove(post.postId)
                            } else {
                                savedPosts.add(post.postId)
                            }
                            viewModel.bookmarkPost(post.postId)
                        },
                        onComment = { onNavigateToComments(post.postId) },
                        onShare = { /* Share */ },
                        onProfileClick = { onNavigateToProfile(post.userId) },
                        showVolumeIndicator = showVolumeIndicator && isCurrentPage
                    )
                }
            }
        }
        
        // Top Header with Back Arrow
        ShortsTopHeader(
            onNavigateBack = onNavigateBack,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .statusBarsPadding()
        )
    }
}

/**
 * Dynamic Ambient Background - Changes color based on content
 */
@Composable
private fun DynamicAmbientBackground(
    colors: List<Color>,
    isAnimating: Boolean
) {
    val animatedAlpha by animateFloatAsState(
        targetValue = if (isAnimating) 0.1f else 0.2f,
        animationSpec = tween(1000),
        label = "bgAlpha"
    )
    
    Box(modifier = Modifier.fillMaxSize()) {
        // Gradient Blur Layer
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.radialGradient(
                        colors = listOf(
                            colors[0].copy(alpha = animatedAlpha),
                            colors[1].copy(alpha = animatedAlpha * 0.5f),
                            Color.Transparent
                        )
                    )
                )
                .blur(150.dp)
        )
        
        // Dark Overlay
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.4f))
        )
    }
}

/**
 * Holographic Short Card - The main video card with all UI elements
 */
@Composable
private fun HolographicShortCard(
    post: Post,
    playerManager: ShortsPlayerManager,
    isCurrentPage: Boolean,
    isMuted: Boolean,
    isLiked: Boolean,
    isSaved: Boolean,
    progress: Float,
    isAnimating: Boolean,
    showVolumeIndicator: Boolean,
    onToggleMute: () -> Unit,
    onLike: () -> Unit,
    onSave: () -> Unit,
    onComment: () -> Unit,
    onShare: () -> Unit,
    onProfileClick: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var showLikeAnimation by remember { mutableStateOf(false) }
    
    // Card animation
    val cardScale by animateFloatAsState(
        targetValue = if (isAnimating) 0.9f else 1f,
        animationSpec = spring(dampingRatio = 0.6f, stiffness = 300f),
        label = "cardScale"
    )
    val cardAlpha by animateFloatAsState(
        targetValue = if (isAnimating) 0.5f else 1f,
        animationSpec = tween(300),
        label = "cardAlpha"
    )
    
    // Video loading state - start with FALSE for instant display
    var isVideoLoading by remember { mutableStateOf(false) }
    var videoProgress by remember { mutableStateOf(0f) }
    
    // Get player from manager (reuses pre-loaded player if available)
    val exoPlayer = remember(post.mediaUrl) {
        post.mediaUrl?.let { url ->
            playerManager.getPlayer(url).apply {
                repeatMode = Player.REPEAT_MODE_ONE
            }
        }
    }
    
    // Continuously poll player state to track loading (like Instagram)
    // This ensures we only show loading when actually buffering
    LaunchedEffect(exoPlayer, isCurrentPage) {
        exoPlayer?.let { player ->
            while (isCurrentPage) {
                // Only show loading if truly buffering AND not playing
                val shouldShowLoading = player.playbackState == Player.STATE_BUFFERING && !player.isPlaying
                isVideoLoading = shouldShowLoading
                delay(100)
            }
        }
    }
    
    // Track video progress for timeline
    LaunchedEffect(isCurrentPage, exoPlayer) {
        while (isCurrentPage && exoPlayer != null) {
            val duration = exoPlayer.duration
            val position = exoPlayer.currentPosition
            if (duration > 0) {
                videoProgress = (position.toFloat() / duration.toFloat()) * 100f
            }
            delay(100) // Update every 100ms
        }
    }
    
    LaunchedEffect(isCurrentPage, isMuted) {
        exoPlayer?.let { player ->
            if (isCurrentPage) {
                player.playWhenReady = true
                player.volume = if (isMuted) 0f else 1f
            } else {
                // Pause when not current (don't release - manager handles that)
                player.playWhenReady = false
                player.seekTo(0)
                player.volume = 0f
            }
        }
    }
    
    // Main Fullscreen Card
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .pointerInput(Unit) {
                detectTapGestures(
                    onTap = { onToggleMute() },
                    onDoubleTap = {
                        if (!isLiked) onLike()
                        showLikeAnimation = true
                        scope.launch {
                            delay(1000)
                            showLikeAnimation = false
                        }
                    }
                )
            }
    ) {
        // Always show thumbnail/poster as background (uses first frame or avatar)
        AsyncImage(
            model = post.userAvatar ?: post.mediaUrl,
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )
        
        // Video player on top (only visible when loaded)
        if (post.mediaType == "video" || post.mediaUrl?.contains(".mp4") == true || 
            post.mediaUrl?.contains(".webm") == true || post.mediaUrl?.contains(".mov") == true) {
            
            // Show video when loaded
            if (!isVideoLoading) {
                AndroidView(
                    factory = { ctx ->
                        PlayerView(ctx).apply {
                            player = exoPlayer
                            useController = false
                            layoutParams = FrameLayout.LayoutParams(
                                ViewGroup.LayoutParams.MATCH_PARENT,
                                ViewGroup.LayoutParams.MATCH_PARENT
                            )
                        }
                    },
                    modifier = Modifier.fillMaxSize()
                )
            }
            
            // Loading indicator while video buffers
            if (isVideoLoading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(
                        color = IndigoAccent,
                        modifier = Modifier.size(48.dp)
                    )
                }
            }
        }
        
        // Dark Gradient Overlays
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Color.Black.copy(alpha = 0.3f),
                            Color.Transparent,
                            Color.Black.copy(alpha = 0.8f)
                        )
                    )
                )
        )
        
        // Center Volume Indicator HUD
        AnimatedVisibility(
            visible = showVolumeIndicator,
            enter = scaleIn(initialScale = 0.5f) + fadeIn(),
            exit = scaleOut(targetScale = 0.5f) + fadeOut(),
            modifier = Modifier.align(Alignment.Center)
        ) {
            VolumeIndicatorHUD(isMuted = isMuted)
        }
        
        // Center Like Animation
        AnimatedVisibility(
            visible = showLikeAnimation,
            enter = scaleIn(initialScale = 0.5f) + fadeIn(),
            exit = scaleOut(targetScale = 1.5f) + fadeOut(),
            modifier = Modifier.align(Alignment.Center)
        ) {
            Icon(
                imageVector = Icons.Filled.Favorite,
                contentDescription = null,
                tint = Color.Red,
                modifier = Modifier.size(120.dp)
            )
        }
        
        // Right Side HUD (Holographic Strip)
        RightSideHUD(
            post = post,
            isLiked = isLiked,
            isSaved = isSaved,
            onLike = onLike,
            onSave = onSave,
            onComment = onComment,
            onShare = onShare,
            onProfileClick = onProfileClick,
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(end = 12.dp, bottom = 60.dp)
        )
        
        // Bottom Info Section
        BottomInfoSection(
            post = post,
            onUserClick = onProfileClick,
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(start = 20.dp, bottom = 50.dp, end = 80.dp) // Above progress bar
        )
        
        // Progress Bar - visible above system nav bar
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 24.dp) // Space above system navigation
                .padding(horizontal = 16.dp)
                .fillMaxWidth()
                .height(5.dp)
                .clip(RoundedCornerShape(3.dp))
                .background(Color.White.copy(alpha = 0.3f))
        ) {
            // Progress fill
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(videoProgress / 100f)
                    .clip(RoundedCornerShape(3.dp))
                    .background(
                        Brush.horizontalGradient(
                            colors = listOf(IndigoAccent, PurpleAccent, PinkAccent)
                        )
                    )
            )
        }
    }
}

/**
 * Volume Indicator HUD - Shows mute/unmute status
 */
@Composable
private fun VolumeIndicatorHUD(isMuted: Boolean) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(24.dp))
            .background(Color.Black.copy(alpha = 0.3f))
            .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(24.dp))
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (isMuted) {
                Icon(
                    imageVector = Icons.Filled.VolumeOff,
                    contentDescription = null,
                    tint = Color.Gray,
                    modifier = Modifier.size(40.dp)
                )
                Text(
                    text = "MUTED",
                    color = Color.Gray,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp
                )
            } else {
                Box {
                    Icon(
                        imageVector = Icons.Filled.VolumeUp,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(40.dp)
                    )
                    // Sound wave indicator
                    Icon(
                        imageVector = Icons.Filled.GraphicEq,
                        contentDescription = null,
                        tint = IndigoAccent,
                        modifier = Modifier
                            .size(20.dp)
                            .align(Alignment.TopEnd)
                            .offset(x = 8.dp, y = (-8).dp)
                    )
                }
                Text(
                    text = "SOUND ON",
                    color = IndigoAccent.copy(alpha = 0.8f),
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp
                )
            }
        }
    }
}

/**
 * Right Side HUD - Action buttons in holographic strip
 */
@Composable
private fun RightSideHUD(
    post: Post,
    isLiked: Boolean,
    isSaved: Boolean,
    onLike: () -> Unit,
    onSave: () -> Unit,
    onComment: () -> Unit,
    onShare: () -> Unit,
    onProfileClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    // Spinning vinyl rotation
    val infiniteTransition = rememberInfiniteTransition(label = "vinyl")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "vinylRotation"
    )
    
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        // Holographic Control Bar
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(30.dp))
                .background(Color.Black.copy(alpha = 0.3f))
                .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(30.dp))
                .padding(horizontal = 8.dp, vertical = 16.dp)
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Like Button
                ShortActionButton(
                    icon = if (isLiked) Icons.Filled.Favorite else Icons.Outlined.FavoriteBorder,
                    count = formatCount(post.likesCount),
                    isActive = isLiked,
                    activeColor = Color.Red,
                    onClick = onLike
                )
                
                // Comment Button
                ShortActionButton(
                    icon = Icons.Outlined.ChatBubbleOutline,
                    count = formatCount(post.commentsCount),
                    onClick = onComment
                )
                
                // Share Button
                ShortActionButton(
                    icon = Icons.Outlined.Share,
                    count = "Share",
                    onClick = onShare
                )
                
                // Save/Bookmark Button
                ShortActionButton(
                    icon = if (isSaved) Icons.Filled.Bookmark else Icons.Outlined.BookmarkBorder,
                    count = formatCount(post.viewsCount),
                    isActive = isSaved,
                    activeColor = IndigoAccent,
                    onClick = onSave
                )
            }
        }
        
        Spacer(modifier = Modifier.height(8.dp))
        
        // Spinning Vinyl (Audio Indicator)
        Box(
            modifier = Modifier
                .size(48.dp)
                .rotate(rotation)
                .clip(CircleShape)
                .background(Color.Black.copy(alpha = 0.5f))
                .border(4.dp, Color(0xFF27272A), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            AsyncImage(
                model = post.userAvatar,
                contentDescription = null,
                modifier = Modifier
                    .fillMaxSize()
                    .clip(CircleShape)
                    .graphicsLayer { alpha = 0.8f },
                contentScale = ContentScale.Crop
            )
            // Center hole
            Box(
                modifier = Modifier
                    .size(16.dp)
                    .clip(CircleShape)
                    .background(Color.Black)
                    .border(1.dp, Color(0xFF3F3F46), CircleShape)
            )
        }
    }
}

/**
 * Short Action Button - Individual action button in HUD
 */
@Composable
private fun ShortActionButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    count: String,
    isActive: Boolean = false,
    activeColor: Color = Color.White,
    onClick: () -> Unit
) {
    var isPressed by remember { mutableStateOf(false) }
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.8f else 1f,
        animationSpec = spring(dampingRatio = 0.4f, stiffness = 400f),
        label = "buttonScale"
    )
    
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .scale(scale)
            .clickable(
                indication = null,
                interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() }
            ) {
                isPressed = true
                onClick()
            }
    ) {
        Box(
            modifier = Modifier
                .size(44.dp)
                .clip(CircleShape)
                .background(
                    if (isActive) activeColor.copy(alpha = 0.2f)
                    else Color.Transparent
                ),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = if (isActive) activeColor else Color.White,
                modifier = Modifier.size(28.dp)
            )
        }
        Text(
            text = count,
            color = Color.White,
            fontSize = 10.sp,
            fontWeight = FontWeight.Medium
        )
    }
    
    LaunchedEffect(isPressed) {
        if (isPressed) {
            delay(100)
            isPressed = false
        }
    }
}

/**
 * Bottom Info Section - User info, caption, and music
 */
@Composable
private fun BottomInfoSection(
    post: Post,
    onUserClick: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        // User Info Row - CLICKABLE to navigate to profile
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            modifier = Modifier
                .clickable { onUserClick() }
                .padding(bottom = 8.dp)
        ) {
            // Avatar
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(CircleShape)
                    .border(1.dp, Color.White.copy(alpha = 0.2f), CircleShape)
            ) {
                AsyncImage(
                    model = post.userAvatar,
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
            }
            
            // Username
            Text(
                text = "@${post.username ?: "user"}",
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
        }
        
        // Caption
        Text(
            text = post.content,
            color = Color.White.copy(alpha = 0.9f),
            fontSize = 14.sp,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
            lineHeight = 20.sp,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        
        // Music Tag
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .clip(RoundedCornerShape(20.dp))
                .background(Color.White.copy(alpha = 0.1f))
                .border(1.dp, Color.White.copy(alpha = 0.05f), RoundedCornerShape(20.dp))
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Icon(
                imageVector = Icons.Filled.MusicNote,
                contentDescription = null,
                tint = IndigoAccent,
                modifier = Modifier.size(12.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "Original Audio - ${post.username ?: "user"}",
                color = Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.widthIn(max = 160.dp)
            )
        }
    }
}

/**
 * Shorts Top Header with Back Arrow
 */
@Composable
private fun ShortsTopHeader(
    onNavigateBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Back arrow
        IconButton(onClick = onNavigateBack) {
            Icon(
                imageVector = Icons.Filled.ArrowBack,
                contentDescription = "Back",
                tint = Color.White,
                modifier = Modifier.size(28.dp)
            )
        }
        
        Text(
            text = "Shorts",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White,
            modifier = Modifier.weight(1f)
        )
        
        IconButton(onClick = { /* Create Short */ }) {
            Icon(
                imageVector = Icons.Outlined.CameraAlt,
                contentDescription = "Create Short",
                tint = Color.White,
                modifier = Modifier.size(28.dp)
            )
        }
    }
}

/**
 * Empty Shorts State
 */
@Composable
private fun EmptyShortsStat() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.linearGradient(
                            colors = listOf(IndigoAccent, PinkAccent)
                        )
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.VideoLibrary,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(50.dp)
                )
            }
            
            Text(
                text = "No Shorts Yet",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            
            Text(
                text = "Be the first to post a short video!",
                fontSize = 14.sp,
                color = Color.White.copy(alpha = 0.6f)
            )
        }
    }
}

private fun formatCount(count: Int): String {
    return when {
        count >= 1_000_000 -> "${count / 1_000_000}M"
        count >= 1_000 -> "${count / 1_000}K"
        else -> count.toString()
    }
}
