package com.orignal.buddylynk.ui.screens

import android.content.Intent
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.FlingBehavior
import androidx.compose.foundation.gestures.ScrollableDefaults
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import kotlinx.coroutines.launch
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.ui.components.*
import com.orignal.buddylynk.ui.theme.*
import com.orignal.buddylynk.ui.viewmodel.HomeViewModel

// =============================================================================
// HOME SCREEN - Real-time posts from AWS
// =============================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    onNavigateToProfile: () -> Unit,
    onNavigateToSearch: () -> Unit,
    onNavigateToCreate: () -> Unit = {},
    onNavigateToTeamUp: () -> Unit = {},
    onNavigateToLive: () -> Unit = {},
    onNavigateToEvents: () -> Unit = {},
    onNavigateToChat: () -> Unit = {},
    onNavigateToUserProfile: (String) -> Unit = {},
    onNavigateToShorts: () -> Unit = {}, // Navigate to Shorts screen
    viewModel: HomeViewModel = viewModel()
) {
    val context = LocalContext.current
    val posts by viewModel.posts.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val isRefreshing by viewModel.isRefreshing.collectAsState()
    val error by viewModel.error.collectAsState()
    
    // Comment sheet state
    var selectedPostForComment by remember { mutableStateOf<Post?>(null) }
    var showCommentSheet by remember { mutableStateOf(false) }
    
    // Track previous connection state for detecting offline->online transition
    var wasOffline by remember { mutableStateOf(false) }
    
    // Network connectivity observer - refresh when internet comes back
    val isConnected by com.orignal.buddylynk.data.network.NetworkObserver
        .observeConnectivity(context)
        .collectAsState(initial = true)
    
    // Refresh when network changes from offline to online
    LaunchedEffect(isConnected) {
        if (isConnected && wasOffline) {
            // Network came back online - refresh
            viewModel.refresh()
        }
        wasOffline = !isConnected
    }
    
    // Lifecycle observer - always refresh when app comes to foreground
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
                // Always refresh on resume to get latest data
                viewModel.refresh()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }
    
    // Keep screen awake while on home page (like TikTok/Instagram)
    val activity = context as? android.app.Activity
    DisposableEffect(Unit) {
        activity?.window?.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            activity?.window?.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }
    
    // Responsive screen info
    val screenInfo = com.orignal.buddylynk.ui.utils.rememberScreenInfo()
    val dims = screenInfo.dimensions
    
    // Active server health check state
    // Start with true (assume online) - only show error on confirmed failure
    var isServerOnline by remember { mutableStateOf(true) }
    var isCheckingServer by remember { mutableStateOf(false) }
    var hasCheckedOnce by remember { mutableStateOf(false) }
    var showServerErrorSnackbar by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    
    // Continuous server health check - polls every 30 seconds
    LaunchedEffect(Unit) {
        // Initial check
        isCheckingServer = true
        try {
            val isHealthy = com.orignal.buddylynk.data.network.ServerHealthObserver.checkServerHealth()
            isServerOnline = isHealthy
            hasCheckedOnce = true
            android.util.Log.d("HomeScreen", "Initial server health check: $isHealthy")
        } catch (e: Exception) {
            isServerOnline = false
            hasCheckedOnce = true
            android.util.Log.e("HomeScreen", "Server health check failed: ${e.message}")
        }
        isCheckingServer = false
        
        // Continuous polling every 30 seconds
        while (true) {
            kotlinx.coroutines.delay(30_000L) // 30 seconds
            try {
                val isHealthy = com.orignal.buddylynk.data.network.ServerHealthObserver.checkServerHealth()
                isServerOnline = isHealthy
                android.util.Log.d("HomeScreen", "Periodic health check: online=$isHealthy")
            } catch (e: Exception) {
                isServerOnline = false
                android.util.Log.e("HomeScreen", "Periodic health check failed: ${e.message}")
            }
        }
    }
    
    // Server down state - shows when server is confirmed offline after health check
    val isServerDown = !isServerOnline && hasCheckedOnce && !isLoading
    
    // Show ServerDownScreen if server is down
    if (isServerDown) {
        Box(modifier = Modifier.fillMaxSize()) {
            ServerDownScreen(
                onRetry = { 
                    isCheckingServer = true
                    // Re-check server on retry
                    scope.launch {
                        try {
                            val isHealthy = com.orignal.buddylynk.data.network.ServerHealthObserver.checkServerHealth()
                            isServerOnline = isHealthy
                            if (isHealthy) {
                                // Server is back! Refresh posts
                                viewModel.refresh()
                            } else {
                                // Still down - show snackbar
                                showServerErrorSnackbar = true
                            }
                        } catch (e: Exception) {
                            isServerOnline = false
                            showServerErrorSnackbar = true
                        }
                        isCheckingServer = false
                    }
                },
                isRetrying = isCheckingServer
            )
            
            // Custom styled popup at the top
            androidx.compose.animation.AnimatedVisibility(
                visible = showServerErrorSnackbar,
                enter = androidx.compose.animation.slideInVertically(
                    initialOffsetY = { -it }
                ) + androidx.compose.animation.fadeIn(),
                exit = androidx.compose.animation.slideOutVertically(
                    targetOffsetY = { -it }
                ) + androidx.compose.animation.fadeOut(),
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 60.dp, start = 24.dp, end = 24.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(
                            Brush.horizontalGradient(
                                colors = listOf(
                                    Color(0xFFDC2626),
                                    Color(0xFFB91C1C)
                                )
                            )
                        )
                        .padding(16.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // Warning icon
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .clip(CircleShape)
                                .background(Color.White.copy(alpha = 0.2f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "⚠️",
                                fontSize = 18.sp
                            )
                        }
                        
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Connection Failed",
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White
                            )
                            Text(
                                text = "Server is still not responding",
                                fontSize = 13.sp,
                                color = Color.White.copy(alpha = 0.9f)
                            )
                        }
                    }
                }
                
                // Auto-dismiss after 3 seconds
                LaunchedEffect(Unit) {
                    kotlinx.coroutines.delay(3000)
                    showServerErrorSnackbar = false
                }
            }
        }
        return
    }
    
    AnimatedGradientBackground(
        modifier = Modifier.fillMaxSize()
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                // statusBarsPadding is in PremiumHomeHeader, no need here
        ) {
            // Premium Animated Header (Like React design)
            val currentUser by AuthManager.currentUser.collectAsState()
            PremiumHomeHeader(
                userAvatar = currentUser?.avatar,
                username = currentUser?.username ?: currentUser?.email?.substringBefore("@") ?: "User",
                onCreateClick = onNavigateToCreate,
                onNotificationClick = { /* TODO */ },
                onProfileClick = onNavigateToProfile
            )
        
            // Smooth scroll state for buttery scrolling
            val listState = rememberLazyListState()
            
            // Track which post is most visible (for video playback like Instagram)
            // Simplified calculation for better scroll performance
            val visiblePostIndex by remember {
                derivedStateOf {
                    // Use first visible item for simpler/faster calculation
                    val firstVisibleIndex = listState.firstVisibleItemIndex
                    // Subtract 1 for header item to get actual post index
                    maxOf(firstVisibleIndex - 1, 0)
                }
            }
            
            // Collect pagination states for infinite scroll
            val hasMorePosts by viewModel.hasMorePosts.collectAsState()
            val isLoadingMore by viewModel.isLoadingMore.collectAsState()
            
            // Infinite scroll: Load more when near bottom
            LaunchedEffect(listState) {
                snapshotFlow {
                    val layoutInfo = listState.layoutInfo
                    val totalItems = layoutInfo.totalItemsCount
                    val lastVisibleIndex = layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: 0
                    lastVisibleIndex >= totalItems - 5 // Trigger when 5 items from end
                }.collect { shouldLoadMore ->
                    if (shouldLoadMore && hasMorePosts && !isLoadingMore) {
                        viewModel.loadMorePosts()
                    }
                }
            }
            
            // Content - responsive padding with smooth scrolling
            LazyColumn(
                state = listState,
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(dims.screenPadding),
                verticalArrangement = Arrangement.spacedBy(dims.itemSpacing),
                flingBehavior = ScrollableDefaults.flingBehavior()
            ) {
                // Feed Section Header - Stories moved to Create page
                item {
                    Text(
                        text = "Your Feed",
                        style = MaterialTheme.typography.titleLarge,
                        color = TextPrimary
                    )
                }
                
                // Loading state
                if (isLoading && posts.isEmpty()) {
                    items(3) {
                        ShimmerPost()
                        Spacer(modifier = Modifier.height(16.dp))
                    }
                }
                
                // Error state
                error?.let { _ ->
                    item {
                        GlassCard(
                            modifier = Modifier.fillMaxWidth(),
                            cornerRadius = 16.dp,
                            glassOpacity = 0.1f
                        ) {
                            Column(
                                modifier = Modifier.padding(16.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Text(
                                    text = "Couldn't load posts",
                                    style = MaterialTheme.typography.bodyLarge,
                                    color = TextPrimary
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                GradientButton(
                                    text = "Retry",
                                    onClick = { viewModel.loadPosts() },
                                    gradient = PremiumGradient
                                )
                            }
                        }
                    }
                }
                
                // Real posts from AWS - Premium Cards with Action Capsule
                if (posts.isNotEmpty()) {
                    itemsIndexed(posts, key = { _, post -> post.postId }) { index, post ->
                        // Animate each post with fade-in + slide-up effect
                        var isVisible by remember { mutableStateOf(false) }
                        LaunchedEffect(post.postId) {
                            isVisible = true
                        }
                        
                        AnimatedVisibility(
                            visible = isVisible,
                            enter = fadeIn(animationSpec = tween(300)) + 
                                   slideInVertically(
                                       animationSpec = tween(300),
                                       initialOffsetY = { it / 4 }
                                   )
                        ) {
                        PremiumPostCard(
                            post = post,
                            isLiked = post.isLiked,
                            isSaved = post.isBookmarked,
                            hasStatus = false, // TODO: Check if user has active story
                            isOwner = post.userId == viewModel.currentUserId,
                            isVisible = index == visiblePostIndex, // Only play video when this post is visible
                            onLike = { viewModel.likePost(post.postId) },
                            onSave = { viewModel.bookmarkPost(post.postId) },
                            onComment = { 
                                // Open comment sheet directly (no toast)
                                selectedPostForComment = post
                                showCommentSheet = true
                            },
                            onShare = { 
                                // Increment share count first
                                viewModel.sharePost(post.postId)
                                // Build share text with app.buddylynk.com link
                                val deepLink = "https://app.buddylynk.com/post/${post.postId}"
                                val shareText = buildString {
                                    post.content?.take(100)?.let {
                                        append(it)
                                        if (it.length >= 100) append("...")
                                        append("\n\n")
                                    }
                                    append("Check out this post on BuddyLynk!\n")
                                    append(deepLink)
                                }
                                // Open Android share sheet
                                val sendIntent = Intent().apply {
                                    action = Intent.ACTION_SEND
                                    putExtra(Intent.EXTRA_TEXT, shareText)
                                    type = "text/plain"
                                }
                                context.startActivity(Intent.createChooser(sendIntent, "Share Post"))
                            },
                            onUserClick = { userId -> onNavigateToUserProfile(userId) },
                            onEdit = { 
                                android.widget.Toast.makeText(context, "Edit feature coming soon!", android.widget.Toast.LENGTH_SHORT).show()
                            },
                            onDelete = {
                                // Delete post
                                viewModel.deletePost(post.postId)
                            },
                            onReport = {
                                viewModel.reportPost(post.postId)
                            },
                            onBlock = {
                                viewModel.blockUser(post.userId)
                                android.widget.Toast.makeText(context, "User blocked", android.widget.Toast.LENGTH_SHORT).show()
                            },
                            onAvatarLongPress = { /* TODO: Show story */ },
                            onNavigateToShorts = onNavigateToShorts
                        )
                        } // End AnimatedVisibility
                        Spacer(modifier = Modifier.height(40.dp))
                    }
                } else if (!isLoading && error == null && posts.isEmpty()) {
                    // Empty state when no posts
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(32.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.Image,
                                    contentDescription = null,
                                    tint = TextTertiary,
                                    modifier = Modifier.size(64.dp)
                                )
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    text = "No posts yet",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = TextSecondary
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    text = "Be the first to share something!",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextTertiary
                                )
                            }
                        }
                    }
                }
                
                // Loading more indicator for infinite scroll
                if (isLoadingMore) {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(32.dp),
                                color = VibrantPurple,
                                strokeWidth = 3.dp
                            )
                        }
                    }
                }
                
                // Bottom padding for nav bar
                item {
                    Spacer(modifier = Modifier.height(80.dp))
                }
            }
        }
    }
    
    // Comment Bottom Sheet
    if (showCommentSheet && selectedPostForComment != null) {
        // State for comment input
        var commentText by remember { mutableStateOf("") }
        val focusRequester = remember { FocusRequester() }
        
        // Auto-focus on sheet open
        LaunchedEffect(showCommentSheet) {
            if (showCommentSheet) {
                kotlinx.coroutines.delay(200)
                focusRequester.requestFocus()
            }
        }
        
        ModalBottomSheet(
            onDismissRequest = { 
                showCommentSheet = false
                selectedPostForComment = null
            },
            containerColor = Color(0xFF1A1A1C),
            contentColor = Color.White,
            dragHandle = {
                Box(
                    modifier = Modifier
                        .padding(vertical = 16.dp)
                        .width(40.dp)
                        .height(4.dp)
                        .clip(RoundedCornerShape(2.dp))
                        .background(Color.White.copy(alpha = 0.3f))
                )
            }
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .imePadding() // Moves up with keyboard!
                    .padding(horizontal = 16.dp)
                    .padding(bottom = 16.dp)
            ) {
                // Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Comments",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    Text(
                        text = "${selectedPostForComment?.commentsCount ?: 0}",
                        fontSize = 16.sp,
                        color = Color(0xFFA1A1AA)
                    )
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Placeholder content
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(120.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.ChatBubbleOutline,
                            contentDescription = null,
                            tint = Color(0xFF71717A),
                            modifier = Modifier.size(40.dp)
                        )
                        Text(
                            text = "Be the first to comment!",
                            color = Color(0xFF71717A),
                            fontSize = 14.sp
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // FloatingInputBar style input
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(28.dp))
                        .background(Color(0xFF121212))
                        .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(28.dp))
                        .padding(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Emoji button
                    IconButton(
                        onClick = { /* Emoji picker */ },
                        modifier = Modifier
                            .size(40.dp)
                            .clip(CircleShape)
                            .background(Color(0xFF27272A))
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.EmojiEmotions,
                            contentDescription = "Emoji",
                            tint = Color(0xFFA1A1AA),
                            modifier = Modifier.size(20.dp)
                        )
                    }
                    
                    // Text input
                    BasicTextField(
                        value = commentText,
                        onValueChange = { commentText = it },
                        modifier = Modifier
                            .weight(1f)
                            .padding(horizontal = 12.dp)
                            .focusRequester(focusRequester),
                        textStyle = androidx.compose.ui.text.TextStyle(
                            color = Color.White,
                            fontSize = 14.sp
                        ),
                        cursorBrush = SolidColor(Color(0xFF8B5CF6)),
                        decorationBox = { innerTextField ->
                            Box {
                                if (commentText.isEmpty()) {
                                    Text(
                                        text = "Add a comment...",
                                        color = Color(0xFF52525B),
                                        fontSize = 14.sp
                                    )
                                }
                                innerTextField()
                            }
                        }
                    )
                    
                    // Send button - only show when there's text
                    if (commentText.isNotBlank()) {
                        IconButton(
                            onClick = {
                                android.widget.Toast.makeText(
                                    context,
                                    "Comment sent: $commentText",
                                    android.widget.Toast.LENGTH_SHORT
                                ).show()
                                commentText = ""
                            },
                            modifier = Modifier
                                .size(40.dp)
                                .clip(CircleShape)
                                .background(Color(0xFF8B5CF6))
                        ) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.Send,
                                contentDescription = "Send",
                                tint = Color.Black,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}


// =============================================================================
// TOP BAR - Unique glassmorphic design
// =============================================================================

@Composable
private fun HomeTopBar(
    onProfileClick: () -> Unit,
    onSearchClick: () -> Unit,
    onNavigateToCreate: () -> Unit = {},
    onRefresh: () -> Unit = {},
    isRefreshing: Boolean = false
) {
    // Get current user for profile picture
    val currentUser by AuthManager.currentUser.collectAsState()
    
    // Static border - no animation for scroll performance
    
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // App Name
        Text(
            text = "Buddylynk",
            style = MaterialTheme.typography.headlineMedium.copy(
                fontWeight = FontWeight.Bold,
                brush = Brush.horizontalGradient(PremiumGradient)
            )
        )
        
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Animated Upload Button - Circle with rotating gradient border
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(CircleShape)
                    .background(
                        brush = Brush.sweepGradient(
                            colors = listOf(
                                Color(0xFF6366F1),
                                Color(0xFF8B5CF6),
                                Color(0xFFA855F7),
                                Color(0xFF00D9FF),
                                Color(0xFF6366F1)
                            )
                        )
                    ),
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                        .background(Color(0xFF0A0A0A))
                        .clickable { onNavigateToCreate() },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Filled.Add,
                        contentDescription = "Upload",
                        tint = Color.White,
                        modifier = Modifier.size(24.dp)
                    )
                }
            }
            
            // Notifications
            GlassSurface(
                modifier = Modifier.size(44.dp),
                cornerRadius = 22.dp
            ) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Notifications,
                        contentDescription = "Notifications",
                        tint = TextPrimary,
                        modifier = Modifier.size(22.dp)
                    )
                    // Notification badge
                    Box(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .offset(x = (-4).dp, y = 4.dp)
                            .size(10.dp)
                            .background(AccentPink, CircleShape)
                    )
                }
            }
            
            // Profile avatar - show actual picture
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(CircleShape)
                    .border(2.dp, Brush.linearGradient(PremiumGradient), CircleShape)
                    .clickable { onProfileClick() },
                contentAlignment = Alignment.Center
            ) {
                val avatarUrl = currentUser?.avatar
                if (!avatarUrl.isNullOrBlank()) {
                    AsyncImage(
                        model = avatarUrl,
                        contentDescription = "Profile",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(CircleShape),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    // Fallback to gradient with icon
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Brush.linearGradient(PremiumGradient)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Person,
                            contentDescription = "Profile",
                            tint = Color.White,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                }
            }
        }
    }
}


// =============================================================================
// GREETING SECTION - Personalized welcome
// =============================================================================

@Composable
private fun GreetingSection() {
    var visible by remember { mutableStateOf(false) }
    
    LaunchedEffect(Unit) {
        visible = true
    }
    
    AnimatedVisibility(
        visible = visible,
        enter = fadeIn(tween(600)) + slideInVertically(
            initialOffsetY = { -20 },
            animationSpec = tween(600)
        )
    ) {
        FloatingGlassCard {
            Column(
                modifier = Modifier.padding(20.dp)
            ) {
                Text(
                    text = "Good evening! 👋",
                    style = MaterialTheme.typography.titleLarge,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Ready to connect with your buddies?",
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextSecondary
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Stats row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    StatItem(value = "24", label = "Buddies")
                    StatItem(value = "12", label = "Online")
                    StatItem(value = "5", label = "New")
                }
            }
        }
    }
}

@Composable
private fun StatItem(value: String, label: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = value,
            style = MaterialTheme.typography.headlineSmall.copy(
                fontWeight = FontWeight.Bold,
                brush = Brush.horizontalGradient(PremiumGradient)
            )
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = TextTertiary
        )
    }
}

// =============================================================================
// QUICK ACTIONS - Unique gradient cards
// =============================================================================

@Composable
private fun QuickActionsSection(
    onTeamUpClick: () -> Unit,
    onLiveClick: () -> Unit,
    onEventsClick: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        QuickActionCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Filled.Groups,
            title = "Team Up",
            gradient = listOf(GradientPurple, GradientBlue),
            onClick = onTeamUpClick
        )
        QuickActionCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Filled.Videocam,
            title = "Live",
            gradient = listOf(GradientPink, GradientOrange),
            onClick = onLiveClick
        )
        QuickActionCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Filled.Event,
            title = "Events",
            gradient = listOf(GradientCyan, GradientTeal),
            onClick = onEventsClick
        )
    }
}

@Composable
private fun QuickActionCard(
    modifier: Modifier = Modifier,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    gradient: List<Color>,
    onClick: () -> Unit
) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.95f else 1f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessMedium
        ),
        label = "scale"
    )
    
    GlassCard(
        modifier = modifier
            .scale(scale)
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = onClick
            ),
        cornerRadius = 20.dp,
        glassOpacity = 0.08f
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(
                        brush = Brush.linearGradient(gradient)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = title,
                    tint = Color.White,
                    modifier = Modifier.size(26.dp)
                )
            }
            
            Spacer(modifier = Modifier.height(10.dp))
            
            Text(
                text = title,
                style = MaterialTheme.typography.labelMedium,
                color = TextPrimary
            )
        }
    }
}

// =============================================================================
// STORIES SECTION - Unique glowing circles
// =============================================================================

@Composable
private fun StoriesSection() {
    Column {
        Text(
            text = "Stories",
            style = MaterialTheme.typography.titleMedium,
            color = TextPrimary,
            modifier = Modifier.padding(vertical = 8.dp)
        )
        
        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Add story button
            item {
                AddStoryButton()
            }
            
            // Real stories would be loaded from ViewModel
            // Empty for now - no dummy data
        }
    }
}

@Composable
private fun AddStoryButton() {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        GlassCard(
            modifier = Modifier.size(68.dp),
            cornerRadius = 34.dp,
            glassOpacity = 0.1f
        ) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.Add,
                    contentDescription = "Add Story",
                    tint = GradientPurple,
                    modifier = Modifier.size(28.dp)
                )
            }
        }
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = "Add",
            style = MaterialTheme.typography.labelSmall,
            color = TextTertiary
        )
    }
}

@Composable
private fun StoryItem(
    name: String,
    hasUnread: Boolean,
    isLive: Boolean
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box {
            // Gradient ring for unread
            Box(
                modifier = Modifier
                    .size(72.dp)
                    .clip(CircleShape)
                    .background(
                        brush = if (hasUnread) {
                            Brush.sweepGradient(VibrantGradient)
                        } else {
                            Brush.linearGradient(
                                listOf(
                                    GlassBorder,
                                    GlassBorder
                                )
                            )
                        }
                    )
                    .padding(3.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(CircleShape)
                        .background(DarkSurface)
                        .padding(2.dp)
                ) {
                    // Avatar
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(CircleShape)
                            .background(
                                brush = Brush.linearGradient(
                                    listOf(
                                        GradientPurple.copy(alpha = 0.3f),
                                        GradientCyan.copy(alpha = 0.3f)
                                    )
                                )
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = name.first().toString(),
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimary
                        )
                    }
                }
            }
            
            // Live indicator
            if (isLive) {
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .offset(y = 4.dp)
                        .background(
                            brush = Brush.horizontalGradient(
                                listOf(GradientPink, GradientOrange)
                            ),
                            shape = RoundedCornerShape(4.dp)
                        )
                        .padding(horizontal = 8.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = "LIVE",
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold
                        ),
                        color = Color.White
                    )
                }
            }
        }
        
        Spacer(modifier = Modifier.height(6.dp))
        
        Text(
            text = name,
            style = MaterialTheme.typography.labelSmall,
            color = if (hasUnread) TextPrimary else TextTertiary
        )
    }
}


// =============================================================================
// REAL FEED CARD - For AWS posts with views
// =============================================================================

@Composable
private fun RealFeedCard(
    post: Post,
    onLike: () -> Unit,
    onBookmark: () -> Unit,
    onView: () -> Unit,
    onEdit: () -> Unit = {},
    onDelete: () -> Unit = {},
    onBlockUser: () -> Unit = {},
    onReportPost: () -> Unit = {},
    isOwnPost: Boolean = false
) {
    val context = LocalContext.current
    var showComments by remember { mutableStateOf(false) }
    var showShare by remember { mutableStateOf(false) }
    var showOptionsMenu by remember { mutableStateOf(false) }
    var localComments by remember { mutableStateOf(emptyList<Comment>()) }
    var localCommentCount by remember { mutableIntStateOf(post.commentsCount) }
    
    // Confirmation dialog states
    var showDeleteConfirmation by remember { mutableStateOf(false) }
    var showBlockConfirmation by remember { mutableStateOf(false) }
    
    // Track view when card becomes visible
    LaunchedEffect(post.postId) {
        onView()
    }
    
    // Delete confirmation dialog
    if (showDeleteConfirmation) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirmation = false },
            containerColor = DarkSurface,
            titleContentColor = TextPrimary,
            textContentColor = TextSecondary,
            title = {
                Text(
                    text = "Delete Post?",
                    fontWeight = FontWeight.Bold
                )
            },
            text = {
                Text("This action cannot be undone. Your post will be permanently deleted.")
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showDeleteConfirmation = false
                        onDelete()
                    }
                ) {
                    Text("Delete", color = LikeRed, fontWeight = FontWeight.SemiBold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirmation = false }) {
                    Text("Cancel", color = TextSecondary)
                }
            }
        )
    }
    
    // Block confirmation dialog
    if (showBlockConfirmation) {
        AlertDialog(
            onDismissRequest = { showBlockConfirmation = false },
            containerColor = DarkSurface,
            titleContentColor = TextPrimary,
            textContentColor = TextSecondary,
            title = {
                Text(
                    text = "Block ${post.username ?: "this user"}?",
                    fontWeight = FontWeight.Bold
                )
            },
            text = {
                Text("They won't be able to see your profile, posts, or message you. You can unblock them later in Settings.")
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showBlockConfirmation = false
                        onBlockUser()
                    }
                ) {
                    Text("Block", color = LikeRed, fontWeight = FontWeight.SemiBold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showBlockConfirmation = false }) {
                    Text("Cancel", color = TextSecondary)
                }
            }
        )
    }
    
    // Comments bottom sheet
    if (showComments) {
        CommentsBottomSheet(
            comments = localComments,
            onDismiss = { showComments = false },
            onSendComment = { text ->
                val newComment = Comment(
                    id = System.currentTimeMillis().toString(),
                    username = "You",
                    content = text,
                    timeAgo = "Just now",
                    likes = 0,
                    isLiked = false
                )
                localComments = listOf(newComment) + localComments
                localCommentCount++
            },
            onLikeComment = { commentId ->
                localComments = localComments.map { comment ->
                    if (comment.id == commentId) {
                        comment.copy(
                            isLiked = !comment.isLiked,
                            likes = if (comment.isLiked) comment.likes - 1 else comment.likes + 1
                        )
                    } else comment
                }
            }
        )
    }
    
    // Native share intent (instead of bottom sheet)
    if (showShare) {
        LaunchedEffect(Unit) {
            val shareText = buildString {
                append(post.content)
                if (!post.mediaUrl.isNullOrBlank()) {
                    append("\n\n")
                    append(post.mediaUrl)
                }
                append("\n\nShared via Buddylynk")
            }
            val shareIntent = Intent().apply {
                action = Intent.ACTION_SEND
                putExtra(Intent.EXTRA_TEXT, shareText)
                type = "text/plain"
            }
            context.startActivity(Intent.createChooser(shareIntent, "Share Post"))
            showShare = false
        }
    }
    
    GlassCard(
        modifier = Modifier.fillMaxWidth(),
        cornerRadius = 24.dp,
        glassOpacity = 0.08f
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Avatar
                    if (post.userAvatar != null) {
                        val avatarRequest = remember(post.userAvatar) {
                            ImageRequest.Builder(context)
                                .data(post.userAvatar)
                                .crossfade(true)
                                .build()
                        }
                        AsyncImage(
                            model = avatarRequest,
                            contentDescription = null,
                            modifier = Modifier
                                .size(44.dp)
                                .clip(CircleShape),
                            contentScale = ContentScale.Crop
                        )
                    } else {
                        Box(
                            modifier = Modifier
                                .size(44.dp)
                                .clip(CircleShape)
                                .background(
                                    brush = Brush.linearGradient(PremiumGradient)
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = post.username?.firstOrNull()?.uppercase() ?: "U",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = Color.White
                            )
                        }
                    }
                    
                    Column {
                        Text(
                            text = post.username ?: "User",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.SemiBold,
                            color = TextPrimary
                        )
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = formatTimeAgo(post.createdAt),
                                style = MaterialTheme.typography.bodySmall,
                                color = TextTertiary
                            )
                            // Views count
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.Visibility,
                                    contentDescription = "Views",
                                    tint = TextTertiary,
                                    modifier = Modifier.size(14.dp)
                                )
                                Text(
                                    text = formatCount(post.viewsCount),
                                    style = MaterialTheme.typography.labelSmall,
                                    color = TextTertiary
                                )
                            }
                        }
                    }
                }
                
                // More options dropdown
                Box {
                    IconButton(onClick = { showOptionsMenu = true }) {
                        Icon(
                            imageVector = Icons.Filled.MoreHoriz,
                            contentDescription = "More options",
                            tint = TextTertiary,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                    
                    DropdownMenu(
                        expanded = showOptionsMenu,
                        onDismissRequest = { showOptionsMenu = false },
                        modifier = Modifier
                            .widthIn(min = 220.dp, max = 260.dp)
                            .clip(RoundedCornerShape(18.dp))
                            .background(Color(0xFF0D0D1A)) // Solid base
                            .background(
                                Brush.verticalGradient(
                                    colors = listOf(
                                        Color(0xFF252545).copy(alpha = 0.98f),
                                        Color(0xFF1A1A35),
                                        Color(0xFF0F0F20)
                                    )
                                )
                            )
                            .border(
                                width = 1.dp,
                                brush = Brush.linearGradient(
                                    colors = listOf(
                                        Color(0xFF8B5CF6).copy(alpha = 0.5f),
                                        Color(0xFFEC4899).copy(alpha = 0.3f),
                                        Color(0xFF6366F1).copy(alpha = 0.4f)
                                    )
                                ),
                                shape = RoundedCornerShape(18.dp)
                            )
                            .padding(vertical = 8.dp, horizontal = 6.dp)
                    ) {
                        if (isOwnPost) {
                            // Edit option - compact
                            DropdownMenuItem(
                                text = { 
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        modifier = Modifier.padding(vertical = 4.dp)
                                    ) {
                                        Box(
                                            modifier = Modifier
                                                .size(36.dp)
                                                .clip(RoundedCornerShape(10.dp))
                                                .background(
                                                    Brush.radialGradient(
                                                        colors = listOf(
                                                            Color(0xFF6366F1).copy(alpha = 0.3f),
                                                            Color(0xFF8B5CF6).copy(alpha = 0.15f)
                                                        )
                                                    )
                                                )
                                                .border(
                                                    width = 0.5.dp,
                                                    color = Color(0xFF818CF8).copy(alpha = 0.3f),
                                                    shape = RoundedCornerShape(10.dp)
                                                ),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Icon(
                                                Icons.Outlined.Edit, 
                                                null, 
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
                                                "Modify content", 
                                                color = Color.White.copy(alpha = 0.45f),
                                                fontSize = 11.sp
                                            )
                                        }
                                    }
                                },
                                onClick = { showOptionsMenu = false; onEdit() }
                            )
                            
                            // Divider
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 20.dp, vertical = 6.dp)
                                    .height(1.dp)
                                    .background(
                                        Brush.horizontalGradient(
                                            colors = listOf(
                                                Color.Transparent,
                                                Color.White.copy(alpha = 0.08f),
                                                Color.Transparent
                                            )
                                        )
                                    )
                            )
                            
                            // Delete option with premium styling
                            DropdownMenuItem(
                                text = { 
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        modifier = Modifier.padding(vertical = 6.dp)
                                    ) {
                                        Box(
                                            modifier = Modifier
                                                .size(42.dp)
                                                .clip(RoundedCornerShape(12.dp))
                                                .background(
                                                    Brush.linearGradient(
                                                        colors = listOf(
                                                            Color(0xFFEF4444).copy(alpha = 0.25f),
                                                            Color(0xFFDC2626).copy(alpha = 0.15f)
                                                        )
                                                    )
                                                ),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Icon(
                                                Icons.Outlined.Delete, 
                                                null, 
                                                tint = Color(0xFFF87171),
                                                modifier = Modifier.size(20.dp)
                                            )
                                        }
                                        Spacer(Modifier.width(14.dp))
                                        Column {
                                            Text(
                                                "Delete Post", 
                                                color = Color(0xFFF87171),
                                                fontWeight = FontWeight.SemiBold,
                                                fontSize = 15.sp
                                            )
                                            Text(
                                                "Remove permanently", 
                                                color = Color(0xFFEF4444).copy(alpha = 0.5f),
                                                fontSize = 12.sp
                                            )
                                        }
                                    }
                                },
                                onClick = { showOptionsMenu = false; showDeleteConfirmation = true }
                            )
                        } else {
                            // Report option with premium styling
                            DropdownMenuItem(
                                text = { 
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        modifier = Modifier.padding(vertical = 6.dp)
                                    ) {
                                        Box(
                                            modifier = Modifier
                                                .size(42.dp)
                                                .clip(RoundedCornerShape(12.dp))
                                                .background(
                                                    Brush.linearGradient(
                                                        colors = listOf(
                                                            Color(0xFFFBBF24).copy(alpha = 0.25f),
                                                            Color(0xFFF59E0B).copy(alpha = 0.15f)
                                                        )
                                                    )
                                                ),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Icon(
                                                Icons.Outlined.Flag, 
                                                null, 
                                                tint = Color(0xFFFCD34D),
                                                modifier = Modifier.size(20.dp)
                                            )
                                        }
                                        Spacer(Modifier.width(14.dp))
                                        Column(modifier = Modifier.weight(1f)) {
                                            Text(
                                                "Report Post", 
                                                color = Color.White,
                                                fontWeight = FontWeight.SemiBold,
                                                fontSize = 15.sp
                                            )
                                            Text(
                                                "Report inappropriate content", 
                                                color = Color.White.copy(alpha = 0.5f),
                                                fontSize = 12.sp
                                            )
                                        }
                                        Icon(
                                            Icons.Filled.ChevronRight,
                                            null,
                                            tint = Color.White.copy(alpha = 0.3f),
                                            modifier = Modifier.size(18.dp)
                                        )
                                    }
                                },
                                onClick = { showOptionsMenu = false; onReportPost() }
                            )
                            
                            // Divider
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 20.dp, vertical = 6.dp)
                                    .height(1.dp)
                                    .background(
                                        Brush.horizontalGradient(
                                            colors = listOf(
                                                Color.Transparent,
                                                Color.White.copy(alpha = 0.08f),
                                                Color.Transparent
                                            )
                                        )
                                    )
                            )
                            
                            // Block option with premium styling
                            DropdownMenuItem(
                                text = { 
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        modifier = Modifier.padding(vertical = 6.dp)
                                    ) {
                                        Box(
                                            modifier = Modifier
                                                .size(42.dp)
                                                .clip(RoundedCornerShape(12.dp))
                                                .background(
                                                    Brush.linearGradient(
                                                        colors = listOf(
                                                            Color(0xFFEF4444).copy(alpha = 0.25f),
                                                            Color(0xFFDC2626).copy(alpha = 0.15f)
                                                        )
                                                    )
                                                ),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Icon(
                                                Icons.Outlined.Block, 
                                                null, 
                                                tint = Color(0xFFF87171),
                                                modifier = Modifier.size(20.dp)
                                            )
                                        }
                                        Spacer(Modifier.width(14.dp))
                                        Column(modifier = Modifier.weight(1f)) {
                                            Text(
                                                "Block User", 
                                                color = Color(0xFFF87171),
                                                fontWeight = FontWeight.SemiBold,
                                                fontSize = 15.sp
                                            )
                                            Text(
                                                "Hide all their content", 
                                                color = Color(0xFFEF4444).copy(alpha = 0.5f),
                                                fontSize = 12.sp
                                            )
                                        }
                                        Icon(
                                            Icons.Filled.ChevronRight,
                                            null,
                                            tint = Color.White.copy(alpha = 0.3f),
                                            modifier = Modifier.size(18.dp)
                                        )
                                    }
                                },
                                onClick = { showOptionsMenu = false; showBlockConfirmation = true }
                            )
                        }
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Content text
            if (post.content.isNotBlank()) {
                Text(
                    text = post.content,
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(12.dp))
            }
            
            // Media - show directly without loading indicator
            if (!post.mediaUrl.isNullOrBlank()) {
                var isError by remember { mutableStateOf(false) }
                
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(280.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(DarkSurface)
                ) {
                    // Error state
                    if (isError) {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(GlassWhite),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.ImageNotSupported,
                                    contentDescription = null,
                                    tint = TextTertiary,
                                    modifier = Modifier.size(48.dp)
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    text = "Media unavailable",
                                    color = TextTertiary,
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                    }
                    
                    // Image/Video - shows directly with crossfade animation
                    AsyncImage(
                        model = ImageRequest.Builder(context)
                            .data(post.mediaUrl)
                            .crossfade(300)
                            .listener(
                                onError = { _, _ -> isError = true }
                            )
                            .build(),
                        contentDescription = "Post media",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )
                    
                    // Video play button overlay
                    if (post.mediaType == "video" && !isError) {
                        Box(
                            modifier = Modifier
                                .align(Alignment.Center)
                                .size(56.dp)
                                .clip(CircleShape)
                                .background(
                                    brush = Brush.linearGradient(PremiumGradient),
                                    alpha = 0.9f
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Filled.PlayArrow,
                                contentDescription = "Play video",
                                tint = Color.White,
                                modifier = Modifier.size(32.dp)
                            )
                        }
                        
                        // Video duration badge
                        Box(
                            modifier = Modifier
                                .align(Alignment.BottomEnd)
                                .padding(8.dp)
                                .background(
                                    color = Color.Black.copy(alpha = 0.7f),
                                    shape = RoundedCornerShape(4.dp)
                                )
                                .padding(horizontal = 6.dp, vertical = 2.dp)
                        ) {
                            Text(
                                text = "VIDEO",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color.White,
                                fontSize = 10.sp
                            )
                        }
                    }
                }
                Spacer(modifier = Modifier.height(12.dp))
            }
            
            // Action bar
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(20.dp)
                ) {
                    // Like
                    PopIconButton(
                        onClick = onLike,
                        isActive = post.isLiked
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Icon(
                                imageVector = if (post.isLiked) Icons.Filled.Favorite else Icons.Outlined.FavoriteBorder,
                                contentDescription = "Like",
                                tint = if (post.isLiked) LikeRed else TextSecondary,
                                modifier = Modifier.size(24.dp)
                            )
                            Text(
                                text = formatCount(post.likesCount),
                                style = MaterialTheme.typography.bodySmall,
                                color = if (post.isLiked) LikeRed else TextSecondary
                            )
                        }
                    }
                    
                    // Comment
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        modifier = Modifier.clickable { showComments = true }
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.ChatBubbleOutline,
                            contentDescription = "Comment",
                            tint = TextSecondary,
                            modifier = Modifier.size(22.dp)
                        )
                        Text(
                            text = formatCount(post.commentsCount),
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary
                        )
                    }
                    
                    // Share
                    Icon(
                        imageVector = Icons.Outlined.Share,
                        contentDescription = "Share",
                        tint = TextSecondary,
                        modifier = Modifier
                            .size(22.dp)
                            .clickable { showShare = true }
                    )
                }
                
                // Bookmark
                PopIconButton(
                    onClick = onBookmark,
                    isActive = post.isBookmarked
                ) {
                    Icon(
                        imageVector = if (post.isBookmarked) Icons.Filled.Bookmark else Icons.Outlined.BookmarkBorder,
                        contentDescription = "Bookmark",
                        tint = if (post.isBookmarked) GradientCoral else TextSecondary,
                        modifier = Modifier.size(22.dp)
                    )
                }
            }
        }
    }
}

// Helper functions
private fun formatTimeAgo(timestamp: String): String {
    return try {
        val time = timestamp.toLongOrNull() ?: return "Just now"
        val diff = System.currentTimeMillis() - time
        val minutes = diff / 60000
        val hours = diff / 3600000
        val days = diff / 86400000
        when {
            minutes < 1 -> "Just now"
            minutes < 60 -> "${minutes}m"
            hours < 24 -> "${hours}h"
            days < 7 -> "${days}d"
            else -> "${days / 7}w"
        }
    } catch (e: Exception) {
        "Just now"
    }
}

private fun formatCount(count: Int): String {
    return when {
        count < 1000 -> count.toString()
        count < 1000000 -> "${count / 1000}K"
        else -> "${count / 1000000}M"
    }
}
