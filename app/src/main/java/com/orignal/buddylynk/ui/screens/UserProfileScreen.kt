package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.ui.components.*
import com.orignal.buddylynk.ui.theme.*
import com.orignal.buddylynk.ui.viewmodel.UserProfileViewModel

// =============================================================================
// USER PROFILE SCREEN - Uses exact same design as ProfileScreen
// =============================================================================

@Composable
fun UserProfileScreen(
    userId: String,
    onNavigateBack: () -> Unit,
    onNavigateToChat: (String) -> Unit = {},
    viewModel: UserProfileViewModel = viewModel()
) {
    val userProfile by viewModel.userProfile.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val isFollowing by viewModel.isFollowing.collectAsState()
    val isFollowLoading by viewModel.isFollowLoading.collectAsState()
    val error by viewModel.error.collectAsState()
    
    // Selected tab
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Gallery", "Mentions", "About")
    
    // Load user profile
    LaunchedEffect(userId) {
        viewModel.loadUserProfile(userId)
    }
    
    // Animated rotation for background orbs
    val infiniteTransition = rememberInfiniteTransition(label = "bg")
    val orbRotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(30000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "orb"
    )
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF050505))
    ) {
        // Dynamic animated background orbs
        UserFuturisticBackground(orbRotation)
        
        if (isLoading) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(
                    color = GradientCyan,
                    strokeWidth = 2.dp
                )
            }
        } else if (error != null) {
            // Error state
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.ErrorOutline,
                        contentDescription = null,
                        tint = Color(0xFFEF4444),
                        modifier = Modifier.size(64.dp)
                    )
                    Text(
                        text = error ?: "Failed to load profile",
                        color = Color.White.copy(alpha = 0.7f),
                        fontSize = 16.sp
                    )
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(12.dp))
                            .background(GradientCyan)
                            .clickable { viewModel.refresh() }
                            .padding(horizontal = 24.dp, vertical = 12.dp)
                    ) {
                        Text(
                            text = "Retry",
                            color = Color.White,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        } else {
            userProfile?.let { profile ->
                // Main content
                LazyColumn(
                    modifier = Modifier
                        .fillMaxSize()
                        .statusBarsPadding(),
                    contentPadding = PaddingValues(bottom = 100.dp)
                ) {
                    // Header with navigation
                    item { 
                        UserFuturisticHeader(
                            onNavigateBack = onNavigateBack
                        ) 
                    }
                    
                    // Profile identity section (same as ProfileScreen)
                    item { 
                        UserProfileIdentitySection(
                            username = profile.user.username,
                            role = profile.user.bio ?: "Digital Creator",
                            avatarUrl = profile.user.avatar,
                            isVerified = profile.user.isVerified
                        ) 
                    }
                    
                    // Action buttons - Follow & Message
                    item { 
                        UserActionButtonsRow(
                            isFollowing = isFollowing,
                            isFollowLoading = isFollowLoading,
                            onFollowClick = { viewModel.toggleFollow() },
                            onMessageClick = { onNavigateToChat(profile.user.userId) }
                        ) 
                    }
                    
                    // Stats section
                    item { 
                        UserFuturisticStats(
                            postsCount = profile.posts.size,
                            followersCount = profile.user.followersCount,
                            followingCount = profile.user.followingCount
                        ) 
                    }
                    
                    // Navigation tabs
                    item {
                        UserFuturisticTabs(
                            tabs = tabs,
                            selectedTab = selectedTab,
                            onTabSelected = { selectedTab = it }
                        )
                    }
                    
                    // Content based on tab
                    when (selectedTab) {
                        0 -> { // Gallery
                            item {
                                UserGallerySectionHeader()
                            }
                            
                            if (profile.posts.isEmpty()) {
                                item { UserEmptyGalleryState() }
                            } else {
                                // Grid of posts
                                items(
                                    items = profile.posts.chunked(2),
                                    key = { it.firstOrNull()?.postId ?: "" }
                                ) { rowPosts ->
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(horizontal = 16.dp),
                                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                                    ) {
                                        rowPosts.forEach { post ->
                                            UserFuturisticMediaCard(
                                                post = post,
                                                modifier = Modifier.weight(1f)
                                            )
                                        }
                                        // Fill empty space if odd number
                                        if (rowPosts.size == 1) {
                                            Spacer(modifier = Modifier.weight(1f))
                                        }
                                    }
                                    Spacer(modifier = Modifier.height(12.dp))
                                }
                            }
                        }
                        1 -> item { UserEmptyTabState("Mentions", Icons.Outlined.AlternateEmail) }
                        2 -> item { UserAboutSection(profile.user.bio ?: "") }
                    }
                }
            }
        }
    }
}

// =============================================================================
// FUTURISTIC BACKGROUND (same as ProfileScreen)
// =============================================================================

@Composable
private fun UserFuturisticBackground(rotation: Float) {
    Box(modifier = Modifier.fillMaxSize()) {
        // Purple orb top-left
        Box(
            modifier = Modifier
                .offset(x = (-100).dp, y = (-50).dp)
                .size(400.dp)
                .graphicsLayer { rotationZ = rotation * 0.3f }
                .blur(120.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            Color(0xFF581C87).copy(alpha = 0.4f),
                            Color.Transparent
                        )
                    ),
                    shape = CircleShape
                )
        )
        
        // Cyan orb bottom-right
        Box(
            modifier = Modifier
                .offset(x = 200.dp, y = 500.dp)
                .size(350.dp)
                .graphicsLayer { rotationZ = -rotation * 0.2f }
                .blur(100.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            Color(0xFF0891B2).copy(alpha = 0.3f),
                            Color.Transparent
                        )
                    ),
                    shape = CircleShape
                )
        )
    }
}

// =============================================================================
// HEADER (same style as ProfileScreen)
// =============================================================================

@Composable
private fun UserFuturisticHeader(
    onNavigateBack: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Back button
        IconButton(
            onClick = onNavigateBack,
            modifier = Modifier
                .size(44.dp)
                .clip(CircleShape)
                .background(Color.White.copy(alpha = 0.05f))
                .border(1.dp, Color.White.copy(alpha = 0.1f), CircleShape)
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "Back",
                tint = Color.Gray
            )
        }
        
        // More options button
        IconButton(
            onClick = { },
            modifier = Modifier
                .size(44.dp)
                .clip(CircleShape)
                .background(Color.White.copy(alpha = 0.05f))
                .border(1.dp, Color.White.copy(alpha = 0.1f), CircleShape)
        ) {
            Icon(
                imageVector = Icons.Filled.MoreVert,
                contentDescription = "More",
                tint = Color.Gray
            )
        }
    }
}

// =============================================================================
// PROFILE IDENTITY SECTION (same as ProfileScreen)
// =============================================================================

@Composable
private fun UserProfileIdentitySection(
    username: String,
    role: String,
    avatarUrl: String?,
    isVerified: Boolean
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Animated story ring avatar
        UserStoryRingAvatar(
            avatarUrl = avatarUrl,
            size = 120.dp,
            username = username
        )
        
        Spacer(modifier = Modifier.height(20.dp))
        
        // Username with badge
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = username,
                fontSize = 28.sp,
                fontWeight = FontWeight.ExtraBold,
                color = Color.White,
                letterSpacing = (-1).sp
            )
            if (isVerified) {
                Icon(
                    imageVector = Icons.Filled.Verified,
                    contentDescription = "Verified",
                    tint = Color(0xFFFBBF24),
                    modifier = Modifier.size(22.dp)
                )
            }
        }
        
        // Role subtitle
        Text(
            text = role.uppercase(),
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium,
            color = GradientCyan,
            letterSpacing = 2.sp,
            textAlign = TextAlign.Center
        )
        
    }
}

@Composable
private fun UserStoryRingAvatar(
    avatarUrl: String?,
    size: androidx.compose.ui.unit.Dp,
    username: String
) {
    val infiniteTransition = rememberInfiniteTransition(label = "ring")
    val ringRotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "ringRotate"
    )
    
    Box(
        modifier = Modifier.size(size + 12.dp),
        contentAlignment = Alignment.Center
    ) {
        // Animated gradient ring
        Box(
            modifier = Modifier
                .fillMaxSize()
                .rotate(ringRotation)
                .clip(CircleShape)
                .background(
                    brush = Brush.sweepGradient(
                        colors = listOf(
                            Color(0xFF6366F1),
                            Color(0xFF8B5CF6),
                            Color(0xFFFBBF24),
                            Color(0xFF00D9FF),
                            Color(0xFF6366F1)
                        )
                    )
                )
        )
        
        // Inner black border
        Box(
            modifier = Modifier
                .size(size + 4.dp)
                .clip(CircleShape)
                .background(Color(0xFF050505))
        )
        
        // Avatar image or default
        Box(
            modifier = Modifier
                .size(size)
                .clip(CircleShape)
                .background(Color(0xFF1A1A1A)),
            contentAlignment = Alignment.Center
        ) {
            if (!avatarUrl.isNullOrBlank()) {
                AsyncImage(
                    model = avatarUrl,
                    contentDescription = "Avatar",
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(CircleShape),
                    contentScale = ContentScale.Crop
                )
            } else {
                // Default avatar with initial
                DefaultAvatar(
                    name = username,
                    size = size - 8.dp
                )
            }
        }
        
        // Online indicator
        Box(
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .offset(x = (-4).dp, y = (-4).dp)
                .size(24.dp)
                .clip(CircleShape)
                .background(Color.Black)
                .padding(4.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .clip(CircleShape)
                    .background(Color(0xFF34D399))
            )
        }
    }
}

@Composable
private fun UserTagChip(tag: String) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(Color.White.copy(alpha = 0.05f))
            .border(1.dp, Color.White.copy(alpha = 0.05f), RoundedCornerShape(20.dp))
            .padding(horizontal = 14.dp, vertical = 8.dp)
    ) {
        Text(
            text = tag,
            fontSize = 11.sp,
            color = Color(0xFFD1D5DB)
        )
    }
}

// =============================================================================
// ACTION BUTTONS - Follow & Message
// =============================================================================

@Composable
private fun UserActionButtonsRow(
    isFollowing: Boolean,
    isFollowLoading: Boolean,
    onFollowClick: () -> Unit,
    onMessageClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Follow button with gradient
        val followScale by animateFloatAsState(
            targetValue = if (isFollowLoading) 0.95f else 1f,
            animationSpec = spring(dampingRatio = 0.6f),
            label = "followScale"
        )
        
        Button(
            onClick = onFollowClick,
            enabled = !isFollowLoading,
            modifier = Modifier
                .weight(1f)
                .height(48.dp)
                .scale(followScale),
            colors = ButtonDefaults.buttonColors(containerColor = Color.Transparent),
            contentPadding = PaddingValues(0.dp),
            shape = RoundedCornerShape(16.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .then(
                        if (isFollowing) {
                            Modifier
                                .background(Color.White.copy(alpha = 0.1f))
                                .border(1.dp, Color.White.copy(alpha = 0.2f), RoundedCornerShape(16.dp))
                        } else {
                            Modifier.background(
                                brush = Brush.linearGradient(
                                    colors = listOf(
                                        Color(0xFF6366F1),
                                        Color(0xFF8B5CF6)
                                    )
                                )
                            )
                        }
                    ),
                contentAlignment = Alignment.Center
            ) {
                if (isFollowLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = Color.White,
                        strokeWidth = 2.dp
                    )
                } else {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = if (isFollowing) Icons.Filled.Check else Icons.Filled.PersonAdd,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(18.dp)
                        )
                        Text(
                            text = if (isFollowing) "Following" else "Follow",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp
                        )
                    }
                }
            }
        }
        
        // Share button
        IconButton(
            onClick = onMessageClick,
            modifier = Modifier
                .size(48.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(Color.Black.copy(alpha = 0.4f))
                .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(12.dp))
        ) {
            Icon(
                imageVector = Icons.Outlined.ChatBubbleOutline,
                contentDescription = "Message",
                tint = Color(0xFFD1D5DB)
            )
        }
    }
}

// =============================================================================
// FUTURISTIC STATS (same as ProfileScreen)
// =============================================================================

@Composable
private fun UserFuturisticStats(
    postsCount: Int,
    followersCount: Int,
    followingCount: Int
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 24.dp),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        UserFuturisticStatItem(
            icon = Icons.Outlined.Podcasts,
            value = formatCount(followersCount),
            label = "Followers"
        )
        UserFuturisticStatItem(
            icon = Icons.Outlined.CenterFocusWeak,
            value = formatCount(followingCount),
            label = "Following"
        )
        UserFuturisticStatItem(
            icon = Icons.Outlined.Layers,
            value = formatCount(postsCount),
            label = "Creations"
        )
    }
}

@Composable
private fun UserFuturisticStatItem(
    icon: ImageVector,
    value: String,
    label: String
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            modifier = Modifier
                .size(32.dp)
                .padding(bottom = 4.dp),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = GradientCyan,
                modifier = Modifier.size(22.dp)
            )
        }
        Text(
            text = value,
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        Text(
            text = label.uppercase(),
            fontSize = 10.sp,
            color = Color(0xFF818CF8).copy(alpha = 0.6f),
            letterSpacing = 2.sp
        )
    }
}

// =============================================================================
// TABS (same as ProfileScreen)
// =============================================================================

@Composable
private fun UserFuturisticTabs(
    tabs: List<String>,
    selectedTab: Int,
    onTabSelected: (Int) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 16.dp)
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        tabs.forEachIndexed { index, tab ->
            Column(
                modifier = Modifier.clickable { onTabSelected(index) }
            ) {
                Text(
                    text = tab,
                    fontSize = 14.sp,
                    fontWeight = if (selectedTab == index) FontWeight.Bold else FontWeight.Normal,
                    color = if (selectedTab == index) Color.White else Color.Gray
                )
                Spacer(modifier = Modifier.height(8.dp))
                if (selectedTab == index) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(2.dp)
                            .background(
                                brush = Brush.horizontalGradient(
                                    colors = listOf(GradientCyan, Color(0xFF8B5CF6))
                                )
                            )
                    )
                }
            }
        }
    }
}

// =============================================================================
// GALLERY SECTION
// =============================================================================

@Composable
private fun UserGallerySectionHeader() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(
                text = "Latest Drops",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            Text(
                text = "Curated visual experiences",
                fontSize = 12.sp,
                color = Color.Gray
            )
        }
        Spacer(modifier = Modifier.weight(1f))
        Text(
            text = "View All",
            fontSize = 12.sp,
            color = GradientCyan,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
private fun UserFuturisticMediaCard(
    post: Post,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .height(200.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFF1A1A1A))
            .border(1.dp, Color.White.copy(alpha = 0.05f), RoundedCornerShape(16.dp))
    ) {
        // Media content
        if (!post.mediaUrl.isNullOrBlank()) {
            AsyncImage(
                model = post.mediaUrl,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
            
            // Video indicator
            if (post.mediaType == "video") {
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(8.dp)
                        .size(28.dp)
                        .clip(CircleShape)
                        .background(Color.Black.copy(alpha = 0.6f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Filled.PlayArrow,
                        contentDescription = "Video",
                        tint = Color.White,
                        modifier = Modifier.size(16.dp)
                    )
                }
            }
        } else {
            // Text post
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        brush = Brush.linearGradient(
                            colors = listOf(
                                Color(0xFF6366F1).copy(alpha = 0.3f),
                                Color(0xFF8B5CF6).copy(alpha = 0.3f)
                            )
                        )
                    ),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = post.content.take(80),
                    fontSize = 14.sp,
                    color = Color.White,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(16.dp)
                )
            }
        }
        
        // Gradient overlay
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
                .align(Alignment.BottomCenter)
                .background(
                    brush = Brush.verticalGradient(
                        colors = listOf(
                            Color.Transparent,
                            Color.Black.copy(alpha = 0.8f)
                        )
                    )
                )
        )
        
        // Stats
        Row(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Filled.Favorite,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(14.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = formatCount(post.likesCount),
                    fontSize = 12.sp,
                    color = Color.White
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Filled.ChatBubble,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(12.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = formatCount(post.commentsCount),
                    fontSize = 12.sp,
                    color = Color.White
                )
            }
        }
    }
}

// =============================================================================
// EMPTY STATES
// =============================================================================

@Composable
private fun UserEmptyGalleryState() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp)
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = Icons.Outlined.PhotoLibrary,
                contentDescription = null,
                tint = Color.Gray,
                modifier = Modifier.size(48.dp)
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "No creations yet",
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                color = Color.Gray
            )
            Text(
                text = "When they share, you'll see them here",
                fontSize = 13.sp,
                color = Color.Gray.copy(alpha = 0.6f)
            )
        }
    }
}

@Composable
private fun UserEmptyTabState(title: String, icon: ImageVector) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = GradientCyan.copy(alpha = 0.5f),
                modifier = Modifier.size(48.dp)
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "No $title yet",
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                color = Color.Gray
            )
        }
    }
}

@Composable
private fun UserAboutSection(bio: String) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(24.dp)
    ) {
        Text(
            text = "About",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        
        Spacer(modifier = Modifier.height(12.dp))
        
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(Color.White.copy(alpha = 0.05f))
                .padding(16.dp)
        ) {
            Text(
                text = bio.ifBlank { "No bio yet." },
                fontSize = 14.sp,
                color = Color.Gray,
                lineHeight = 22.sp
            )
        }
    }
}

// =============================================================================
// UTILITY
// =============================================================================

private fun formatCount(count: Int): String {
    return when {
        count >= 1000000 -> String.format("%.1fM", count / 1000000f)
        count >= 1000 -> String.format("%.1fK", count / 1000f)
        else -> count.toString()
    }
}
