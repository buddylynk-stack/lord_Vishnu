package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.lazy.grid.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import kotlinx.coroutines.launch
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.ui.components.*
import com.orignal.buddylynk.ui.theme.*
import com.orignal.buddylynk.ui.viewmodel.UserProfileViewModel

// =============================================================================
// FUTURISTIC PROFILE SCREEN - Cyberpunk inspired design
// =============================================================================

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun ProfileScreen(
    onNavigateBack: () -> Unit,
    onLogout: () -> Unit = {},
    onNavigateToEditProfile: () -> Unit = {},
    onNavigateToSettings: () -> Unit = {},
    onNavigateToNotifications: () -> Unit = {},
    onNavigateToPrivacy: () -> Unit = {},
    onNavigateToAppearance: () -> Unit = {},
    onNavigateToHelp: () -> Unit = {},
    onNavigateToBlockedUsers: () -> Unit = {},
    onNavigateToSavedPosts: () -> Unit = {},
    viewModel: UserProfileViewModel = viewModel()
) {
    val currentUser by AuthManager.currentUser.collectAsState()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val userPosts by viewModel.userPosts.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    
    // Selected tab
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Gallery", "Mentions", "Saved", "About")
    
    // Load user posts and refresh user data for latest counts
    LaunchedEffect(currentUser?.userId) {
        currentUser?.userId?.let { 
            viewModel.loadUserPosts(it)
            // Refresh user data to get latest followers/following counts
            AuthManager.refreshCurrentUser()
        }
    }
    
    // Delete post handler - deletes from DynamoDB and S3
    val handleDeletePost: (Post) -> Unit = { post ->
        scope.launch {
            try {
                // Delete from S3 first (if has media)
                post.mediaUrl?.let { url ->
                    // Extract key from URL
                    val key = url.substringAfter(".com/")
                    com.orignal.buddylynk.data.aws.S3Service.deleteFile(key)
                }
                
                // Delete from backend via API
                com.orignal.buddylynk.data.repository.BackendRepository.deletePost(post.postId)
                
                // Refresh posts
                currentUser?.userId?.let { viewModel.loadUserPosts(it) }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
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
        FuturisticBackground(orbRotation)
        
        // Main content
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding(),
            contentPadding = PaddingValues(bottom = 100.dp)
        ) {
            // Header with navigation
            item { 
                FuturisticHeader(
                    onNavigateBack = onNavigateBack,
                    onNavigateToSettings = onNavigateToSettings
                ) 
            }
            
            // Profile identity section
            item { 
                ProfileIdentitySection(
                    username = currentUser?.username ?: "User",
                    role = currentUser?.bio ?: "Digital Creator",
                    avatarUrl = currentUser?.avatar,
                    isVerified = currentUser?.isVerified ?: false,
                    onEditProfile = onNavigateToEditProfile
                ) 
            }
            
            // Action buttons
            item { ActionButtonsRow(onEditProfile = onNavigateToEditProfile) }
            
            // Stats section - using real data from currentUser
            item { 
                FuturisticStats(
                    postsCount = userPosts.size,
                    followersCount = currentUser?.followersCount ?: 0,
                    followingCount = currentUser?.followingCount ?: 0
                ) 
            }
            
            // Navigation tabs
            item {
                FuturisticTabs(
                    tabs = tabs,
                    selectedTab = selectedTab,
                    onTabSelected = { selectedTab = it }
                )
            }
            
            // Content based on tab
            when (selectedTab) {
                0 -> { // Gallery
                    item {
                        GallerySectionHeader()
                    }
                    
                    if (userPosts.isEmpty() && !isLoading) {
                        item { EmptyGalleryState() }
                    } else {
                        // Grid of posts
                        items(
                            items = userPosts.chunked(2),
                            key = { it.firstOrNull()?.postId ?: "" }
                        ) { rowPosts ->
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 16.dp),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                rowPosts.forEach { post ->
                                    FuturisticMediaCard(
                                        post = post,
                                        modifier = Modifier.weight(1f),
                                        onDelete = handleDeletePost
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
                1 -> item { EmptyTabState("Mentions", Icons.Outlined.AlternateEmail) }
                2 -> { // Saved - tap to view saved posts
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(300.dp)
                                .clickable { onNavigateToSavedPosts() },
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.BookmarkBorder,
                                    contentDescription = null,
                                    tint = GradientCyan,
                                    modifier = Modifier.size(64.dp)
                                )
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    text = "View Saved Posts",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = Color.White,
                                    fontWeight = FontWeight.SemiBold
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "Tap to see your saved posts",
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = Color.Gray
                                )
                            }
                        }
                    }
                }
                3 -> item { AboutSection(currentUser?.bio ?: "") }
            }
        }
        
        // Loading overlay
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
        }
    }
}

// =============================================================================
// FUTURISTIC BACKGROUND
// =============================================================================

@Composable
private fun FuturisticBackground(rotation: Float) {
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
        
        // Blue orb bottom-right
        Box(
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .offset(x = 100.dp, y = 100.dp)
                .size(350.dp)
                .graphicsLayer { rotationZ = -rotation * 0.2f }
                .blur(100.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            Color(0xFF1E3A8A).copy(alpha = 0.3f),
                            Color.Transparent
                        )
                    ),
                    shape = CircleShape
                )
        )
        
        // Cyan orb center
        Box(
            modifier = Modifier
                .align(Alignment.Center)
                .offset(y = 100.dp)
                .size(200.dp)
                .blur(80.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            Color(0xFF164E63).copy(alpha = 0.25f),
                            Color.Transparent
                        )
                    ),
                    shape = CircleShape
                )
        )
        
        // Grid overlay effect
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    brush = Brush.verticalGradient(
                        colors = listOf(
                            Color.Transparent,
                            Color.White.copy(alpha = 0.02f),
                            Color.Transparent
                        )
                    )
                )
        )
    }
}

// =============================================================================
// FUTURISTIC HEADER
// =============================================================================

@Composable
private fun FuturisticHeader(
    onNavigateBack: () -> Unit,
    onNavigateToSettings: () -> Unit
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
                tint = Color.White.copy(alpha = 0.8f)
            )
        }
        
        // Empty spacer for alignment (removed 3-dot menu)
        Spacer(modifier = Modifier.size(44.dp))
    }
}

// =============================================================================
// PROFILE IDENTITY SECTION
// =============================================================================

@Composable
private fun ProfileIdentitySection(
    username: String,
    role: String,
    avatarUrl: String?,
    isVerified: Boolean,
    onEditProfile: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Animated story ring avatar
        StoryRingAvatar(
            avatarUrl = avatarUrl,
            size = 120.dp,
            username = username
        )
        
        Spacer(modifier = Modifier.height(20.dp))
        
        // Username with verification
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = username,
                fontSize = 32.sp,
                fontWeight = FontWeight.ExtraBold,
                color = Color.White,
                letterSpacing = (-1).sp
            )
            if (isVerified) {
                Icon(
                    imageVector = Icons.Filled.Verified,
                    contentDescription = "Verified",
                    tint = Color(0xFFFBBF24),
                    modifier = Modifier.size(24.dp)
                )
            }
        }
        
        Spacer(modifier = Modifier.height(6.dp))
        
        // Role/Bio
        Text(
            text = role.ifBlank { "Digital Architect • Creator" },
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium,
            color = GradientCyan,
            letterSpacing = 2.sp,
            textAlign = TextAlign.Center
        )
        
    }
}

@Composable
private fun StoryRingAvatar(
    avatarUrl: String?,
    size: androidx.compose.ui.unit.Dp,
    username: String
) {
    // DEBUG: Log the values
    android.util.Log.d("StoryRingAvatar", "username=$username, avatarUrl=$avatarUrl")
    
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
    
    // Get letter and color
    val letter = if (username.isNotBlank()) username.first().uppercaseChar() else 'U'
    val letterColor = when (letter) {
        'A' -> Color(0xFFE91E63); 'B' -> Color(0xFF9C27B0); 'C' -> Color(0xFF673AB7)
        'D' -> Color(0xFF3F51B5); 'E' -> Color(0xFF2196F3); 'F' -> Color(0xFF03A9F4)
        'G' -> Color(0xFF00BCD4); 'H' -> Color(0xFF009688); 'I' -> Color(0xFF4CAF50)
        'J' -> Color(0xFF8BC34A); 'K' -> Color(0xFFCDDC39); 'L' -> Color(0xFFFFEB3B)
        'M' -> Color(0xFFFFC107); 'N' -> Color(0xFFFF9800); 'O' -> Color(0xFFFF5722)
        'P' -> Color(0xFF795548); 'Q' -> Color(0xFF607D8B); 'R' -> Color(0xFFF44336)
        'S' -> Color(0xFF9C27B0); 'T' -> Color(0xFF3F51B5); 'U' -> Color(0xFF00BCD4)
        'V' -> Color(0xFF4CAF50); 'W' -> Color(0xFFFF9800); 'X' -> Color(0xFFE91E63)
        'Y' -> Color(0xFF673AB7); 'Z' -> Color(0xFF2196F3)
        else -> Color(0xFF757575)
    }
    val hasValidImage = avatarUrl != null && avatarUrl.isNotBlank() && avatarUrl != "null"
    
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
                            Color(0xFFA855F7),
                            Color(0xFFFBBF24),
                            Color(0xFF6366F1)
                        )
                    )
                )
        )
        
        // Inner content with avatar
        Box(
            modifier = Modifier
                .size(size + 4.dp)
                .clip(CircleShape)
                .background(Color(0xFF0A0A0A))
                .padding(2.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .clip(CircleShape)
                    .background(letterColor),
                contentAlignment = Alignment.Center
            ) {
                if (hasValidImage) {
                    AsyncImage(
                        model = avatarUrl,
                        contentDescription = "Avatar",
                        modifier = Modifier.fillMaxSize().clip(CircleShape),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    Text(
                        text = letter.toString(),
                        fontSize = 48.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }
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
private fun TagChip(tag: String) {
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
// ACTION BUTTONS
// =============================================================================

@Composable
private fun ActionButtonsRow(onEditProfile: () -> Unit = {}) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Edit Profile button
        Button(
            onClick = onEditProfile,
            modifier = Modifier
                .weight(1f)
                .height(48.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color.Transparent),
            contentPadding = PaddingValues(0.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        brush = Brush.linearGradient(
                            colors = listOf(Color(0xFF4F46E5), Color(0xFF7C3AED))
                        ),
                        shape = RoundedCornerShape(12.dp)
                    )
                    .border(
                        1.dp,
                        Color(0xFF818CF8).copy(alpha = 0.3f),
                        RoundedCornerShape(12.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Edit,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        text = "Edit Profile",
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }
            }
        }
        
        // Share button
        IconButton(
            onClick = { },
            modifier = Modifier
                .size(48.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(Color.Black.copy(alpha = 0.4f))
                .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(12.dp))
        ) {
            Icon(
                imageVector = Icons.Outlined.Share,
                contentDescription = "Share",
                tint = Color(0xFFD1D5DB)
            )
        }
    }
}

// =============================================================================
// FUTURISTIC STATS
// =============================================================================

@Composable
private fun FuturisticStats(
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
        FuturisticStatItem(
            icon = Icons.Outlined.Podcasts,
            value = formatCount(followersCount),
            label = "Followers"
        )
        FuturisticStatItem(
            icon = Icons.Outlined.CenterFocusWeak,
            value = formatCount(followingCount),
            label = "Following"
        )
        FuturisticStatItem(
            icon = Icons.Outlined.Layers,
            value = formatCount(postsCount),
            label = "Creations"
        )
    }
}

@Composable
private fun FuturisticStatItem(
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

private fun formatCount(count: Int): String {
    return when {
        count >= 1000000 -> String.format("%.1fM", count / 1000000f)
        count >= 1000 -> String.format("%.1fK", count / 1000f)
        else -> count.toString()
    }
}

// =============================================================================
// FUTURISTIC TABS
// =============================================================================

@Composable
private fun FuturisticTabs(
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
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp,
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
private fun GallerySectionHeader() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Latest Drops",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                // Animated live dot
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(Color.Red)
                )
            }
            Text(
                text = "Curated visual experiences",
                fontSize = 13.sp,
                color = Color.Gray
            )
        }
        
        Text(
            text = "View All",
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            color = GradientCyan,
            letterSpacing = 1.sp
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun FuturisticMediaCard(
    post: Post,
    modifier: Modifier = Modifier,
    onDelete: ((Post) -> Unit)? = null,
    onTogglePrivate: ((Post) -> Unit)? = null
) {
    var showMenu by remember { mutableStateOf(false) }
    var showDeleteDialog by remember { mutableStateOf(false) }
    
    Box(
        modifier = modifier
            .height(200.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFF1A1A1A))
            .border(1.dp, Color.White.copy(alpha = 0.05f), RoundedCornerShape(16.dp))
            .combinedClickable(
                onClick = { /* Open post detail */ },
                onLongClick = { showMenu = true }
            )
    ) {
        // Image
        if (!post.mediaUrl.isNullOrBlank()) {
            AsyncImage(
                model = post.mediaUrl,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        }
        
        // Gradient overlay
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    brush = Brush.verticalGradient(
                        colors = listOf(
                            Color.Transparent,
                            Color.Black.copy(alpha = 0.7f),
                            Color.Black.copy(alpha = 0.9f)
                        ),
                        startY = 0f,
                        endY = Float.POSITIVE_INFINITY
                    )
                )
        )
        
        // Content
        Column(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(12.dp)
        ) {
            // Category tag
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(2.dp))
                    .background(GradientCyan)
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            ) {
                Text(
                    text = post.mediaType?.uppercase() ?: "POST",
                    fontSize = 8.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black,
                    letterSpacing = 1.sp
                )
            }
            
            Spacer(modifier = Modifier.height(6.dp))
            
            // Content preview
            Text(
                text = post.content.take(50) + if (post.content.length > 50) "..." else "",
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Stats row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.FavoriteBorder,
                        contentDescription = null,
                        tint = Color.White.copy(alpha = 0.7f),
                        modifier = Modifier.size(14.dp)
                    )
                    Text(
                        text = post.likesCount.toString(),
                        fontSize = 12.sp,
                        color = Color.White.copy(alpha = 0.7f)
                    )
                }
                
                Box(
                    modifier = Modifier
                        .size(28.dp)
                        .clip(CircleShape)
                        .background(Color.White.copy(alpha = 0.1f))
                        .border(1.dp, Color.White.copy(alpha = 0.2f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Filled.Bolt,
                        contentDescription = null,
                        tint = Color(0xFFFBBF24),
                        modifier = Modifier.size(14.dp)
                    )
                }
            }
        }
        
        // Long-press dropdown menu
        DropdownMenu(
            expanded = showMenu,
            onDismissRequest = { showMenu = false },
            modifier = Modifier.background(Color(0xFF1A1A2E))
        ) {
            DropdownMenuItem(
                text = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.Lock,
                            contentDescription = null,
                            tint = Color.White
                        )
                        Text("Make Private", color = Color.White)
                    }
                },
                onClick = {
                    showMenu = false
                    onTogglePrivate?.invoke(post)
                }
            )
            DropdownMenuItem(
                text = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.Delete,
                            contentDescription = null,
                            tint = Color(0xFFEF4444)
                        )
                        Text("Delete", color = Color(0xFFEF4444))
                    }
                },
                onClick = {
                    showMenu = false
                    showDeleteDialog = true
                }
            )
        }
    }
    
    // Delete confirmation dialog
    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            containerColor = Color(0xFF1A1A2E),
            title = {
                Text(
                    text = "Delete Post?",
                    color = Color.White,
                    fontWeight = FontWeight.Bold
                )
            },
            text = {
                Text(
                    text = "This action cannot be undone. The post and its media will be permanently deleted.",
                    color = Color.White.copy(alpha = 0.7f)
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showDeleteDialog = false
                        onDelete?.invoke(post)
                    }
                ) {
                    Text("Delete", color = Color(0xFFEF4444), fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) {
                    Text("Cancel", color = Color.White)
                }
            }
        )
    }
}

// =============================================================================
// ACTIVITY SECTION
// =============================================================================

@Composable
private fun ActivitySection() {
    Column(
        modifier = Modifier.padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = "Recent Activity",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        
        // Activity cards
        ActivityCard(
            icon = Icons.Filled.Bolt,
            iconColor = Color(0xFFA855F7),
            title = "New Milestone Reached",
            description = "Your content has reached new heights! Keep creating.",
            time = "2h ago"
        )
        
        ActivityCard(
            icon = Icons.Filled.EmojiEvents,
            iconColor = GradientCyan,
            title = "Top Creator Badge",
            description = "Awarded for outstanding engagement this week.",
            time = "5h ago"
        )
    }
}

@Composable
private fun ActivityCard(
    icon: ImageVector,
    iconColor: Color,
    title: String,
    description: String,
    time: String
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color.White.copy(alpha = 0.05f))
            .border(1.dp, Color.White.copy(alpha = 0.05f), RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(iconColor.copy(alpha = 0.2f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = icon,
                        contentDescription = null,
                        tint = iconColor,
                        modifier = Modifier.size(20.dp)
                    )
                }
                
                Text(
                    text = time,
                    fontSize = 11.sp,
                    color = Color.Gray
                )
            }
            
            Spacer(modifier = Modifier.height(12.dp))
            
            Text(
                text = title,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            
            Spacer(modifier = Modifier.height(4.dp))
            
            Text(
                text = description,
                fontSize = 13.sp,
                color = Color.Gray,
                lineHeight = 18.sp
            )
        }
    }
}

// =============================================================================
// EMPTY STATES
// =============================================================================

@Composable
private fun EmptyGalleryState() {
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
                text = "Start creating to fill your gallery",
                fontSize = 13.sp,
                color = Color.Gray.copy(alpha = 0.6f)
            )
        }
    }
}

@Composable
private fun EmptyTabState(tabName: String, icon: ImageVector) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(300.dp)
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = Color.Gray,
                modifier = Modifier.size(48.dp)
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "No $tabName",
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                color = Color.Gray
            )
        }
    }
}

@Composable
private fun AboutSection(bio: String) {
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
                text = bio.ifBlank { "No bio yet. Tap edit profile to add one!" },
                fontSize = 14.sp,
                color = Color.Gray,
                lineHeight = 22.sp
            )
        }
    }
}
