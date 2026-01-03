package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.orignal.buddylynk.data.moderation.ModerationService
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.data.model.User
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.ui.components.*
import com.orignal.buddylynk.ui.theme.*
import kotlinx.coroutines.launch

// =============================================================================
// NOTIFICATIONS SETTINGS SCREEN
// =============================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NotificationsSettingsScreen(onNavigateBack: () -> Unit) {
    var pushEnabled by remember { mutableStateOf(true) }
    var likesEnabled by remember { mutableStateOf(true) }
    var commentsEnabled by remember { mutableStateOf(true) }
    var followsEnabled by remember { mutableStateOf(true) }
    var messagesEnabled by remember { mutableStateOf(true) }
    var mentionsEnabled by remember { mutableStateOf(true) }
    var emailEnabled by remember { mutableStateOf(false) }
    
    AnimatedGradientBackground(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            // Header
            SettingsHeader(title = "Notifications", onNavigateBack = onNavigateBack)
            
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    Text(
                        text = "Push Notifications",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.Notifications,
                        title = "Push Notifications",
                        subtitle = "Receive push notifications",
                        isEnabled = pushEnabled,
                        onToggle = { pushEnabled = it }
                    )
                }
                
                item {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Activity Notifications",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.Favorite,
                        title = "Likes",
                        subtitle = "When someone likes your post",
                        isEnabled = likesEnabled,
                        onToggle = { likesEnabled = it }
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.ChatBubbleOutline,
                        title = "Comments",
                        subtitle = "When someone comments on your post",
                        isEnabled = commentsEnabled,
                        onToggle = { commentsEnabled = it }
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.PersonAdd,
                        title = "New Followers",
                        subtitle = "When someone follows you",
                        isEnabled = followsEnabled,
                        onToggle = { followsEnabled = it }
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.Email,
                        title = "Messages",
                        subtitle = "When you receive a new message",
                        isEnabled = messagesEnabled,
                        onToggle = { messagesEnabled = it }
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.AlternateEmail,
                        title = "Mentions",
                        subtitle = "When someone mentions you",
                        isEnabled = mentionsEnabled,
                        onToggle = { mentionsEnabled = it }
                    )
                }
                
                item {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Email Notifications",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.MarkEmailRead,
                        title = "Email Updates",
                        subtitle = "Receive weekly digest emails",
                        isEnabled = emailEnabled,
                        onToggle = { emailEnabled = it }
                    )
                }
                
                item { Spacer(modifier = Modifier.height(80.dp)) }
            }
        }
    }
}

// =============================================================================
// PRIVACY SETTINGS SCREEN
// =============================================================================

@Composable
fun PrivacySettingsScreen(
    onNavigateBack: () -> Unit,
    onNavigateToBlockedUsers: () -> Unit = {}
) {
    var privateAccount by remember { mutableStateOf(false) }
    var showOnlineStatus by remember { mutableStateOf(true) }
    var showLastSeen by remember { mutableStateOf(true) }
    var allowTagging by remember { mutableStateOf(true) }
    var allowMentions by remember { mutableStateOf(true) }
    var showActivity by remember { mutableStateOf(true) }
    
    AnimatedGradientBackground(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            SettingsHeader(title = "Privacy", onNavigateBack = onNavigateBack)
            
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    Text(
                        text = "Account Privacy",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.Lock,
                        title = "Private Account",
                        subtitle = "Only followers can see your posts",
                        isEnabled = privateAccount,
                        onToggle = { privateAccount = it }
                    )
                }
                
                item {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Activity Status",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Filled.Circle,
                        title = "Show Online Status",
                        subtitle = "Let others see when you're online",
                        isEnabled = showOnlineStatus,
                        onToggle = { showOnlineStatus = it }
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.AccessTime,
                        title = "Show Last Seen",
                        subtitle = "Show when you were last active",
                        isEnabled = showLastSeen,
                        onToggle = { showLastSeen = it }
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.Visibility,
                        title = "Activity Status",
                        subtitle = "Show your activity to others",
                        isEnabled = showActivity,
                        onToggle = { showActivity = it }
                    )
                }
                
                item {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Interactions",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.Tag,
                        title = "Allow Tagging",
                        subtitle = "Let others tag you in posts",
                        isEnabled = allowTagging,
                        onToggle = { allowTagging = it }
                    )
                }
                
                item {
                    SettingsToggleItem(
                        icon = Icons.Outlined.AlternateEmail,
                        title = "Allow Mentions",
                        subtitle = "Let others mention you",
                        isEnabled = allowMentions,
                        onToggle = { allowMentions = it }
                    )
                }
                
                item {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Blocked Users",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }
                
                item {
                    GlassCard(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable(onClick = onNavigateToBlockedUsers),
                        cornerRadius = 16.dp,
                        glassOpacity = 0.06f
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(14.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.Block,
                                    contentDescription = null,
                                    tint = LikeRed,
                                    modifier = Modifier.size(24.dp)
                                )
                                Column {
                                    Text(
                                        text = "Blocked Users",
                                        style = MaterialTheme.typography.bodyLarge,
                                        color = TextPrimary
                                    )
                                    Text(
                                        text = "Manage blocked accounts",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = TextSecondary
                                    )
                                }
                            }
                            Icon(
                                imageVector = Icons.Filled.ChevronRight,
                                contentDescription = null,
                                tint = TextTertiary
                            )
                        }
                    }
                }
                
                item { Spacer(modifier = Modifier.height(80.dp)) }
            }
        }
    }
}

// =============================================================================
// BLOCKED USERS SCREEN
// =============================================================================

@Composable
fun BlockedUsersScreen(onNavigateBack: () -> Unit) {
    val scope = rememberCoroutineScope()
    var blockedUsers by remember { mutableStateOf<List<User>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var showUnblockDialog by remember { mutableStateOf<User?>(null) }
    
    // Load blocked users
    LaunchedEffect(Unit) {
        isLoading = true
        blockedUsers = ModerationService.getBlockedUsers()
        isLoading = false
    }
    
    // Unblock confirmation dialog
    showUnblockDialog?.let { user ->
        AlertDialog(
            onDismissRequest = { showUnblockDialog = null },
            containerColor = DarkSurface,
            titleContentColor = TextPrimary,
            textContentColor = TextSecondary,
            title = { Text("Unblock ${user.username}?", fontWeight = FontWeight.Bold) },
            text = { Text("They will be able to see your profile and posts again.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        scope.launch {
                            ModerationService.unblockUser(user.userId)
                            blockedUsers = blockedUsers.filter { it.userId != user.userId }
                        }
                        showUnblockDialog = null
                    }
                ) {
                    Text("Unblock", color = GradientCyan)
                }
            },
            dismissButton = {
                TextButton(onClick = { showUnblockDialog = null }) {
                    Text("Cancel", color = TextSecondary)
                }
            }
        )
    }
    
    AnimatedGradientBackground(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            SettingsHeader(title = "Blocked Users", onNavigateBack = onNavigateBack)
            
            if (isLoading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = GradientCyan)
                }
            } else if (blockedUsers.isEmpty()) {
                // Empty state
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.Block,
                            contentDescription = null,
                            tint = TextTertiary,
                            modifier = Modifier.size(64.dp)
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "No Blocked Users",
                            style = MaterialTheme.typography.titleMedium,
                            color = TextPrimary,
                            fontWeight = FontWeight.SemiBold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "You haven't blocked anyone yet",
                            style = MaterialTheme.typography.bodyMedium,
                            color = TextSecondary
                        )
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    item {
                        Text(
                            text = "${blockedUsers.size} blocked user${if (blockedUsers.size != 1) "s" else ""}",
                            style = MaterialTheme.typography.bodyMedium,
                            color = TextSecondary
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                    
                    items(blockedUsers.size) { index ->
                        val user = blockedUsers[index]
                        GlassCard(
                            modifier = Modifier.fillMaxWidth(),
                            cornerRadius = 16.dp,
                            glassOpacity = 0.06f
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(
                                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    // User avatar
                                    Box(
                                        modifier = Modifier
                                            .size(44.dp)
                                            .clip(CircleShape)
                                            .background(Brush.linearGradient(PremiumGradient)),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Text(
                                            text = user.username.firstOrNull()?.uppercase() ?: "U",
                                            style = MaterialTheme.typography.titleSmall,
                                            fontWeight = FontWeight.Bold,
                                            color = Color.White
                                        )
                                    }
                                    
                                    Column {
                                        Text(
                                            text = user.username,
                                            style = MaterialTheme.typography.bodyLarge,
                                            fontWeight = FontWeight.SemiBold,
                                            color = TextPrimary
                                        )
                                        Text(
                                            text = "@${user.username.lowercase()}",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = TextSecondary
                                        )
                                    }
                                }
                                
                                // Unblock button
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(8.dp))
                                        .background(LikeRed.copy(alpha = 0.2f))
                                        .clickable { showUnblockDialog = user }
                                        .padding(horizontal = 12.dp, vertical = 6.dp)
                                ) {
                                    Text(
                                        text = "Unblock",
                                        style = MaterialTheme.typography.labelMedium,
                                        color = LikeRed,
                                        fontWeight = FontWeight.Medium
                                    )
                                }
                            }
                        }
                    }
                    
                    item { Spacer(modifier = Modifier.height(80.dp)) }
                }
            }
        }
    }
}

// =============================================================================
// SAVED POSTS SCREEN
// =============================================================================

@Composable
fun SavedPostsScreen(
    onNavigateBack: () -> Unit,
    onNavigateToComments: (String) -> Unit = {},
    onNavigateToProfile: (String) -> Unit = {}
) {
    val scope = rememberCoroutineScope()
    var savedPosts by remember { mutableStateOf<List<Post>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    val currentUserId = com.orignal.buddylynk.data.auth.AuthManager.currentUser.value?.userId ?: ""
    
    // Load saved posts
    LaunchedEffect(Unit) {
        isLoading = true
        try {
            // Get saved post IDs from API
            val savedIds = BackendRepository.getSavedPostIds()
            
            if (savedIds.isNotEmpty()) {
                // Get all feed posts and filter to just the saved ones
                val allPosts = BackendRepository.getFeedPosts()
                val savedPostsList = allPosts.filter { post -> 
                    savedIds.contains(post.postId) 
                }
                savedPosts = savedPostsList
            }
        } catch (e: Exception) {
            android.util.Log.e("SavedPostsScreen", "Error loading saved posts: ${e.message}")
        }
        isLoading = false
    }
    
    AnimatedGradientBackground(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            SettingsHeader(title = "Saved Posts", onNavigateBack = onNavigateBack)
            
            if (isLoading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = GradientCyan)
                }
            } else if (savedPosts.isEmpty()) {
                // Empty state
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.BookmarkBorder,
                            contentDescription = null,
                            tint = TextTertiary,
                            modifier = Modifier.size(64.dp)
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "No Saved Posts",
                            style = MaterialTheme.typography.titleMedium,
                            color = TextPrimary,
                            fontWeight = FontWeight.SemiBold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Tap the bookmark icon to save posts",
                            style = MaterialTheme.typography.bodyMedium,
                            color = TextSecondary
                        )
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    item {
                        Text(
                            text = "${savedPosts.size} saved post${if (savedPosts.size != 1) "s" else ""}",
                            style = MaterialTheme.typography.bodyMedium,
                            color = TextSecondary
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                    
                    items(savedPosts.size) { index ->
                        val post = savedPosts[index]
                        SavedPostCard(
                            post = post,
                            onUnsave = {
                                scope.launch {
                                    BackendRepository.unsavePost(post.postId)
                                    savedPosts = savedPosts.filter { it.postId != post.postId }
                                }
                            },
                            onClick = { onNavigateToProfile(post.userId) }
                        )
                    }
                    
                    item { Spacer(modifier = Modifier.height(80.dp)) }
                }
            }
        }
    }
}

@Composable
private fun SavedPostCard(
    post: Post,
    onUnsave: () -> Unit,
    onClick: () -> Unit
) {
    GlassCard(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        cornerRadius = 16.dp,
        glassOpacity = 0.06f
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Post thumbnail or avatar
            Box(
                modifier = Modifier
                    .size(60.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .then(
                        if (post.mediaUrl != null) Modifier.background(Color.Transparent)
                        else Modifier.background(Brush.linearGradient(PremiumGradient))
                    ),
                contentAlignment = Alignment.Center
            ) {
                if (post.mediaUrl != null) {
                    coil.compose.AsyncImage(
                        model = post.mediaUrl,
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = androidx.compose.ui.layout.ContentScale.Crop
                    )
                } else {
                    Text(
                        text = post.username?.firstOrNull()?.uppercase() ?: "P",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }
            }
            
            // Post info
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = post.username ?: "User",
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary,
                    maxLines = 1
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = post.content,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                    maxLines = 2,
                    overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                )
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "❤ ${post.likesCount}",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextTertiary
                    )
                    Text(
                        text = "💬 ${post.commentsCount}",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextTertiary
                    )
                }
            }
            
            // Unsave button
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(LikeRed.copy(alpha = 0.2f))
                    .clickable { onUnsave() },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.BookmarkRemove,
                    contentDescription = "Unsave",
                    tint = LikeRed,
                    modifier = Modifier.size(20.dp)
                )
            }
        }
    }
}

// =============================================================================
// APPEARANCE SETTINGS SCREEN - Theme Switcher
// =============================================================================

@Composable
fun AppearanceSettingsScreen(onNavigateBack: () -> Unit) {
    var currentTheme by remember { mutableStateOf(ThemeManager.currentTheme) }
    
    AnimatedGradientBackground(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            SettingsHeader(title = "Appearance", onNavigateBack = onNavigateBack)
            
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    Text(
                        text = "Theme",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }
                
                item {
                    ThemeOptionCard(
                        icon = "🌙",
                        title = "Dark Mode",
                        subtitle = "Rich dark colors, easy on eyes",
                        isSelected = currentTheme == ThemeMode.DARK,
                        gradient = listOf(Color(0xFF0F0F13), Color(0xFF18181D)),
                        onClick = {
                            ThemeManager.setTheme(ThemeMode.DARK)
                            currentTheme = ThemeMode.DARK
                        }
                    )
                }
                
                item {
                    ThemeOptionCard(
                        icon = "🧊",
                        title = "Frosted Glass",
                        subtitle = "Premium glassmorphism effect",
                        isSelected = currentTheme == ThemeMode.FROSTED_GLASS,
                        gradient = listOf(Color(0xFF0A1628), Color(0xFF00D4FF).copy(alpha = 0.2f)),
                        onClick = {
                            ThemeManager.setTheme(ThemeMode.FROSTED_GLASS)
                            currentTheme = ThemeMode.FROSTED_GLASS
                        }
                    )
                }
                
                item {
                    ThemeOptionCard(
                        icon = "🌄",
                        title = "Nature",
                        subtitle = "Warm earthy tones",
                        isSelected = currentTheme == ThemeMode.NATURE,
                        gradient = listOf(Color(0xFF1A1512), Color(0xFFE8A87C).copy(alpha = 0.2f)),
                        onClick = {
                            ThemeManager.setTheme(ThemeMode.NATURE)
                            currentTheme = ThemeMode.NATURE
                        }
                    )
                }
                
                item { Spacer(modifier = Modifier.height(80.dp)) }
            }
        }
    }
}

@Composable
private fun ThemeOptionCard(
    icon: String,
    title: String,
    subtitle: String,
    isSelected: Boolean,
    gradient: List<Color>,
    onClick: () -> Unit
) {
    val borderColor by animateColorAsState(
        targetValue = if (isSelected) GradientCoral else GlassBorder,
        animationSpec = tween(300),
        label = "border"
    )
    
    GlassCard(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        cornerRadius = 20.dp,
        glassOpacity = if (isSelected) 0.15f else 0.08f,
        animatedBorder = isSelected
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Theme preview
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Brush.linearGradient(gradient))
                        .border(1.dp, borderColor, RoundedCornerShape(12.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(text = icon, style = MaterialTheme.typography.headlineSmall)
                }
                
                Column {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                }
            }
            
            // Selected indicator
            AnimatedVisibility(
                visible = isSelected,
                enter = scaleIn() + fadeIn(),
                exit = scaleOut() + fadeOut()
            ) {
                Box(
                    modifier = Modifier
                        .size(24.dp)
                        .clip(CircleShape)
                        .background(Brush.linearGradient(PremiumGradient)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Filled.Check,
                        contentDescription = "Selected",
                        tint = Color.White,
                        modifier = Modifier.size(16.dp)
                    )
                }
            }
        }
    }
}

// =============================================================================
// HELP SCREEN
// =============================================================================

@Composable
fun HelpScreen(onNavigateBack: () -> Unit) {
    val helpItems = listOf(
        "Getting Started" to Icons.Outlined.PlayCircle,
        "Account Settings" to Icons.Outlined.ManageAccounts,
        "Privacy & Security" to Icons.Outlined.Security,
        "Posting & Sharing" to Icons.Outlined.Share,
        "Messages & Chats" to Icons.Outlined.Chat,
        "Groups & Communities" to Icons.Outlined.Groups,
        "Report a Problem" to Icons.Outlined.BugReport,
        "Contact Support" to Icons.Outlined.SupportAgent
    )
    
    AnimatedGradientBackground(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            SettingsHeader(title = "Help & Support", onNavigateBack = onNavigateBack)
            
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                item {
                    // Search help
                    GlassCard(
                        modifier = Modifier.fillMaxWidth(),
                        cornerRadius = 16.dp,
                        glassOpacity = 0.1f
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                imageVector = Icons.Outlined.Search,
                                contentDescription = null,
                                tint = TextSecondary
                            )
                            Spacer(modifier = Modifier.width(12.dp))
                            Text(
                                text = "Search help topics...",
                                style = MaterialTheme.typography.bodyLarge,
                                color = TextTertiary
                            )
                        }
                    }
                }
                
                item {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Help Topics",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }
                
                helpItems.forEach { (title, icon) ->
                    item {
                        GlassCard(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { /* Open help topic */ },
                            cornerRadius = 16.dp,
                            glassOpacity = 0.06f
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(
                                    horizontalArrangement = Arrangement.spacedBy(14.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        imageVector = icon,
                                        contentDescription = null,
                                        tint = GradientCoral,
                                        modifier = Modifier.size(24.dp)
                                    )
                                    Text(
                                        text = title,
                                        style = MaterialTheme.typography.bodyLarge,
                                        color = TextPrimary
                                    )
                                }
                                Icon(
                                    imageVector = Icons.Filled.ChevronRight,
                                    contentDescription = null,
                                    tint = TextTertiary
                                )
                            }
                        }
                    }
                }
                
                item { Spacer(modifier = Modifier.height(80.dp)) }
            }
        }
    }
}

// =============================================================================
// SHARED COMPONENTS
// =============================================================================

@Composable
private fun SettingsHeader(title: String, onNavigateBack: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        IconButton(onClick = onNavigateBack) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "Back",
                tint = TextPrimary
            )
        }
        Text(
            text = title,
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = TextPrimary
        )
    }
}

@Composable
private fun SettingsToggleItem(
    icon: ImageVector,
    title: String,
    subtitle: String,
    isEnabled: Boolean,
    onToggle: (Boolean) -> Unit
) {
    GlassCard(
        modifier = Modifier.fillMaxWidth(),
        cornerRadius = 16.dp,
        glassOpacity = 0.06f
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(14.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = if (isEnabled) GradientCoral else TextTertiary,
                    modifier = Modifier.size(24.dp)
                )
                Column {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.bodyLarge,
                        color = TextPrimary
                    )
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                }
            }
            
            Switch(
                checked = isEnabled,
                onCheckedChange = onToggle,
                colors = SwitchDefaults.colors(
                    checkedThumbColor = Color.White,
                    checkedTrackColor = GradientCoral,
                    uncheckedThumbColor = TextTertiary,
                    uncheckedTrackColor = GlassBorder
                )
            )
        }
    }
}
