package com.orignal.buddylynk.ui.screens

import android.content.Intent
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.data.model.User
import com.orignal.buddylynk.ui.components.PremiumPostCard
import com.orignal.buddylynk.ui.viewmodel.SearchViewModel

// Premium Colors
private val DarkBg = Color(0xFF050505)
private val CardBg = Color(0xFF1A1A1A)
private val BorderWhite5 = Color.White.copy(alpha = 0.05f)
private val BorderWhite10 = Color.White.copy(alpha = 0.1f)
private val IndigoAccent = Color(0xFF6366F1)
private val FuchsiaAccent = Color(0xFFEC4899)

// Categories matching React
private val CATEGORIES = listOf("All", "Teams", "Event", "Trending")

/**
 * Premium Search Screen matching React design
 */
@Composable
fun PremiumSearchScreen(
    onNavigateBack: () -> Unit,
    onNavigateToProfile: (String) -> Unit = {},
    viewModel: SearchViewModel = viewModel()
) {
    val context = LocalContext.current
    val searchQuery by viewModel.searchQuery.collectAsState()
    val selectedCategory by viewModel.selectedCategory.collectAsState()
    val suggestedUsers by viewModel.suggestedUsers.collectAsState()
    val trendingPosts by viewModel.trendingPosts.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    
    // Search results
    val searchedUsers by viewModel.users.collectAsState()
    val searchedPosts by viewModel.posts.collectAsState()
    val isSearching by viewModel.isSearching.collectAsState()
    
    // Debug logging
    LaunchedEffect(suggestedUsers.size, trendingPosts.size, isLoading, searchedUsers.size) {
        android.util.Log.d("PremiumSearchScreen", "UI State: users=${suggestedUsers.size}, posts=${trendingPosts.size}, loading=$isLoading, searchedUsers=${searchedUsers.size}")
    }
    
    var isScrolled by remember { mutableStateOf(false) }
    val likedPosts = remember { mutableStateListOf<String>() }
    val savedPosts = remember { mutableStateListOf<String>() }
    val followingUsers = remember { mutableStateListOf<String>() }
    
    // Keep screen awake while on search page
    val activity = context as? android.app.Activity
    DisposableEffect(Unit) {
        activity?.window?.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            activity?.window?.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBg)
    ) {
        // Ambient Background
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(400.dp)
                .blur(150.dp)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            IndigoAccent.copy(alpha = 0.1f),
                            Color.Transparent
                        )
                    )
                )
        )
        Box(
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .size(320.dp)
                .blur(150.dp)
                .background(FuchsiaAccent.copy(alpha = 0.1f))
        )

        // Main Content
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(bottom = 100.dp)
        ) {
            // Sticky Search Header
            item {
                PremiumSearchHeader(
                    query = searchQuery,
                    onQueryChange = { viewModel.updateSearchQuery(it) },
                    selectedCategory = selectedCategory,
                    onCategorySelect = { viewModel.selectCategory(it) },
                    isScrolled = isScrolled
                )
            }

            // ====== SEARCH RESULTS SECTION ======
            if (searchQuery.isNotBlank()) {
                // Show search loading indicator
                if (isSearching) {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(32.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                CircularProgressIndicator(
                                    color = IndigoAccent,
                                    modifier = Modifier.size(32.dp),
                                    strokeWidth = 3.dp
                                )
                                Text(
                                    text = "Searching...",
                                    color = Color(0xFF71717A),
                                    fontSize = 14.sp
                                )
                            }
                        }
                    }
                }
                
                // Users search results
                if (searchedUsers.isNotEmpty()) {
                    item {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(start = 16.dp, top = 20.dp, bottom = 12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Filled.People,
                                contentDescription = null,
                                tint = IndigoAccent,
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                text = "USERS (${searchedUsers.size})",
                                color = Color(0xFFA1A1AA),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                        }
                    }
                    
                    // User cards
                    items(searchedUsers, key = { it.userId }) { user ->
                        SearchUserCard(
                            user = user,
                            isFollowing = followingUsers.contains(user.userId),
                            onClick = { onNavigateToProfile(user.userId) },
                            onFollowClick = {
                                if (followingUsers.contains(user.userId)) {
                                    followingUsers.remove(user.userId)
                                } else {
                                    followingUsers.add(user.userId)
                                }
                                viewModel.followUser(user.userId)
                            }
                        )
                    }
                }
                
                // Posts search results
                if (searchedPosts.isNotEmpty()) {
                    item {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(start = 16.dp, top = 24.dp, bottom = 12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Filled.Article,
                                contentDescription = null,
                                tint = Color(0xFFF97316),
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                text = "POSTS (${searchedPosts.size})",
                                color = Color(0xFFA1A1AA),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                        }
                    }
                    
                    // Post cards
                    items(searchedPosts, key = { it.postId }) { post ->
                        Column(modifier = Modifier.padding(horizontal = 8.dp)) {
                            PremiumPostCard(
                                post = post,
                                isLiked = likedPosts.contains(post.postId),
                                isSaved = savedPosts.contains(post.postId),
                                hasStatus = false,
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
                                    viewModel.savePost(post.postId)
                                },
                                onComment = { },
                                onShare = {
                                    val sendIntent = Intent().apply {
                                        action = Intent.ACTION_SEND
                                        putExtra(Intent.EXTRA_TEXT, "Check out this post on BuddyLynk!")
                                        type = "text/plain"
                                    }
                                    context.startActivity(Intent.createChooser(sendIntent, null))
                                },
                                onMoreClick = { }
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                        }
                    }
                }
                
                // No results state
                if (!isSearching && searchedUsers.isEmpty() && searchedPosts.isEmpty()) {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(48.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.SearchOff,
                                    contentDescription = null,
                                    tint = Color(0xFF52525B),
                                    modifier = Modifier.size(48.dp)
                                )
                                Text(
                                    text = "No results for \"$searchQuery\"",
                                    color = Color(0xFF71717A),
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.Medium
                                )
                                Text(
                                    text = "Try another search term",
                                    color = Color(0xFF52525B),
                                    fontSize = 14.sp
                                )
                            }
                        }
                    }
                }
            }

            // ====== SUGGESTED USERS (only when not searching) ======
            if (searchQuery.isBlank() && suggestedUsers.isNotEmpty()) {
                item {
                    SuggestedUsersSection(
                        users = suggestedUsers,
                        followingUsers = followingUsers,
                        onUserClick = { onNavigateToProfile(it.userId) },
                        onFollowClick = { userId ->
                            if (followingUsers.contains(userId)) {
                                followingUsers.remove(userId)
                            } else {
                                followingUsers.add(userId)
                            }
                            viewModel.followUser(userId)
                        }
                    )
                }
            }

            // Trending Posts Header
            if (trendingPosts.isNotEmpty()) {
                item {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(start = 16.dp, top = 24.dp, bottom = 16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.LocalFireDepartment,
                            contentDescription = null,
                            tint = Color(0xFFF97316),
                            modifier = Modifier.size(16.dp)
                        )
                        Text(
                            text = "TRENDING POSTS",
                            color = Color(0xFFA1A1AA),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        )
                    }
                }

                // Trending Posts Feed
                itemsIndexed(trendingPosts) { index, post ->
                    Column(
                        modifier = Modifier.padding(horizontal = 8.dp)
                    ) {
                        PremiumPostCard(
                            post = post,
                            isLiked = likedPosts.contains(post.postId),
                            isSaved = savedPosts.contains(post.postId),
                            hasStatus = false,
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
                                viewModel.savePost(post.postId)
                            },
                            onComment = { /* TODO */ },
                            onShare = {
                                val sendIntent = Intent().apply {
                                    action = Intent.ACTION_SEND
                                    putExtra(Intent.EXTRA_TEXT, "Check out this post on BuddyLynk!")
                                    type = "text/plain"
                                }
                                context.startActivity(Intent.createChooser(sendIntent, null))
                            },
                            onMoreClick = { /* TODO */ }
                        )

                        Spacer(modifier = Modifier.height(16.dp))
                    }
                }
            }

            // Loading state - Premium skeleton loading
            if (isLoading) {
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 24.dp)
                    ) {
                        // Header skeleton
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            ShimmerBox(
                                modifier = Modifier
                                    .size(16.dp)
                                    .clip(CircleShape)
                            )
                            ShimmerBox(
                                modifier = Modifier
                                    .width(120.dp)
                                    .height(12.dp)
                                    .clip(RoundedCornerShape(6.dp))
                            )
                        }
                        
                        Spacer(modifier = Modifier.height(20.dp))
                        
                        // Skeleton post cards
                        repeat(2) {
                            SkeletonPostCard()
                            Spacer(modifier = Modifier.height(16.dp))
                        }
                    }
                }
            }
            
            // Empty state when no content
            if (!isLoading && trendingPosts.isEmpty() && searchQuery.isBlank()) {
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(32.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            // Icon with gradient background
                            Box(
                                modifier = Modifier
                                    .size(80.dp)
                                    .clip(CircleShape)
                                    .background(
                                        Brush.linearGradient(
                                            colors = listOf(
                                                IndigoAccent.copy(alpha = 0.2f),
                                                FuchsiaAccent.copy(alpha = 0.2f)
                                            )
                                        )
                                    ),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.Explore,
                                    contentDescription = null,
                                    tint = IndigoAccent,
                                    modifier = Modifier.size(40.dp)
                                )
                            }
                            
                            Text(
                                text = "Explore BuddyLynk",
                                color = Color.White,
                                fontSize = 20.sp,
                                fontWeight = FontWeight.Bold
                            )
                            
                            Text(
                                text = "Discover new people, trending posts,\nand communities",
                                color = Color(0xFF71717A),
                                fontSize = 14.sp,
                                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                                lineHeight = 20.sp
                            )
                            
                            // Refresh button
                            TextButton(
                                onClick = { viewModel.refresh() },
                                modifier = Modifier.padding(top = 8.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Filled.Refresh,
                                    contentDescription = null,
                                    tint = IndigoAccent,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = "Refresh",
                                    color = IndigoAccent,
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.SemiBold
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PremiumSearchHeader(
    query: String,
    onQueryChange: (String) -> Unit,
    selectedCategory: String,
    onCategorySelect: (String) -> Unit,
    isScrolled: Boolean
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .background(
                if (isScrolled) DarkBg.copy(alpha = 0.95f) else Color.Transparent
            )
            .padding(horizontal = 16.dp, vertical = if (isScrolled) 8.dp else 24.dp)
    ) {
        // Search Input with Glow
        Box {
            // Glow effect when typing
            if (query.isNotBlank()) {
                Box(
                    modifier = Modifier
                        .matchParentSize()
                        .blur(8.dp)
                        .background(
                            Brush.horizontalGradient(
                                colors = listOf(
                                    IndigoAccent.copy(alpha = 0.1f),
                                    FuchsiaAccent.copy(alpha = 0.1f)
                                )
                            ),
                            RoundedCornerShape(16.dp)
                        )
                )
            }
            
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(CardBg)
                    .border(1.dp, BorderWhite10, RoundedCornerShape(16.dp))
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Outlined.Search,
                    contentDescription = null,
                    tint = if (query.isNotBlank()) IndigoAccent else Color(0xFF71717A),
                    modifier = Modifier.size(18.dp)
                )
                
                Spacer(modifier = Modifier.width(12.dp))
                
                BasicTextField(
                    value = query,
                    onValueChange = onQueryChange,
                    modifier = Modifier.weight(1f),
                    textStyle = androidx.compose.ui.text.TextStyle(
                        color = Color.White,
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Medium
                    ),
                    cursorBrush = SolidColor(IndigoAccent),
                    decorationBox = { innerTextField ->
                        Box {
                            if (query.isEmpty()) {
                                Text(
                                    "Search...",
                                    color = Color(0xFF71717A),
                                    fontSize = 15.sp,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                            innerTextField()
                        }
                    }
                )
                
                if (query.isNotBlank()) {
                    IconButton(
                        onClick = { onQueryChange("") },
                        modifier = Modifier.size(32.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Close,
                            contentDescription = "Clear",
                            tint = Color(0xFFA1A1AA),
                            modifier = Modifier.size(16.dp)
                        )
                    }
                } else {
                    Icon(
                        imageVector = Icons.Outlined.Mic,
                        contentDescription = "Voice search",
                        tint = Color(0xFF71717A),
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }

        // Category Tabs
        if (query.isBlank()) {
            Spacer(modifier = Modifier.height(16.dp))
            
            LazyRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(CATEGORIES) { category ->
                    CategoryChip(
                        text = category,
                        isSelected = selectedCategory == category,
                        onClick = { onCategorySelect(category) }
                    )
                }
            }
        }
    }
}

@Composable
private fun CategoryChip(
    text: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val scale by animateFloatAsState(
        targetValue = if (isSelected) 1f else 0.95f,
        animationSpec = spring(dampingRatio = 0.6f),
        label = "scale"
    )
    
    Box(
        modifier = Modifier
            .scale(scale)
            .clip(RoundedCornerShape(20.dp))
            .background(
                if (isSelected) Color.White else CardBg
            )
            .border(
                1.dp,
                if (isSelected) Color.White else BorderWhite5,
                RoundedCornerShape(20.dp)
            )
            .clickable { onClick() }
            .padding(horizontal = 24.dp, vertical = 10.dp)
    ) {
        Text(
            text = text,
            color = if (isSelected) Color.Black else Color(0xFFA1A1AA),
            fontSize = 13.sp,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
private fun SuggestedUsersSection(
    users: List<User>,
    followingUsers: List<String>,
    onUserClick: (User) -> Unit,
    onFollowClick: (String) -> Unit
) {
    Column(
        modifier = Modifier.padding(top = 20.dp)
    ) {
        // Header with icon
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Icon(
                imageVector = Icons.Filled.PersonAdd,
                contentDescription = null,
                tint = IndigoAccent,
                modifier = Modifier.size(16.dp)
            )
            Text(
                text = "Suggested for you",
                color = Color.White,
                fontSize = 14.sp,
                fontWeight = FontWeight.SemiBold
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // User Cards with better spacing
        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(horizontal = 16.dp)
        ) {
            items(users) { user ->
                PremiumUserCard(
                    user = user,
                    isFollowing = followingUsers.contains(user.userId),
                    onClick = { onUserClick(user) },
                    onFollowClick = { onFollowClick(user.userId) }
                )
            }
        }
    }
}

@Composable
private fun PremiumUserCard(
    user: User,
    isFollowing: Boolean,
    onClick: () -> Unit,
    onFollowClick: () -> Unit
) {
    // Premium glassmorphism card with glow effect
    Box(
        modifier = Modifier
            .width(140.dp)
            .clip(RoundedCornerShape(24.dp))
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        Color(0xFF1E1E2E),
                        Color(0xFF151520)
                    )
                )
            )
            .border(
                width = 1.dp,
                brush = Brush.verticalGradient(
                    colors = listOf(
                        Color.White.copy(alpha = 0.15f),
                        Color.White.copy(alpha = 0.05f)
                    )
                ),
                shape = RoundedCornerShape(24.dp)
            )
            .clickable { onClick() }
            .padding(16.dp)
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.fillMaxWidth()
        ) {
            // Avatar with animated gradient ring
            Box(
                contentAlignment = Alignment.Center
            ) {
                // Outer glow
                Box(
                    modifier = Modifier
                        .size(76.dp)
                        .clip(CircleShape)
                        .background(
                            Brush.radialGradient(
                                colors = listOf(
                                    IndigoAccent.copy(alpha = 0.3f),
                                    Color.Transparent
                                )
                            )
                        )
                )
                
                // Gradient ring
                Box(
                    modifier = Modifier
                        .size(72.dp)
                        .clip(CircleShape)
                        .background(
                            Brush.sweepGradient(
                                colors = listOf(
                                    Color(0xFF6366F1),
                                    Color(0xFFEC4899),
                                    Color(0xFF8B5CF6),
                                    Color(0xFF6366F1)
                                )
                            )
                        )
                        .padding(3.dp)
                ) {
                    AsyncImage(
                        model = user.avatar,
                        contentDescription = null,
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(CircleShape)
                            .background(Color(0xFF0A0A0F)),
                        contentScale = ContentScale.Crop
                    )
                }
                
                // Online indicator
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .offset(x = (-4).dp, y = (-4).dp)
                        .size(16.dp)
                        .clip(CircleShape)
                        .background(Color(0xFF0A0A0F))
                        .padding(2.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(CircleShape)
                            .background(Color(0xFF22C55E))
                    )
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            // Username with subtle glow
            Text(
                text = user.username.replaceFirstChar { it.uppercase() },
                color = Color.White,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.fillMaxWidth(),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center
            )

            Spacer(modifier = Modifier.height(2.dp))

            // @handle
            Text(
                text = "@${user.username.lowercase().replace(" ", "")}",
                color = Color(0xFF71717A),
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.fillMaxWidth(),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center
            )

            Spacer(modifier = Modifier.height(14.dp))

            // Premium Follow Button
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(36.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(
                        if (isFollowing)
                            Brush.horizontalGradient(
                                colors = listOf(
                                    Color(0xFF1F1F2E),
                                    Color(0xFF1A1A28)
                                )
                            )
                        else
                            Brush.horizontalGradient(
                                colors = listOf(
                                    Color(0xFF6366F1),
                                    Color(0xFF8B5CF6)
                                )
                            )
                    )
                    .then(
                        if (isFollowing)
                            Modifier.border(
                                1.dp,
                                Color(0xFF6366F1).copy(alpha = 0.3f),
                                RoundedCornerShape(12.dp)
                            )
                        else
                            Modifier
                    )
                    .clickable { onFollowClick() },
                contentAlignment = Alignment.Center
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    if (isFollowing) {
                        Icon(
                            imageVector = Icons.Filled.Check,
                            contentDescription = null,
                            tint = Color(0xFF6366F1),
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                    }
                    Text(
                        text = if (isFollowing) "Following" else "Follow",
                        color = if (isFollowing) Color(0xFF6366F1) else Color.White,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }
    }
}


// Shimmer effect for loading states
@Composable
private fun ShimmerBox(
    modifier: Modifier = Modifier
) {
    val shimmerColors = listOf(
        Color(0xFF1A1A2E),
        Color(0xFF252540),
        Color(0xFF1A1A2E)
    )
    
    val transition = rememberInfiniteTransition(label = "shimmer")
    val translateAnim by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1000f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "shimmerTranslate"
    )
    
    val brush = Brush.linearGradient(
        colors = shimmerColors,
        start = androidx.compose.ui.geometry.Offset(translateAnim - 500f, 0f),
        end = androidx.compose.ui.geometry.Offset(translateAnim, 0f)
    )
    
    Box(
        modifier = modifier.background(brush)
    )
}

// Skeleton Post Card for loading state
@Composable
private fun SkeletonPostCard() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(3f / 4f)
            .clip(RoundedCornerShape(32.dp))
            .background(CardBg)
            .border(1.dp, BorderWhite10, RoundedCornerShape(32.dp))
    ) {
        // Shimmer overlay
        ShimmerBox(
            modifier = Modifier
                .fillMaxSize()
                .clip(RoundedCornerShape(32.dp))
        )
        
        // Top user info skeleton
        Row(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
                .clip(RoundedCornerShape(24.dp))
                .background(Color.Black.copy(alpha = 0.3f))
                .padding(start = 6.dp, end = 12.dp, top = 6.dp, bottom = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Avatar skeleton
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(CircleShape)
                    .background(Color(0xFF2A2A3E))
            )
            // Username skeleton
            Box(
                modifier = Modifier
                    .width(60.dp)
                    .height(10.dp)
                    .clip(RoundedCornerShape(5.dp))
                    .background(Color(0xFF2A2A3E))
            )
        }
    }
    
    Spacer(modifier = Modifier.height(16.dp))
    
    // Action buttons skeleton
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Action capsule skeleton
        Box(
            modifier = Modifier
                .weight(1f)
                .height(44.dp)
                .clip(RoundedCornerShape(28.dp))
                .background(CardBg)
                .border(1.dp, BorderWhite10, RoundedCornerShape(28.dp))
        )
        
        Spacer(modifier = Modifier.width(12.dp))
        
        // Bookmark skeleton
        Box(
            modifier = Modifier
                .size(44.dp)
                .clip(CircleShape)
                .background(CardBg)
                .border(1.dp, BorderWhite10, CircleShape)
        )
    }
    
    Spacer(modifier = Modifier.height(12.dp))
    
    // Caption skeleton
    Column(
        modifier = Modifier.padding(horizontal = 4.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        ShimmerBox(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .height(12.dp)
                .clip(RoundedCornerShape(6.dp))
        )
        ShimmerBox(
            modifier = Modifier
                .fillMaxWidth(0.6f)
                .height(12.dp)
                .clip(RoundedCornerShape(6.dp))
        )
        ShimmerBox(
            modifier = Modifier
                .width(60.dp)
                .height(10.dp)
                .clip(RoundedCornerShape(5.dp))
        )
    }
}

/**
 * Search User Card - Horizontal card for search results
 */
@Composable
private fun SearchUserCard(
    user: User,
    isFollowing: Boolean,
    onClick: () -> Unit,
    onFollowClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() }
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Avatar with gradient ring
        Box(
            modifier = Modifier
                .size(56.dp)
                .clip(CircleShape)
                .background(
                    Brush.sweepGradient(
                        colors = listOf(
                            Color(0xFF6366F1),
                            Color(0xFFEC4899),
                            Color(0xFF8B5CF6),
                            Color(0xFF6366F1)
                        )
                    )
                )
                .padding(2.5.dp)
        ) {
            AsyncImage(
                model = user.avatar,
                contentDescription = null,
                modifier = Modifier
                    .fillMaxSize()
                    .clip(CircleShape)
                    .background(Color(0xFF0A0A0F)),
                contentScale = ContentScale.Crop
            )
        }
        
        // User info
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            // Username
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Text(
                    text = user.username,
                    color = Color.White,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                if (user.isVerified) {
                    Icon(
                        imageVector = Icons.Filled.Verified,
                        contentDescription = "Verified",
                        tint = Color(0xFF3B82F6),
                        modifier = Modifier.size(14.dp)
                    )
                }
            }
            
            // Bio or handle
            Text(
                text = user.bio?.take(50) ?: "@${user.username.lowercase()}",
                color = Color(0xFF71717A),
                fontSize = 13.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            
            // Followers count
            Text(
                text = "${formatFollowerCount(user.followersCount)} followers",
                color = Color(0xFF52525B),
                fontSize = 12.sp
            )
        }
        
        // Follow button
        val followButtonModifier = if (isFollowing) {
            Modifier
                .clip(RoundedCornerShape(8.dp))
                .background(Color(0xFF1F1F2E))
                .border(1.dp, Color(0xFF6366F1).copy(alpha = 0.3f), RoundedCornerShape(8.dp))
                .clickable { onFollowClick() }
                .padding(horizontal = 16.dp, vertical = 8.dp)
        } else {
            Modifier
                .clip(RoundedCornerShape(8.dp))
                .background(
                    Brush.horizontalGradient(
                        colors = listOf(Color(0xFF6366F1), Color(0xFF8B5CF6))
                    )
                )
                .clickable { onFollowClick() }
                .padding(horizontal = 16.dp, vertical = 8.dp)
        }
        
        Box(modifier = followButtonModifier) {
            Text(
                text = if (isFollowing) "Following" else "Follow",
                color = if (isFollowing) Color(0xFF6366F1) else Color.White,
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

private fun formatFollowerCount(count: Int): String {
    return when {
        count >= 1_000_000 -> "${count / 1_000_000}M"
        count >= 1_000 -> "${count / 1_000}K"
        else -> count.toString()
    }
}
