package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.*
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.orignal.buddylynk.data.model.Activity
import com.orignal.buddylynk.ui.components.*
import com.orignal.buddylynk.ui.theme.*
import com.orignal.buddylynk.ui.viewmodel.ActivityViewModel

/**
 * ActivityScreen - Shows all user activities (likes, follows, comments, mentions)
 */
@Composable
fun ActivityScreen(
    onNavigateBack: () -> Unit,
    onNavigateToProfile: (String) -> Unit = {},
    onNavigateToPost: (String) -> Unit = {},
    viewModel: ActivityViewModel = viewModel()
) {
    val activities by viewModel.activities.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val selectedFilter by viewModel.selectedFilter.collectAsState()
    
    LaunchedEffect(Unit) {
        viewModel.loadActivities()
    }
    
    AnimatedGradientBackground(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            // Top Bar
            ActivityTopBar(onNavigateBack = onNavigateBack)
            
            // Filter Chips
            ActivityFilterChips(
                selectedFilter = selectedFilter,
                onFilterChange = { viewModel.setFilter(it) }
            )
            
            // Activity List
            if (isLoading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = GradientPurple)
                }
            } else if (activities.isEmpty()) {
                EmptyActivityState()
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    // Group by time
                    val today = activities.filter { isToday(it.createdAt) }
                    val thisWeek = activities.filter { isThisWeek(it.createdAt) && !isToday(it.createdAt) }
                    val older = activities.filter { !isThisWeek(it.createdAt) }
                    
                    if (today.isNotEmpty()) {
                        item {
                            SectionHeader("Today")
                        }
                        items(today) { activity ->
                            ActivityItem(
                                activity = activity,
                                onProfileClick = { onNavigateToProfile(activity.actorId) },
                                onContentClick = {
                                    activity.targetId?.let { onNavigateToPost(it) }
                                }
                            )
                        }
                    }
                    
                    if (thisWeek.isNotEmpty()) {
                        item {
                            SectionHeader("This Week")
                        }
                        items(thisWeek) { activity ->
                            ActivityItem(
                                activity = activity,
                                onProfileClick = { onNavigateToProfile(activity.actorId) },
                                onContentClick = {
                                    activity.targetId?.let { onNavigateToPost(it) }
                                }
                            )
                        }
                    }
                    
                    if (older.isNotEmpty()) {
                        item {
                            SectionHeader("Earlier")
                        }
                        items(older) { activity ->
                            ActivityItem(
                                activity = activity,
                                onProfileClick = { onNavigateToProfile(activity.actorId) },
                                onContentClick = {
                                    activity.targetId?.let { onNavigateToPost(it) }
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ActivityTopBar(onNavigateBack: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        IconButton(
            onClick = onNavigateBack,
            modifier = Modifier.align(Alignment.CenterStart)
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "Back",
                tint = TextPrimary
            )
        }
        
        Text(
            text = "Activity",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            color = TextPrimary,
            modifier = Modifier.align(Alignment.Center)
        )
    }
}

@Composable
private fun ActivityFilterChips(
    selectedFilter: String,
    onFilterChange: (String) -> Unit
) {
    val filters = listOf("All", "Likes", "Comments", "Follows", "Mentions")
    
    LazyRow(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        contentPadding = PaddingValues(horizontal = 16.dp)
    ) {
        items(filters) { filter ->
            val isSelected = filter == selectedFilter
            
            FilterChip(
                selected = isSelected,
                onClick = { onFilterChange(filter) },
                label = { Text(filter) },
                colors = FilterChipDefaults.filterChipColors(
                    containerColor = if (isSelected) GradientPurple else GlassWhite,
                    labelColor = if (isSelected) Color.White else TextSecondary,
                    selectedContainerColor = GradientPurple,
                    selectedLabelColor = Color.White
                ),
                border = if (isSelected) null else FilterChipDefaults.filterChipBorder(
                    borderColor = GlassBorder,
                    enabled = true,
                    selected = false
                )
            )
        }
    }
}

@Composable
private fun ActivityItem(
    activity: Activity,
    onProfileClick: () -> Unit,
    onContentClick: () -> Unit
) {
    val isUnread = !activity.isRead
    
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(if (isUnread) GlassWhite.copy(alpha = 0.5f) else Color.Transparent)
            .clickable { onContentClick() }
            .padding(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Avatar
        Box(
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .clickable { onProfileClick() }
        ) {
            if (activity.actorAvatar != null) {
                AsyncImage(
                    model = activity.actorAvatar,
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            brush = Brush.linearGradient(PremiumGradient),
                            shape = CircleShape
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = activity.actorUsername.firstOrNull()?.uppercase() ?: "",
                        color = Color.White,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
            
            // Activity type icon
            Box(
                modifier = Modifier
                    .size(20.dp)
                    .align(Alignment.BottomEnd)
                    .clip(CircleShape)
                    .background(getActivityColor(activity.type))
                    .padding(3.dp)
            ) {
                Icon(
                    imageVector = getActivityIcon(activity.type),
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(14.dp)
                )
            }
        }
        
        Spacer(modifier = Modifier.width(12.dp))
        
        // Activity text
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = buildAnnotatedString {
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                        append(activity.actorUsername)
                    }
                    append(" ")
                    append(getActivityText(activity.type))
                },
                style = MaterialTheme.typography.bodyMedium,
                color = TextPrimary
            )
            
            Spacer(modifier = Modifier.height(4.dp))
            
            Text(
                text = getTimeAgo(activity.createdAt),
                style = MaterialTheme.typography.bodySmall,
                color = TextTertiary
            )
        }
        
        // Post preview (if applicable)
        activity.targetPreview?.let { preview ->
            if (preview.isNotBlank() && preview.startsWith("http")) {
                AsyncImage(
                    model = preview,
                    contentDescription = null,
                    modifier = Modifier
                        .size(48.dp)
                        .clip(RoundedCornerShape(8.dp)),
                    contentScale = ContentScale.Crop
                )
            }
        }
    }
}

@Composable
private fun SectionHeader(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.titleSmall,
        fontWeight = FontWeight.Bold,
        color = TextSecondary,
        modifier = Modifier.padding(vertical = 8.dp)
    )
}

@Composable
private fun EmptyActivityState() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = Icons.Outlined.Notifications,
                contentDescription = null,
                tint = TextTertiary,
                modifier = Modifier.size(64.dp)
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "No Activity Yet",
                style = MaterialTheme.typography.titleMedium,
                color = TextSecondary
            )
            Text(
                text = "When people interact with you, you'll see it here",
                style = MaterialTheme.typography.bodySmall,
                color = TextTertiary
            )
        }
    }
}

private fun getActivityIcon(type: String) = when (type) {
    "like" -> Icons.Filled.Favorite
    "comment" -> Icons.Filled.ChatBubble
    "follow" -> Icons.Filled.PersonAdd
    "mention" -> Icons.Filled.AlternateEmail
    else -> Icons.Filled.Notifications
}

private fun getActivityColor(type: String) = when (type) {
    "like" -> Color(0xFFE91E63)
    "comment" -> Color(0xFF2196F3)
    "follow" -> Color(0xFF4CAF50)
    "mention" -> Color(0xFFFF9800)
    else -> GradientPurple
}

private fun getActivityText(type: String) = when (type) {
    "like" -> "liked your post"
    "comment" -> "commented on your post"
    "follow" -> "started following you"
    "mention" -> "mentioned you in a post"
    else -> "interacted with you"
}

private fun getTimeAgo(timestamp: String): String {
    val time = timestamp.toLongOrNull() ?: return ""
    val now = System.currentTimeMillis()
    val diff = now - time
    
    return when {
        diff < 60_000 -> "Just now"
        diff < 3_600_000 -> "${diff / 60_000}m"
        diff < 86_400_000 -> "${diff / 3_600_000}h"
        diff < 604_800_000 -> "${diff / 86_400_000}d"
        else -> "${diff / 604_800_000}w"
    }
}

private fun isToday(timestamp: String): Boolean {
    val time = timestamp.toLongOrNull() ?: return false
    val now = System.currentTimeMillis()
    val diff = now - time
    return diff < 86_400_000 // 24 hours
}

private fun isThisWeek(timestamp: String): Boolean {
    val time = timestamp.toLongOrNull() ?: return false
    val now = System.currentTimeMillis()
    val diff = now - time
    return diff < 604_800_000 // 7 days
}
