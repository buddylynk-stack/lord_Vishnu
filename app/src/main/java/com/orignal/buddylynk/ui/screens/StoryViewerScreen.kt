package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.*
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.orignal.buddylynk.data.model.Story
import com.orignal.buddylynk.data.model.User
import com.orignal.buddylynk.data.stories.StoriesService
import com.orignal.buddylynk.data.stories.UserStories
import com.orignal.buddylynk.ui.theme.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * StoryViewerScreen - Full-screen story viewer with Instagram-like UX
 */
@Composable
fun StoryViewerScreen(
    userStories: UserStories,
    allUserStories: List<UserStories>,
    startIndex: Int = 0,
    onClose: () -> Unit,
    onNavigateToProfile: (String) -> Unit = {}
) {
    // Current user index in all stories
    var currentUserIndex by remember { mutableIntStateOf(
        allUserStories.indexOf(userStories).coerceAtLeast(0)
    ) }
    
    // Current story index for current user
    var currentStoryIndex by remember { mutableIntStateOf(startIndex) }
    
    // Get current user stories
    val currentUser = allUserStories.getOrNull(currentUserIndex)
    val stories = currentUser?.stories ?: emptyList()
    val currentStory = stories.getOrNull(currentStoryIndex)
    
    // Progress animation
    val storyDuration = 5000L // 5 seconds per story
    var isPaused by remember { mutableStateOf(false) }
    
    // Auto-advance progress
    val progress = remember { Animatable(0f) }
    val scope = rememberCoroutineScope()
    
    // Mark story as viewed and handle auto-advance
    LaunchedEffect(currentUserIndex, currentStoryIndex) {
        currentStory?.let {
            StoriesService.markViewed(it.storyId)
        }
        
        progress.snapTo(0f)
        if (!isPaused) {
            progress.animateTo(
                targetValue = 1f,
                animationSpec = tween(storyDuration.toInt(), easing = LinearEasing)
            )
            // Auto advance
            goToNext(
                stories = stories,
                allStories = allUserStories,
                currentStoryIndex = currentStoryIndex,
                currentUserIndex = currentUserIndex,
                onStoryChange = { currentStoryIndex = it },
                onUserChange = { currentUserIndex = it },
                onClose = onClose
            )
        }
    }
    
    // Handle pause
    LaunchedEffect(isPaused) {
        if (isPaused) {
            progress.stop()
        } else {
            progress.animateTo(
                targetValue = 1f,
                animationSpec = tween(
                    durationMillis = ((1f - progress.value) * storyDuration).toInt(),
                    easing = LinearEasing
                )
            )
            goToNext(
                stories = stories,
                allStories = allUserStories,
                currentStoryIndex = currentStoryIndex,
                currentUserIndex = currentUserIndex,
                onStoryChange = { currentStoryIndex = it },
                onUserChange = { currentUserIndex = it },
                onClose = onClose
            )
        }
    }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // Story content
        currentStory?.let { story ->
            AsyncImage(
                model = story.mediaUrl,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        }
        
        // Tap areas for navigation
        Row(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            // Left tap - previous
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .clickable(
                        indication = null,
                        interactionSource = remember { MutableInteractionSource() }
                    ) {
                        if (currentStoryIndex > 0) {
                            currentStoryIndex--
                        } else if (currentUserIndex > 0) {
                            currentUserIndex--
                            currentStoryIndex = allUserStories[currentUserIndex].stories.lastIndex
                        }
                    }
                    .pointerInput(Unit) {
                        detectTapGestures(
                            onPress = {
                                isPaused = true
                                tryAwaitRelease()
                                isPaused = false
                            }
                        )
                    }
            )
            
            // Right tap - next
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .clickable(
                        indication = null,
                        interactionSource = remember { MutableInteractionSource() }
                    ) {
                        goToNext(
                            stories = stories,
                            allStories = allUserStories,
                            currentStoryIndex = currentStoryIndex,
                            currentUserIndex = currentUserIndex,
                            onStoryChange = { currentStoryIndex = it },
                            onUserChange = { currentUserIndex = it },
                            onClose = onClose
                        )
                    }
            )
        }
        
        // Top overlay with progress bars and user info
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(top = 8.dp, start = 12.dp, end = 12.dp)
        ) {
            // Progress bars
            Row(
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                stories.forEachIndexed { index, _ ->
                    val barProgress = when {
                        index < currentStoryIndex -> 1f
                        index == currentStoryIndex -> progress.value
                        else -> 0f
                    }
                    
                    LinearProgressIndicator(
                        progress = { barProgress },
                        modifier = Modifier
                            .weight(1f)
                            .height(2.dp)
                            .clip(RoundedCornerShape(1.dp)),
                        color = Color.White,
                        trackColor = Color.White.copy(alpha = 0.3f)
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // User info row
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Avatar
                Box(
                    modifier = Modifier
                        .size(36.dp)
                        .clip(CircleShape)
                        .background(GlassBorder)
                        .clickable { 
                            currentUser?.user?.userId?.let { onNavigateToProfile(it) }
                        }
                ) {
                    currentUser?.user?.avatar?.let { avatar ->
                        AsyncImage(
                            model = avatar,
                            contentDescription = null,
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Crop
                        )
                    } ?: Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                brush = Brush.linearGradient(PremiumGradient),
                                shape = CircleShape
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = currentUser?.user?.username?.firstOrNull()?.uppercase() ?: "",
                            color = Color.White,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
                
                Spacer(modifier = Modifier.width(8.dp))
                
                // Username and time
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = currentUser?.user?.username ?: "",
                        color = Color.White,
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 14.sp
                    )
                    Text(
                        text = getTimeAgo(currentStory?.createdAt ?: 0L),
                        color = Color.White.copy(alpha = 0.7f),
                        fontSize = 12.sp
                    )
                }
                
                // Close button
                IconButton(onClick = onClose) {
                    Icon(
                        imageVector = Icons.Filled.Close,
                        contentDescription = "Close",
                        tint = Color.White
                    )
                }
            }
        }
        
        // Caption at bottom
        currentStory?.caption?.let { caption ->
            if (caption.isNotBlank()) {
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .background(
                            brush = Brush.verticalGradient(
                                listOf(Color.Transparent, Color.Black.copy(alpha = 0.7f))
                            )
                        )
                        .padding(16.dp)
                        .navigationBarsPadding()
                ) {
                    Text(
                        text = caption,
                        color = Color.White,
                        fontSize = 14.sp
                    )
                }
            }
        }
        
        // Reply button at bottom
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(16.dp)
                .navigationBarsPadding(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .height(44.dp)
                    .clip(RoundedCornerShape(22.dp))
                    .background(Color.White.copy(alpha = 0.1f))
                    .border(1.dp, Color.White.copy(alpha = 0.3f), RoundedCornerShape(22.dp))
                    .clickable { /* Open reply */ },
                contentAlignment = Alignment.CenterStart
            ) {
                Text(
                    text = "Send a message...",
                    color = Color.White.copy(alpha = 0.7f),
                    modifier = Modifier.padding(horizontal = 16.dp)
                )
            }
            
            Spacer(modifier = Modifier.width(8.dp))
            
            // Heart reaction
            IconButton(
                onClick = { /* React */ },
                modifier = Modifier.size(44.dp)
            ) {
                Icon(
                    imageVector = Icons.Filled.FavoriteBorder,
                    contentDescription = "Like",
                    tint = Color.White
                )
            }
            
            // Share
            IconButton(
                onClick = { /* Share */ },
                modifier = Modifier.size(44.dp)
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.Send,
                    contentDescription = "Share",
                    tint = Color.White
                )
            }
        }
    }
}

private fun goToNext(
    stories: List<Story>,
    allStories: List<UserStories>,
    currentStoryIndex: Int,
    currentUserIndex: Int,
    onStoryChange: (Int) -> Unit,
    onUserChange: (Int) -> Unit,
    onClose: () -> Unit
) {
    if (currentStoryIndex < stories.lastIndex) {
        // Next story
        onStoryChange(currentStoryIndex + 1)
    } else if (currentUserIndex < allStories.lastIndex) {
        // Next user
        onUserChange(currentUserIndex + 1)
        onStoryChange(0)
    } else {
        // End of all stories
        onClose()
    }
}

private fun getTimeAgo(timestamp: Long): String {
    val now = System.currentTimeMillis()
    val diff = now - timestamp
    
    return when {
        diff < 60_000 -> "Just now"
        diff < 3_600_000 -> "${diff / 60_000}m ago"
        diff < 86_400_000 -> "${diff / 3_600_000}h ago"
        else -> "${diff / 86_400_000}d ago"
    }
}
