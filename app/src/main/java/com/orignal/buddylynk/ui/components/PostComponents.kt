package com.orignal.buddylynk.ui.components

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.orignal.buddylynk.ui.theme.*

// =============================================================================
// POST INTERACTIONS - Like, Comment, Share, Bookmark
// =============================================================================

data class PostState(
    val postId: String = "",
    val isLiked: Boolean = false,
    val isBookmarked: Boolean = false,
    val likesCount: Int = 0,
    val commentsCount: Int = 0,
    val sharesCount: Int = 0,
    val comments: List<Comment> = emptyList()
)

data class Comment(
    val id: String,
    val username: String,
    val content: String,
    val timeAgo: String,
    val likes: Int = 0,
    val isLiked: Boolean = false
)

/**
 * Post Action Bar with working like, comment, share, bookmark
 */
@Composable
fun PostActionBar(
    postState: PostState,
    onLikeClick: () -> Unit,
    onCommentClick: () -> Unit,
    onShareClick: () -> Unit,
    onBookmarkClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Like button with pop animation
            PopIconButton(
                onClick = onLikeClick,
                isActive = postState.isLiked
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Icon(
                        imageVector = if (postState.isLiked) Icons.Filled.Favorite else Icons.Outlined.FavoriteBorder,
                        contentDescription = "Like",
                        tint = if (postState.isLiked) LikeRed else TextSecondary,
                        modifier = Modifier.size(24.dp)
                    )
                    Text(
                        text = postState.likesCount.toString(),
                        style = MaterialTheme.typography.bodySmall,
                        color = if (postState.isLiked) LikeRed else TextSecondary
                    )
                }
            }
            
            // Comment button
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.clickable { onCommentClick() }
            ) {
                Icon(
                    imageVector = Icons.Outlined.ChatBubbleOutline,
                    contentDescription = "Comment",
                    tint = TextSecondary,
                    modifier = Modifier.size(24.dp)
                )
                Text(
                    text = postState.commentsCount.toString(),
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
            }
            
            // Share button
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.clickable { onShareClick() }
            ) {
                Icon(
                    imageVector = Icons.Outlined.Share,
                    contentDescription = "Share",
                    tint = TextSecondary,
                    modifier = Modifier.size(22.dp)
                )
                if (postState.sharesCount > 0) {
                    Text(
                        text = postState.sharesCount.toString(),
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                }
            }
        }
        
        // Bookmark button
        PopIconButton(
            onClick = onBookmarkClick,
            isActive = postState.isBookmarked
        ) {
            Icon(
                imageVector = if (postState.isBookmarked) Icons.Filled.Bookmark else Icons.Outlined.BookmarkBorder,
                contentDescription = "Bookmark",
                tint = if (postState.isBookmarked) GradientCoral else TextSecondary,
                modifier = Modifier.size(24.dp)
            )
        }
    }
}

/**
 * Comments Bottom Sheet
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CommentsBottomSheet(
    comments: List<Comment>,
    onDismiss: () -> Unit,
    onSendComment: (String) -> Unit,
    onLikeComment: (String) -> Unit
) {
    var commentText by remember { mutableStateOf("") }
    
    // Get current user for avatar
    val currentUser = com.orignal.buddylynk.data.auth.AuthManager.currentUser.collectAsState().value
    val userInitial = currentUser?.username?.firstOrNull()?.uppercase() ?: "U"
    
    // Focus requester for auto-focus on input
    val focusRequester = remember { androidx.compose.ui.focus.FocusRequester() }
    val focusManager = androidx.compose.ui.platform.LocalFocusManager.current
    val keyboardController = androidx.compose.ui.platform.LocalSoftwareKeyboardController.current
    
    // Auto-focus and show keyboard when sheet opens
    LaunchedEffect(Unit) {
        kotlinx.coroutines.delay(300) // Wait for sheet animation
        focusRequester.requestFocus()
        keyboardController?.show()
    }
    
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = DarkSurface,
        dragHandle = {
            Box(
                modifier = Modifier
                    .padding(vertical = 12.dp)
                    .width(40.dp)
                    .height(4.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(TextTertiary)
            )
        }
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 300.dp)
        ) {
            // Header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Comments",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                Text(
                    text = "${comments.size}",
                    style = MaterialTheme.typography.titleMedium,
                    color = TextSecondary
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Comments list OR empty state - tap to dismiss keyboard
            if (comments.isEmpty()) {
                // Empty state - tap to dismiss keyboard and just browse
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(32.dp)
                        .clickable(
                            indication = null,
                            interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() }
                        ) {
                            focusManager.clearFocus()
                            keyboardController?.hide()
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.ChatBubbleOutline,
                            contentDescription = null,
                            tint = TextTertiary,
                            modifier = Modifier.size(48.dp)
                        )
                        Text(
                            text = "No comments yet",
                            style = MaterialTheme.typography.titleMedium,
                            color = TextSecondary
                        )
                        Text(
                            text = "Be the first to comment!",
                            style = MaterialTheme.typography.bodyMedium,
                            color = TextTertiary
                        )
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                        .clickable(
                            indication = null,
                            interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() }
                        ) {
                            focusManager.clearFocus()
                            keyboardController?.hide()
                        },
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(comments) { comment ->
                        CommentItem(
                            comment = comment,
                            onLikeClick = { onLikeComment(comment.id) }
                        )
                    }
                    
                    item {
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                }
            }
            
            // Comment input - Instagram style bottom bar
            HorizontalDivider(color = GlassWhite.copy(alpha = 0.1f))
            
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(DarkSurface)
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Small Avatar
                Box(
                    modifier = Modifier
                        .size(32.dp)
                        .clip(CircleShape)
                        .background(
                            if (currentUser?.avatar.isNullOrBlank()) 
                                Brush.linearGradient(PremiumGradient)
                            else 
                                Brush.linearGradient(listOf(Color.Gray, Color.Gray))
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    if (!currentUser?.avatar.isNullOrBlank()) {
                        coil.compose.AsyncImage(
                            model = currentUser?.avatar,
                            contentDescription = "Avatar",
                            modifier = Modifier
                                .fillMaxSize()
                                .clip(CircleShape),
                            contentScale = androidx.compose.ui.layout.ContentScale.Crop
                        )
                    } else {
                        Text(
                            text = userInitial,
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }
                }
                
                // Simple text input - Instagram style
                BasicTextField(
                    value = commentText,
                    onValueChange = { commentText = it },
                    modifier = Modifier
                        .weight(1f)
                        .focusRequester(focusRequester),
                    textStyle = MaterialTheme.typography.bodyMedium.copy(color = TextPrimary),
                    singleLine = true,
                    cursorBrush = Brush.linearGradient(PremiumGradient),
                    decorationBox = { innerTextField ->
                        Box {
                            if (commentText.isEmpty()) {
                                Text(
                                    text = "Add a comment for ${currentUser?.username ?: "this user"}...",
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = TextTertiary
                                )
                            }
                            innerTextField()
                        }
                    }
                )
                
                // Post button - Instagram style text
                AnimatedVisibility(
                    visible = commentText.isNotBlank(),
                    enter = fadeIn(),
                    exit = fadeOut()
                ) {
                    Text(
                        text = "Post",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold,
                        color = GradientCyan,
                        modifier = Modifier.clickable {
                            if (commentText.isNotBlank()) {
                                onSendComment(commentText.trim())
                                commentText = ""
                            }
                        }
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

@Composable
private fun CommentItem(
    comment: Comment,
    onLikeClick: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Avatar
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(CircleShape)
                .background(
                    Brush.linearGradient(VibrantGradient.shuffled().take(2))
                ),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = comment.username.first().toString(),
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
        }
        
        Column(modifier = Modifier.weight(1f)) {
            // Username and time
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = comment.username,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary
                )
                Text(
                    text = comment.timeAgo,
                    style = MaterialTheme.typography.labelSmall,
                    color = TextTertiary
                )
            }
            
            // Content
            Text(
                text = comment.content,
                style = MaterialTheme.typography.bodyMedium,
                color = TextSecondary
            )
            
            // Like button
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                modifier = Modifier
                    .padding(top = 4.dp)
                    .clickable { onLikeClick() }
            ) {
                Icon(
                    imageVector = if (comment.isLiked) Icons.Filled.Favorite else Icons.Outlined.FavoriteBorder,
                    contentDescription = "Like",
                    tint = if (comment.isLiked) LikeRed else TextTertiary,
                    modifier = Modifier.size(16.dp)
                )
                if (comment.likes > 0) {
                    Text(
                        text = comment.likes.toString(),
                        style = MaterialTheme.typography.labelSmall,
                        color = TextTertiary
                    )
                }
            }
        }
    }
}

/**
 * Share Bottom Sheet
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ShareBottomSheet(
    onDismiss: () -> Unit,
    onShareClick: (String) -> Unit
) {
    val shareOptions = listOf(
        "Copy Link" to Icons.Outlined.Link,
        "Share to Story" to Icons.Outlined.AddCircleOutline,
        "Send to Friends" to Icons.AutoMirrored.Filled.Send,
        "Share via..." to Icons.Outlined.Share
    )
    
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = DarkSurface
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "Share Post",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = TextPrimary
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            shareOptions.forEach { (label, icon) ->
                GlassCard(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onShareClick(label) },
                    cornerRadius = 16.dp,
                    glassOpacity = 0.06f
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.spacedBy(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = icon,
                            contentDescription = null,
                            tint = GradientCoral,
                            modifier = Modifier.size(24.dp)
                        )
                        Text(
                            text = label,
                            style = MaterialTheme.typography.bodyLarge,
                            color = TextPrimary
                        )
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
            }
            
            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}
