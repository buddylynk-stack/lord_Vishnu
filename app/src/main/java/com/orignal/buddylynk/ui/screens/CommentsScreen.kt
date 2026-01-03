package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Reply
import androidx.compose.material.icons.automirrored.filled.Send
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
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.model.Comment
import com.orignal.buddylynk.ui.components.*
import com.orignal.buddylynk.ui.theme.*
import com.orignal.buddylynk.ui.viewmodel.CommentsViewModel

/**
 * CommentsScreen - Display and post comments with pin/edit/delete features
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CommentsScreen(
    postId: String,
    onNavigateBack: () -> Unit,
    onNavigateToProfile: (String) -> Unit = {},
    postOwnerId: String = "", // Pass the post owner's ID
    viewModel: CommentsViewModel = viewModel()
) {
    val comments by viewModel.comments.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val isSending by viewModel.isSending.collectAsState()
    val replyingTo by viewModel.replyingTo.collectAsState()
    val currentUser by AuthManager.currentUser.collectAsState()
    
    var commentText by remember { mutableStateOf("") }
    var editingComment by remember { mutableStateOf<Comment?>(null) }
    val keyboardController = LocalSoftwareKeyboardController.current
    
    val isPostOwner = currentUser?.userId == postOwnerId
    
    LaunchedEffect(postId) {
        viewModel.loadComments(postId)
    }
    
    // Sort comments: pinned first, then by time
    val sortedComments = comments
        .filter { it.parentCommentId == null }
        .sortedByDescending { it.isPinned }
    
    AnimatedGradientBackground(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            // Top Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onNavigateBack) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = TextPrimary
                    )
                }
                
                // Professional comment icon
                Icon(
                    imageVector = Icons.Filled.ModeComment,
                    contentDescription = null,
                    tint = GradientPurple,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                
                Text(
                    text = "Comments",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary,
                    modifier = Modifier.weight(1f)
                )
                
                Box(
                    modifier = Modifier
                        .background(GradientPurple.copy(alpha = 0.2f), RoundedCornerShape(12.dp))
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = "${comments.size}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = GradientPurple,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
            
            // Comments list
            if (isLoading) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = GradientPurple)
                }
            } else if (comments.isEmpty()) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(
                            imageVector = Icons.Filled.ModeComment,
                            contentDescription = null,
                            tint = TextTertiary,
                            modifier = Modifier.size(64.dp)
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "No comments yet",
                            style = MaterialTheme.typography.titleMedium,
                            color = TextSecondary
                        )
                        Text(
                            text = "Be the first to comment!",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextTertiary
                        )
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    items(sortedComments) { comment ->
                        CommentItem(
                            comment = comment,
                            isPostOwner = isPostOwner,
                            isCommentOwner = comment.userId == currentUser?.userId,
                            onProfileClick = { onNavigateToProfile(comment.userId) },
                            onLikeClick = { viewModel.toggleLike(comment.commentId) },
                            onReplyClick = { viewModel.setReplyingTo(comment) },
                            onPinClick = { viewModel.togglePin(comment.commentId) },
                            onEditClick = { editingComment = comment; commentText = comment.content },
                            onDeleteClick = { viewModel.deleteComment(comment.commentId) },
                            replies = comments.filter { it.parentCommentId == comment.commentId },
                            onReplyProfileClick = onNavigateToProfile,
                            currentUserId = currentUser?.userId ?: ""
                        )
                    }
                }
            }
            
            // Reply indicator
            AnimatedVisibility(
                visible = replyingTo != null || editingComment != null,
                enter = slideInVertically { it } + fadeIn(),
                exit = slideOutVertically { it } + fadeOut()
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(GlassWhite)
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = if (editingComment != null) Icons.Filled.Edit else Icons.AutoMirrored.Filled.Reply,
                        contentDescription = null,
                        tint = GradientPurple,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (editingComment != null) "Editing comment" else "Replying to ${replyingTo?.username}",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                        modifier = Modifier.weight(1f)
                    )
                    IconButton(
                        onClick = { 
                            viewModel.cancelReply()
                            editingComment = null
                            commentText = ""
                        },
                        modifier = Modifier.size(24.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Close,
                            contentDescription = "Cancel",
                            tint = TextSecondary,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }
            
            // Comment input
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(GlassWhite)
                    .padding(horizontal = 16.dp, vertical = 12.dp)
                    .navigationBarsPadding(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = commentText,
                    onValueChange = { commentText = it },
                    modifier = Modifier.weight(1f),
                    placeholder = {
                        Text(
                            text = when {
                                editingComment != null -> "Edit your comment..."
                                replyingTo != null -> "Write a reply..."
                                else -> "Add a comment..."
                            },
                            color = TextTertiary
                        )
                    },
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = GradientPurple,
                        unfocusedBorderColor = GlassBorder,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary
                    ),
                    shape = RoundedCornerShape(24.dp),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                    keyboardActions = KeyboardActions(
                        onSend = {
                            if (commentText.isNotBlank()) {
                                if (editingComment != null) {
                                    viewModel.editComment(editingComment!!.commentId, commentText)
                                    editingComment = null
                                } else {
                                    viewModel.postComment(postId, commentText)
                                }
                                commentText = ""
                                keyboardController?.hide()
                            }
                        }
                    ),
                    singleLine = true
                )
                
                Spacer(modifier = Modifier.width(8.dp))
                
                // Send button
                IconButton(
                    onClick = {
                        if (commentText.isNotBlank()) {
                            if (editingComment != null) {
                                viewModel.editComment(editingComment!!.commentId, commentText)
                                editingComment = null
                            } else {
                                viewModel.postComment(postId, commentText)
                            }
                            commentText = ""
                            keyboardController?.hide()
                        }
                    },
                    enabled = commentText.isNotBlank() && !isSending,
                    modifier = Modifier
                        .size(48.dp)
                        .background(
                            brush = if (commentText.isNotBlank()) 
                                Brush.linearGradient(PremiumGradient) 
                            else Brush.linearGradient(listOf(GlassWhite, GlassWhite)),
                            shape = CircleShape
                        )
                ) {
                    if (isSending) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            color = Color.White,
                            strokeWidth = 2.dp
                        )
                    } else {
                        Icon(
                            imageVector = if (editingComment != null) Icons.Filled.Check else Icons.AutoMirrored.Filled.Send,
                            contentDescription = if (editingComment != null) "Save" else "Send",
                            tint = if (commentText.isNotBlank()) Color.White else TextTertiary
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun CommentItem(
    comment: Comment,
    isPostOwner: Boolean,
    isCommentOwner: Boolean,
    onProfileClick: () -> Unit,
    onLikeClick: () -> Unit,
    onReplyClick: () -> Unit,
    onPinClick: () -> Unit,
    onEditClick: () -> Unit,
    onDeleteClick: () -> Unit,
    replies: List<Comment>,
    onReplyProfileClick: (String) -> Unit,
    currentUserId: String
) {
    var showReplies by remember { mutableStateOf(false) }
    var showOptions by remember { mutableStateOf(false) }
    
    Column {
        // Pinned indicator
        if (comment.isPinned) {
            Row(
                modifier = Modifier.padding(bottom = 6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Filled.PushPin,
                    contentDescription = null,
                    tint = GradientOrange,
                    modifier = Modifier.size(14.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "Pinned by creator",
                    color = GradientOrange,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }
        
        Row(modifier = Modifier.fillMaxWidth()) {
            // Avatar
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .clickable { onProfileClick() }
            ) {
                if (comment.userAvatar != null) {
                    AsyncImage(
                        model = comment.userAvatar,
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
                            text = comment.username.firstOrNull()?.uppercase() ?: "",
                            color = Color.White,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
            
            Spacer(modifier = Modifier.width(12.dp))
            
            // Comment content
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = comment.username,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary,
                        fontSize = 14.sp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = getTimeAgo(comment.createdAt),
                        color = TextTertiary,
                        fontSize = 12.sp
                    )
                    // Edited watermark
                    if (comment.isEdited) {
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "(edited)",
                            color = TextTertiary.copy(alpha = 0.7f),
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Light
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(4.dp))
                
                Text(
                    text = comment.content,
                    color = TextPrimary,
                    fontSize = 14.sp
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Actions
                Row(verticalAlignment = Alignment.CenterVertically) {
                    // Like
                    Row(
                        modifier = Modifier.clickable { onLikeClick() },
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = if (comment.isLiked) Icons.Filled.Favorite else Icons.Outlined.FavoriteBorder,
                            contentDescription = "Like",
                            tint = if (comment.isLiked) LikeRed else TextTertiary,
                            modifier = Modifier.size(16.dp)
                        )
                        if (comment.likesCount > 0) {
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = comment.likesCount.toString(),
                                color = TextTertiary,
                                fontSize = 12.sp
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.width(16.dp))
                    
                    // Reply
                    Text(
                        text = "Reply",
                        color = TextSecondary,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium,
                        modifier = Modifier.clickable { onReplyClick() }
                    )
                    
                    // Show replies
                    if (replies.isNotEmpty()) {
                        Spacer(modifier = Modifier.width(16.dp))
                        Text(
                            text = if (showReplies) "Hide replies" else "${replies.size} replies",
                            color = GradientPurple,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium,
                            modifier = Modifier.clickable { showReplies = !showReplies }
                        )
                    }
                }
            }
            
            // Options menu (3-dot) - show for post owner or comment owner
            if (isPostOwner || isCommentOwner) {
                Box {
                    IconButton(
                        onClick = { showOptions = true },
                        modifier = Modifier.size(32.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.MoreVert,
                            contentDescription = "Options",
                            tint = TextTertiary,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                    
                    DropdownMenu(
                        expanded = showOptions,
                        onDismissRequest = { showOptions = false },
                        modifier = Modifier.background(DarkSurface)
                    ) {
                        // Post owner options
                        if (isPostOwner) {
                            DropdownMenuItem(
                                text = {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(
                                            imageVector = if (comment.isPinned) Icons.Outlined.PushPin else Icons.Filled.PushPin,
                                            contentDescription = null,
                                            tint = GradientOrange,
                                            modifier = Modifier.size(18.dp)
                                        )
                                        Spacer(Modifier.width(10.dp))
                                        Text(
                                            text = if (comment.isPinned) "Unpin" else "Pin Comment",
                                            color = TextPrimary
                                        )
                                    }
                                },
                                onClick = {
                                    showOptions = false
                                    onPinClick()
                                }
                            )
                        }
                        
                        // Comment owner options
                        if (isCommentOwner) {
                            DropdownMenuItem(
                                text = {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(
                                            Icons.Outlined.Edit,
                                            null,
                                            tint = GradientCyan,
                                            modifier = Modifier.size(18.dp)
                                        )
                                        Spacer(Modifier.width(10.dp))
                                        Text("Edit", color = TextPrimary)
                                    }
                                },
                                onClick = {
                                    showOptions = false
                                    onEditClick()
                                }
                            )
                        }
                        
                        // Delete option for both post owner and comment owner
                        DropdownMenuItem(
                            text = {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        Icons.Outlined.Delete,
                                        null,
                                        tint = LikeRed,
                                        modifier = Modifier.size(18.dp)
                                    )
                                    Spacer(Modifier.width(10.dp))
                                    Text("Delete", color = LikeRed)
                                }
                            },
                            onClick = {
                                showOptions = false
                                onDeleteClick()
                            }
                        )
                    }
                }
            }
        }
        
        // Replies
        AnimatedVisibility(visible = showReplies) {
            Column(
                modifier = Modifier.padding(start = 52.dp, top = 12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                replies.forEach { reply ->
                    ReplyItem(
                        comment = reply,
                        onProfileClick = { onReplyProfileClick(reply.userId) },
                        isOwner = reply.userId == currentUserId,
                        onEditClick = {},
                        onDeleteClick = {}
                    )
                }
            }
        }
    }
}

@Composable
private fun ReplyItem(
    comment: Comment,
    onProfileClick: () -> Unit,
    isOwner: Boolean = false,
    onEditClick: () -> Unit,
    onDeleteClick: () -> Unit
) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(CircleShape)
                .clickable { onProfileClick() }
        ) {
            if (comment.userAvatar != null) {
                AsyncImage(
                    model = comment.userAvatar,
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
                        text = comment.username.firstOrNull()?.uppercase() ?: "",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
        
        Spacer(modifier = Modifier.width(8.dp))
        
        Column(modifier = Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = comment.username,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary,
                    fontSize = 13.sp
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = getTimeAgo(comment.createdAt),
                    color = TextTertiary,
                    fontSize = 11.sp
                )
                // Edited watermark for replies
                if (comment.isEdited) {
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "(edited)",
                        color = TextTertiary.copy(alpha = 0.7f),
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Light
                    )
                }
            }
            Text(
                text = comment.content,
                color = TextPrimary,
                fontSize = 13.sp
            )
        }
    }
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
