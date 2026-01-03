package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
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
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.orignal.buddylynk.data.model.Message
import com.orignal.buddylynk.ui.viewmodel.ChatViewModel
import com.orignal.buddylynk.ui.viewmodel.ChatListViewModel
import kotlinx.coroutines.delay

// Premium Colors
private val DarkBg = Color(0xFF09090B)
private val IndigoAccent = Color(0xFF6366F1)
private val VioletAccent = Color(0xFFA78BFA)
private val PurpleAccent = Color(0xFF8B5CF6)
private val ZincDark = Color(0xFF18181B)
private val Zinc800 = Color(0xFF27272A)
private val Zinc600 = Color(0xFF52525B)
private val Zinc500 = Color(0xFF71717A)
private val Zinc400 = Color(0xFFA1A1AA)
private val CyanAccent = Color(0xFF22D3EE)

/**
 * Premium Chat List Screen - Inbox view with glassmorphic design
 */
@Composable
fun PremiumChatListScreen(
    onNavigateBack: () -> Unit = {},
    onNavigateToChat: (String) -> Unit = {},
    onCreateGroup: (Boolean) -> Unit = {},  // isChannel parameter
    viewModel: ChatListViewModel = viewModel()
) {
    // Get conversations from ViewModel
    val conversations by viewModel.conversations.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val error by viewModel.error.collectAsState()
    
    val unreadCount = conversations.sumOf { it.unread }
    
    // Keep screen awake while on Chat page
    val context = androidx.compose.ui.platform.LocalContext.current
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
        AmbientBackground()
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            // Header
            InboxHeader(unreadCount = unreadCount)
            
            // Chat List
            when {
                isLoading -> {
                    // Loading state
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator(
                            color = IndigoAccent,
                            modifier = Modifier.size(40.dp)
                        )
                    }
                }
                conversations.isEmpty() -> {
                    // Empty state
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Icon(
                                imageVector = Icons.Outlined.ChatBubbleOutline,
                                contentDescription = null,
                                tint = Zinc600,
                                modifier = Modifier.size(64.dp)
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = "No Conversations",
                                color = Zinc400,
                                fontSize = 18.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                            Text(
                                text = "Follow people to start chatting",
                                color = Zinc600,
                                fontSize = 14.sp
                            )
                        }
                    }
                }
                else -> {
                    LazyColumn(
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        item {
                            Text(
                                text = "RECENT CHATS",
                                color = Zinc600,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 8.dp)
                            )
                        }
                        
                        items(conversations) { conversation ->
                            PremiumChatItemFromViewModel(
                                conversation = conversation,
                                onClick = { onNavigateToChat(conversation.id) }
                            )
                        }
                        
                        // Bottom spacing for nav bar
                        item {
                            Spacer(modifier = Modifier.height(100.dp))
                        }
                    }
                }
            }
        }
    }
}

/**
 * Ambient Background with blur effect
 */
@Composable
private fun AmbientBackground() {
    Box(modifier = Modifier.fillMaxSize()) {
        // Indigo blur top-left
        Box(
            modifier = Modifier
                .offset(x = (-80).dp, y = (-40).dp)
                .size(300.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            IndigoAccent.copy(alpha = 0.1f),
                            Color.Transparent
                        )
                    )
                )
                .blur(150.dp)
        )
        
        // Purple blur bottom-right
        Box(
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .offset(x = 80.dp, y = 40.dp)
                .size(300.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            PurpleAccent.copy(alpha = 0.1f),
                            Color.Transparent
                        )
                    )
                )
                .blur(150.dp)
        )
    }
}

/**
 * Inbox Header
 */
@Composable
private fun InboxHeader(unreadCount: Int) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(
                text = "Inbox",
                fontSize = 30.sp,
                fontWeight = FontWeight.Black,
                color = Color.White
            )
            if (unreadCount > 0) {
                Text(
                    text = "$unreadCount NEW MESSAGES",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Medium,
                    color = Zinc500,
                    letterSpacing = 1.sp
                )
            }
        }
        
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            // Search button
            IconButton(
                onClick = { /* Search */ },
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(Zinc800.copy(alpha = 0.5f))
                    .border(1.dp, Color.White.copy(alpha = 0.05f), CircleShape)
            ) {
                Icon(
                    imageVector = Icons.Outlined.Search,
                    contentDescription = "Search",
                    tint = Zinc400,
                    modifier = Modifier.size(18.dp)
                )
            }
        }
    }
}

/**
 * Premium Chat Item - Glassmorphic chat list item
 */
@Composable
private fun PremiumChatItem(
    conversation: PremiumConversation,
    onClick: () -> Unit
) {
    val hasUnread = conversation.unread > 0
    
    var isPressed by remember { mutableStateOf(false) }
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.98f else 1f,
        animationSpec = spring(dampingRatio = 0.5f, stiffness = 400f),
        label = "itemScale"
    )
    
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .scale(scale)
            .clip(RoundedCornerShape(24.dp))
            .clickable { onClick() }
            .padding(horizontal = 12.dp, vertical = 14.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            // Avatar with online indicator
            Box {
                
                // Avatar
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .clip(CircleShape)
                        .border(2.dp, DarkBg, CircleShape)
                ) {
                    AsyncImage(
                        model = conversation.avatar,
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )
                }
                
                // Online indicator
                if (conversation.isOnline) {
                    Box(
                        modifier = Modifier
                            .align(Alignment.BottomEnd)
                            .offset(x = 2.dp, y = 2.dp)
                            .size(14.dp)
                            .clip(CircleShape)
                            .background(Color.Black)
                            .padding(2.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .clip(CircleShape)
                                .background(Color(0xFF22C55E))
                        )
                    }
                }
            }
            
            Spacer(modifier = Modifier.width(16.dp))
            
            // Content
            Column(modifier = Modifier.weight(1f)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = conversation.name,
                        fontSize = 16.sp,
                        fontWeight = if (hasUnread) FontWeight.Bold else FontWeight.SemiBold,
                        color = if (hasUnread) Color.White else Color(0xFFE4E4E7)
                    )
                    Text(
                        text = conversation.time,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium,
                        color = if (hasUnread) IndigoAccent else Zinc600
                    )
                }
                
                Spacer(modifier = Modifier.height(4.dp))
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Last message
                    if (conversation.messageType == "photo") {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Filled.Image,
                                contentDescription = null,
                                tint = if (hasUnread) IndigoAccent else Zinc500,
                                modifier = Modifier.size(14.dp)
                            )
                            Text(
                                text = "Photo",
                                fontSize = 15.sp,
                                color = if (hasUnread) IndigoAccent else Zinc500
                            )
                        }
                    } else {
                        Text(
                            text = conversation.lastMessage,
                            fontSize = 15.sp,
                            color = if (hasUnread) Color(0xFFF4F4F5) else Zinc500,
                            fontWeight = if (hasUnread) FontWeight.Medium else FontWeight.Normal,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    
                    // Unread badge
                    if (hasUnread) {
                        Box(
                            modifier = Modifier
                                .padding(start = 8.dp)
                                .height(20.dp)
                                .widthIn(min = 20.dp)
                                .clip(CircleShape)
                                .background(
                                    Brush.horizontalGradient(
                                        colors = listOf(IndigoAccent, PurpleAccent)
                                    )
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = conversation.unread.toString(),
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White,
                                modifier = Modifier.padding(horizontal = 6.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Premium Chat Item from ViewModel - Works with ChatListViewModel.ConversationItem
 */
@Composable
private fun PremiumChatItemFromViewModel(
    conversation: ChatListViewModel.ConversationItem,
    onClick: () -> Unit
) {
    val hasUnread = conversation.unread > 0
    
    var isPressed by remember { mutableStateOf(false) }
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.98f else 1f,
        animationSpec = spring(dampingRatio = 0.5f, stiffness = 400f),
        label = "itemScale"
    )
    
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .scale(scale)
            .clip(RoundedCornerShape(24.dp))
            .clickable { onClick() }
            .padding(horizontal = 12.dp, vertical = 14.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            // Avatar with online indicator
            Box {
                // Avatar
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .clip(CircleShape)
                        .border(2.dp, DarkBg, CircleShape)
                ) {
                    if (conversation.avatar != null) {
                        AsyncImage(
                            model = conversation.avatar,
                            contentDescription = null,
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Crop
                        )
                    } else {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Zinc800),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = conversation.name.firstOrNull()?.uppercase() ?: "?",
                                fontSize = 20.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White
                            )
                        }
                    }
                }
                
                // Online indicator
                if (conversation.isOnline) {
                    Box(
                        modifier = Modifier
                            .align(Alignment.BottomEnd)
                            .offset(x = 2.dp, y = 2.dp)
                            .size(14.dp)
                            .clip(CircleShape)
                            .background(Color.Black)
                            .padding(2.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .clip(CircleShape)
                                .background(Color(0xFF22C55E))
                        )
                    }
                }
            }
            
            Spacer(modifier = Modifier.width(16.dp))
            
            // Content
            Column(modifier = Modifier.weight(1f)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = conversation.name,
                        fontSize = 16.sp,
                        fontWeight = if (hasUnread) FontWeight.Bold else FontWeight.SemiBold,
                        color = if (hasUnread) Color.White else Color(0xFFE4E4E7)
                    )
                    Text(
                        text = conversation.time,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium,
                        color = if (hasUnread) IndigoAccent else Zinc600
                    )
                }
                
                Spacer(modifier = Modifier.height(4.dp))
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = conversation.lastMessage,
                        fontSize = 15.sp,
                        color = if (hasUnread) Color(0xFFF4F4F5) else Zinc500,
                        fontWeight = if (hasUnread) FontWeight.Medium else FontWeight.Normal,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )
                    
                    // Unread badge
                    if (hasUnread) {
                        Box(
                            modifier = Modifier
                                .padding(start = 8.dp)
                                .height(20.dp)
                                .widthIn(min = 20.dp)
                                .clip(CircleShape)
                                .background(
                                    Brush.horizontalGradient(
                                        colors = listOf(IndigoAccent, PurpleAccent)
                                    )
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = conversation.unread.toString(),
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White,
                                modifier = Modifier.padding(horizontal = 6.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Premium Inner Chat Screen - Chat conversation view
 */
@Composable
fun PremiumInnerChatScreen(
    conversationId: String,
    onNavigateBack: () -> Unit = {},
    onNavigateToCall: () -> Unit = {},
    viewModel: ChatViewModel = viewModel()
) {
    val messages by viewModel.messages.collectAsState()
    val partnerUser by viewModel.partnerUser.collectAsState()
    val isOnline by viewModel.isOnline.collectAsState()
    
    var messageInput by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    val isLoading by viewModel.isLoading.collectAsState()
    
    LaunchedEffect(conversationId) {
        viewModel.loadConversation(conversationId)
    }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBg)
    ) {
        // Background pattern
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    brush = Brush.verticalGradient(
                        colors = listOf(DarkBg, Color(0xFF0F0F12))
                    )
                )
        )
        // Dot pattern overlay
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.White.copy(alpha = 0.03f))
        )
        
        Column(modifier = Modifier.fillMaxSize()) {
            // Chat Header
            ChatHeader(
                name = partnerUser?.username ?: if (isLoading) "Loading..." else "User",
                avatar = partnerUser?.avatar ?: "",
                isOnline = isOnline,
                onBack = onNavigateBack,
                onCall = onNavigateToCall
            )
            
            // Messages Area
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
                reverseLayout = false
            ) {
                // Date separator
                item {
                    Box(
                        modifier = Modifier.fillMaxWidth(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "TODAY",
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            color = Zinc500,
                            letterSpacing = 2.sp,
                            modifier = Modifier
                                .clip(CircleShape)
                                .background(Zinc800.copy(alpha = 0.5f))
                                .border(1.dp, Color.White.copy(alpha = 0.05f), CircleShape)
                                .padding(horizontal = 16.dp, vertical = 6.dp)
                        )
                    }
                }
                
                // Show real messages from ViewModel
                if (messages.isEmpty()) {
                    item {
                        Box(
                            modifier = Modifier.fillMaxWidth().padding(32.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "Send a message to start chatting!",
                                color = Zinc500,
                                fontSize = 14.sp
                            )
                        }
                    }
                } else {
                    items(messages) { message ->
                        val isFromMe = viewModel.isFromCurrentUser(message)
                        val time = try {
                            val timestamp = message.createdAt.toLong()
                            val formatter = java.text.SimpleDateFormat("h:mm a", java.util.Locale.getDefault())
                            formatter.format(java.util.Date(timestamp))
                        } catch (e: Exception) {
                            ""
                        }
                        
                        MessageBubble(
                            message = PremiumChatMessage(
                                id = message.messageId,
                                text = message.content,
                                imageUrl = message.mediaUrl,
                                isMine = isFromMe,
                                time = time,
                                status = if (message.isRead) "read" else "sent"
                            )
                        )
                    }
                }
                
                // Bottom spacing for input
                item {
                    Spacer(modifier = Modifier.height(80.dp))
                }
            }
        }
        
        // Floating Input Area - with keyboard handling
        FloatingInputBar(
            value = messageInput,
            onValueChange = { messageInput = it },
            onSend = {
                if (messageInput.isNotBlank()) {
                    viewModel.sendMessage(messageInput)
                    messageInput = ""
                }
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .windowInsetsPadding(WindowInsets.ime)
                .windowInsetsPadding(WindowInsets.navigationBars)
                .padding(horizontal = 16.dp, vertical = 8.dp)
        )
    }
}

/**
 * Chat Header
 */
@Composable
private fun ChatHeader(
    name: String,
    avatar: String,
    isOnline: Boolean,
    onBack: () -> Unit,
    onCall: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(ZincDark.copy(alpha = 0.95f))
            .statusBarsPadding()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Back button
            IconButton(onClick = onBack) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint = Color(0xFFD4D4D8),
                    modifier = Modifier.size(22.dp)
                )
            }
            
            Spacer(modifier = Modifier.width(4.dp))
            
            // Avatar
            Box {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                ) {
                    AsyncImage(
                        model = avatar,
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )
                }
                if (isOnline) {
                    Box(
                        modifier = Modifier
                            .align(Alignment.BottomEnd)
                            .size(12.dp)
                            .clip(CircleShape)
                            .background(ZincDark)
                            .padding(2.dp)
                            .clip(CircleShape)
                            .background(Color(0xFF22C55E))
                    )
                }
            }
            
            Spacer(modifier = Modifier.width(12.dp))
            
            // Name and status
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = name,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Text(
                    text = if (isOnline) "Online now" else "Seen 2h ago",
                    fontSize = 12.sp,
                    color = Zinc400,
                    fontWeight = FontWeight.Medium
                )
            }
            
            // Call buttons
            IconButton(onClick = onCall) {
                Icon(
                    imageVector = Icons.Filled.Phone,
                    contentDescription = "Call",
                    tint = Zinc400,
                    modifier = Modifier.size(20.dp)
                )
            }
            IconButton(onClick = { /* Video call */ }) {
                Icon(
                    imageVector = Icons.Filled.Videocam,
                    contentDescription = "Video",
                    tint = Zinc400,
                    modifier = Modifier.size(22.dp)
                )
            }
        }
        
        // Bottom border
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .align(Alignment.BottomCenter)
                .background(Color.White.copy(alpha = 0.1f))
        )
    }
}

/**
 * Message Bubble
 */
@Composable
private fun MessageBubble(message: PremiumChatMessage) {
    val isMine = message.isMine
    
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isMine) Arrangement.End else Arrangement.Start
    ) {
        Column(
            horizontalAlignment = if (isMine) Alignment.End else Alignment.Start,
            modifier = Modifier.widthIn(max = 280.dp)
        ) {
            // Message content
            if (message.text != null) {
                Box(
                    modifier = Modifier
                        .clip(
                            RoundedCornerShape(
                                topStart = 24.dp,
                                topEnd = 24.dp,
                                bottomStart = if (isMine) 24.dp else 0.dp,
                                bottomEnd = if (isMine) 0.dp else 24.dp
                            )
                        )
                        .background(
                            if (isMine) VioletAccent
                            else Color(0xFF1E1E22)
                        )
                        .then(
                            if (!isMine) Modifier.border(
                                1.dp,
                                Color.White.copy(alpha = 0.05f),
                                RoundedCornerShape(
                                    topStart = 24.dp,
                                    topEnd = 24.dp,
                                    bottomEnd = 24.dp
                                )
                            ) else Modifier
                        )
                        .padding(horizontal = 20.dp, vertical = 14.dp)
                ) {
                    Text(
                        text = message.text,
                        fontSize = 15.sp,
                        lineHeight = 22.sp,
                        color = if (isMine) Color.Black else Color(0xFFE4E4E7),
                        fontWeight = if (isMine) FontWeight.Medium else FontWeight.Normal
                    )
                }
            } else if (message.imageUrl != null) {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(24.dp))
                        .background(Color(0xFF1E1E22))
                        .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(24.dp))
                        .padding(4.dp)
                ) {
                    AsyncImage(
                        model = message.imageUrl,
                        contentDescription = null,
                        modifier = Modifier
                            .widthIn(max = 250.dp)
                            .clip(RoundedCornerShape(20.dp)),
                        contentScale = ContentScale.FillWidth
                    )
                }
            }
            
            // Time and status
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = if (isMine) Arrangement.End else Arrangement.Start,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
            ) {
                if (isMine) {
                    // Status icon
                    Icon(
                        imageVector = when (message.status) {
                            "read" -> Icons.Outlined.Visibility       // Open eye = seen
                            "sent" -> Icons.Outlined.VisibilityOff    // Closed eye = sent but not read
                            "sending" -> Icons.Filled.Bolt            // Thunder = sending
                            else -> Icons.Outlined.VisibilityOff      // Default = closed eye (sent)
                        },
                        contentDescription = null,
                        tint = when (message.status) {
                            "read" -> VioletAccent                    // Purple for seen
                            "sending" -> Color(0xFFFFD700)            // Yellow/Gold for sending
                            else -> Zinc500                           // Gray for sent
                        },
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                }
                Text(
                    text = message.time,
                    fontSize = 10.sp,
                    color = Zinc600,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }
}

/**
 * Floating Input Bar
 */
@Composable
private fun FloatingInputBar(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(28.dp))
            .background(Color(0xFF121212))
            .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(28.dp))
            .padding(6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Plus button
        IconButton(
            onClick = { /* Attachments */ },
            modifier = Modifier
                .size(40.dp)
                .clip(CircleShape)
                .background(Zinc800)
        ) {
            Icon(
                imageVector = Icons.Filled.Add,
                contentDescription = "Add",
                tint = Zinc400,
                modifier = Modifier.size(20.dp)
            )
        }
        
        // Text input
        BasicTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 12.dp),
            textStyle = TextStyle(
                color = Color.White,
                fontSize = 14.sp
            ),
            cursorBrush = SolidColor(VioletAccent),
            decorationBox = { innerTextField ->
                Box {
                    if (value.isEmpty()) {
                        Text(
                            text = "Type a message...",
                            color = Zinc600,
                            fontSize = 14.sp
                        )
                    }
                    innerTextField()
                }
            }
        )
        
        if (value.isEmpty()) {
            // Emoji and Mic buttons
            IconButton(onClick = { /* Emoji */ }) {
                Icon(
                    imageVector = Icons.Outlined.EmojiEmotions,
                    contentDescription = "Emoji",
                    tint = Zinc500,
                    modifier = Modifier.size(20.dp)
                )
            }
            IconButton(onClick = { /* Voice */ }) {
                Icon(
                    imageVector = Icons.Filled.Mic,
                    contentDescription = "Voice",
                    tint = Zinc500,
                    modifier = Modifier.size(20.dp)
                )
            }
        } else {
            // Send button
            IconButton(
                onClick = onSend,
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(VioletAccent)
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.Send,
                    contentDescription = "Send",
                    tint = Color.Black,
                    modifier = Modifier
                        .size(18.dp)
                        .offset(x = 1.dp)
                )
            }
        }
    }
}

// Data classes
private data class PremiumConversation(
    val id: String,
    val name: String,
    val avatar: String,
    val lastMessage: String,
    val time: String,
    val unread: Int,
    val isOnline: Boolean,
    val messageType: String = "text"
)

private data class PremiumChatMessage(
    val id: String,
    val text: String? = null,
    val imageUrl: String? = null,
    val isMine: Boolean,
    val time: String,
    val status: String = "sent"
)
