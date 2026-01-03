package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
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
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.zIndex
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import android.net.Uri
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.ui.AspectRatioFrameLayout
import androidx.media3.ui.PlayerView
import com.orignal.buddylynk.data.cache.VideoPlayerCache
import com.orignal.buddylynk.data.cache.FeedPlayerManager
import com.orignal.buddylynk.ui.viewmodel.TeamUpViewModel

// Premium Colors
private val DarkBg = Color(0xFF050505)
private val IndigoAccent = Color(0xFF6366F1)
private val CyanAccent = Color(0xFF22D3EE)
private val BlueAccent = Color(0xFF3B82F6)
private val PurpleAccent = Color(0xFF8B5CF6)
private val PinkAccent = Color(0xFFEC4899)
private val ZincDark = Color(0xFF0A0A0A)
private val Zinc800 = Color(0xFF27272A)
private val Zinc600 = Color(0xFF52525B)
private val Zinc500 = Color(0xFF71717A)
private val Zinc400 = Color(0xFFA1A1AA)
private val Zinc200 = Color(0xFFE4E4E7)

/**
 * Premium TeamUp Screen - Team list view with React design
 */
@Composable
fun PremiumTeamUpScreen(
    onNavigateBack: () -> Unit = {},
    onNavigateToTeam: (String) -> Unit = {},
    onInnerViewChanged: (Boolean) -> Unit = {}, // Callback to hide/show bottom nav
    onCreateGroup: (Boolean) -> Unit = {},  // isChannel: true = channel, false = group
    teamsViewModel: TeamUpViewModel = viewModel()
) {
    var showCreateMenu by remember { mutableStateOf(false) }
    var selectedTeam by remember { mutableStateOf<PremiumTeam?>(null) }
    var view by remember { mutableStateOf("list") } // "list" or "inner"
    
    // Notify parent when inner view state changes
    val isInnerView = view == "inner"
    LaunchedEffect(isInnerView) {
        onInnerViewChanged(isInnerView)
    }
    
    // Reset nav state when screen is disposed (user navigates away)
    DisposableEffect(Unit) {
        onDispose {
            onInnerViewChanged(false)
        }
    }
    
    // Keep screen awake while on TeamUp page
    val context = androidx.compose.ui.platform.LocalContext.current
    val activity = context as? android.app.Activity
    DisposableEffect(Unit) {
        activity?.window?.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            activity?.window?.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }
    
    // Get teams from ViewModel
    val teamsFromVm by teamsViewModel.teams.collectAsState()
    val isLoading by teamsViewModel.isLoading.collectAsState()
    
    // Convert ViewModel data to PremiumTeam for compatibility
    val teams = teamsFromVm.map { item ->
        PremiumTeam(
            id = item.id,
            name = item.name,
            type = item.type,
            members = item.members,
            active = item.active,
            avatar = item.avatar ?: "",
            lastMsg = item.lastMsg,
            time = item.time,
            unread = item.unread
        )
    }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBg)
    ) {
        // Ambient Background
        TeamUpAmbientBackground()
        
        // List View
        AnimatedVisibility(
            visible = view == "list",
            enter = fadeIn() + scaleIn(initialScale = 0.95f),
            exit = fadeOut() + scaleOut(targetScale = 0.95f)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .statusBarsPadding()
            ) {
                // Header
                TeamUpHeader()
                
                // Content
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    // Stats Cards
                    item {
                        val groupCount = teams.count { it.type == "group" }
                        val channelCount = teams.count { it.type == "channel" }
                        StatsCardsRow(groupCount = groupCount, channelCount = channelCount)
                    }
                    
                    // Section Header
                    item {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 8.dp, vertical = 8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "YOUR TEAMS",
                                color = Zinc500,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                            Text(
                                text = "Sort by Activity",
                                color = CyanAccent,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                    
                    // Loading State
                    if (isLoading) {
                        item {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(200.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Column(
                                    horizontalAlignment = Alignment.CenterHorizontally
                                ) {
                                    CircularProgressIndicator(
                                        color = CyanAccent,
                                        strokeWidth = 3.dp,
                                        modifier = Modifier.size(48.dp)
                                    )
                                    Spacer(modifier = Modifier.height(16.dp))
                                    Text(
                                        text = "Loading Teams...",
                                        color = Zinc400,
                                        fontSize = 14.sp
                                    )
                                }
                            }
                        }
                    } else if (teams.isEmpty()) {
                        // Empty State
                        item {
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
                                        imageVector = Icons.Outlined.Groups,
                                        contentDescription = null,
                                        tint = Zinc600,
                                        modifier = Modifier.size(64.dp)
                                    )
                                    Spacer(modifier = Modifier.height(16.dp))
                                    Text(
                                        text = "No Teams Yet",
                                        color = Zinc400,
                                        fontSize = 18.sp,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                    Text(
                                        text = "Create or join a team to get started",
                                        color = Zinc600,
                                        fontSize = 14.sp
                                    )
                                }
                            }
                        }
                    } else {
                        items(teams) { team ->
                            TeamItem(
                                team = team,
                                onClick = {
                                    selectedTeam = team
                                    view = "inner"
                                }
                            )
                        }
                    }
                    
                    // Bottom spacing
                    item {
                        Spacer(modifier = Modifier.height(140.dp))
                    }
                }
            }
        }
        
        // FAB - outside AnimatedVisibility for proper positioning
        if (view == "list") {
            CreateFAB(
                showMenu = showCreateMenu,
                onToggle = { showCreateMenu = !showCreateMenu },
                onNewGroup = { onCreateGroup(false) },
                onNewChannel = { onCreateGroup(true) },
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(end = 24.dp, bottom = 110.dp)
            )
        }
        
        // Inner Team View - with zIndex and fillMaxSize to overlay bottom navigation
        AnimatedVisibility(
            visible = view == "inner" && selectedTeam != null,
            enter = slideInHorizontally(initialOffsetX = { it }),
            exit = slideOutHorizontally(targetOffsetX = { it }),
            modifier = Modifier
                .fillMaxSize()
                .zIndex(100f)
        ) {
            selectedTeam?.let { team ->
                InnerTeamView(
                    team = team,
                    onBack = {
                        view = "list"
                        selectedTeam = null
                    }
                )
            }
        }
    }
}

/**
 * Ambient Background
 */
@Composable
private fun TeamUpAmbientBackground() {
    Box(modifier = Modifier.fillMaxSize()) {
        // Cyan blur top-right
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .offset(x = 40.dp, y = (-40).dp)
                .size(280.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(CyanAccent.copy(alpha = 0.1f), Color.Transparent)
                    )
                )
                .blur(150.dp)
        )
        
        // Blue blur bottom-left
        Box(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .offset(x = (-40).dp, y = 40.dp)
                .size(280.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(BlueAccent.copy(alpha = 0.1f), Color.Transparent)
                    )
                )
                .blur(150.dp)
        )
    }
}

/**
 * TeamUp Header
 */
@Composable
private fun TeamUpHeader() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(
                text = "Team Up",
                fontSize = 30.sp,
                fontWeight = FontWeight.Black,
                color = Color.White
            )
            Text(
                text = "COLLABORATE & CONNECT",
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium,
                color = Zinc500,
                letterSpacing = 1.sp
            )
        }
        
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

/**
 * Stats Cards Row
 */
@Composable
private fun StatsCardsRow(groupCount: Int = 0, channelCount: Int = 0) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Groups Card
        StatsCard(
            count = groupCount,
            label = "JOINED GROUPS",
            icon = Icons.Filled.People,
            gradientColors = listOf(CyanAccent, BlueAccent),
            modifier = Modifier.weight(1f)
        )
        
        // Channels Card
        StatsCard(
            count = channelCount,
            label = "JOINED CHANNELS",
            icon = Icons.Filled.Tag,
            gradientColors = listOf(PurpleAccent, PinkAccent),
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun StatsCard(
    count: Int,
    label: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    gradientColors: List<Color>,
    modifier: Modifier = Modifier
) {
    var isPressed by remember { mutableStateOf(false) }
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.95f else 1f,
        animationSpec = spring(dampingRatio = 0.5f),
        label = "cardScale"
    )
    
    Box(
        modifier = modifier
            .scale(scale)
            .height(170.dp)
            .clip(RoundedCornerShape(30.dp))
            .background(ZincDark)
            .border(1.dp, Color.White.copy(alpha = 0.05f), RoundedCornerShape(30.dp))
            .clickable { }
    ) {
        // Gradient blur effects
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .size(100.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(gradientColors[0].copy(alpha = 0.1f), Color.Transparent)
                    )
                )
                .blur(60.dp)
        )
        Box(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .size(80.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(gradientColors[1].copy(alpha = 0.1f), Color.Transparent)
                    )
                )
                .blur(50.dp)
        )
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(20.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Icon
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(gradientColors[0].copy(alpha = 0.1f))
                    .border(1.dp, gradientColors[0].copy(alpha = 0.2f), RoundedCornerShape(16.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = gradientColors[0],
                    modifier = Modifier.size(24.dp)
                )
            }
            
            // Count and Label
            Column {
                Text(
                    text = count.toString(),
                    fontSize = 36.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Text(
                    text = label,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    color = gradientColors[0].copy(alpha = 0.8f),
                    letterSpacing = 1.sp
                )
            }
        }
    }
}

/**
 * Team Item
 */
@Composable
private fun TeamItem(
    team: PremiumTeam,
    onClick: () -> Unit
) {
    val hasUnread = team.unread > 0
    
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 4.dp)
    ) {
        // Unread indicator bar
        if (hasUnread) {
            Box(
                modifier = Modifier
                    .align(Alignment.CenterStart)
                    .width(4.dp)
                    .height(40.dp)
                    .clip(RoundedCornerShape(topEnd = 4.dp, bottomEnd = 4.dp))
                    .background(
                        Brush.verticalGradient(
                            colors = listOf(CyanAccent, BlueAccent)
                        )
                    )
            )
        }
        
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = if (hasUnread) 12.dp else 0.dp, top = 16.dp, bottom = 16.dp, end = 0.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Avatar with type badge
            Box {
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .border(
                            1.dp,
                            if (hasUnread) CyanAccent.copy(alpha = 0.3f) else Color.White.copy(alpha = 0.1f),
                            RoundedCornerShape(16.dp)
                        )
                ) {
                    AsyncImage(
                        model = team.avatar,
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )
                }
                
                // Type badge
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .offset(x = 8.dp, y = 8.dp)
                        .size(28.dp)
                        .clip(CircleShape)
                        .background(DarkBg)
                        .padding(3.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(CircleShape)
                            .background(
                                if (team.type == "group")
                                    Brush.linearGradient(listOf(CyanAccent, BlueAccent))
                                else
                                    Brush.linearGradient(listOf(PurpleAccent, PinkAccent))
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = if (team.type == "group") Icons.Filled.People else Icons.Filled.Tag,
                            contentDescription = null,
                            tint = if (team.type == "group") Color.Black else Color.White,
                            modifier = Modifier.size(12.dp)
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
                        text = team.name,
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (hasUnread) Color.White else Color(0xFFD4D4D8)
                    )
                    Text(
                        text = team.time,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Medium,
                        color = if (hasUnread) CyanAccent else Zinc600
                    )
                }
                
                Spacer(modifier = Modifier.height(4.dp))
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = team.lastMsg,
                        fontSize = 14.sp,
                        color = if (hasUnread) Zinc200 else Zinc500,
                        fontWeight = if (hasUnread) FontWeight.Medium else FontWeight.Normal,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )
                    
                    if (hasUnread) {
                        Box(
                            modifier = Modifier
                                .padding(start = 8.dp)
                                .height(22.dp)
                                .widthIn(min = 22.dp)
                                .clip(CircleShape)
                                .background(CyanAccent.copy(alpha = 0.1f))
                                .border(1.dp, CyanAccent.copy(alpha = 0.3f), CircleShape),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = team.unread.toString(),
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                color = CyanAccent,
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
 * Create FAB with menu
 */
@Composable
private fun CreateFAB(
    showMenu: Boolean,
    onToggle: () -> Unit,
    onNewGroup: () -> Unit = {},
    onNewChannel: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val rotation by animateFloatAsState(
        targetValue = if (showMenu) 45f else 0f,
        animationSpec = spring(dampingRatio = 0.6f, stiffness = 400f),
        label = "fabRotation"
    )
    
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.End,
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Menu items
        AnimatedVisibility(
            visible = showMenu,
            enter = fadeIn() + scaleIn(initialScale = 0.75f) + slideInVertically(initialOffsetY = { 40 }),
            exit = fadeOut() + scaleOut(targetScale = 0.75f) + slideOutVertically(targetOffsetY = { 40 })
        ) {
            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp),
                horizontalAlignment = Alignment.End
            ) {
                // New Channel
                Row(
                    modifier = Modifier
                        .clip(CircleShape)
                        .background(Zinc800)
                        .border(1.dp, Color.White.copy(alpha = 0.05f), CircleShape)
                        .clickable { onNewChannel(); onToggle() }
                        .padding(horizontal = 20.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = "New Channel",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = Color.White
                    )
                    Icon(
                        imageVector = Icons.Filled.Tag,
                        contentDescription = null,
                        tint = PurpleAccent,
                        modifier = Modifier.size(18.dp)
                    )
                }
                
                // New Group
                Row(
                    modifier = Modifier
                        .clip(CircleShape)
                        .background(Zinc800)
                        .border(1.dp, Color.White.copy(alpha = 0.05f), CircleShape)
                        .clickable { onNewGroup(); onToggle() }
                        .padding(horizontal = 20.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = "New Group",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = Color.White
                    )
                    Icon(
                        imageVector = Icons.Filled.People,
                        contentDescription = null,
                        tint = CyanAccent,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }
        
        // FAB button
        Box(
            modifier = Modifier
                .size(56.dp)
                .clip(CircleShape)
                .background(
                    brush = if (showMenu) Brush.linearGradient(listOf(Zinc800, Zinc800))
                    else Brush.linearGradient(listOf(CyanAccent, BlueAccent))
                )
                .border(1.dp, Color.White.copy(alpha = 0.1f), CircleShape)
                .clickable(onClick = onToggle),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Filled.Add,
                contentDescription = "Create",
                tint = if (showMenu) Zinc400 else Color.White,
                modifier = Modifier
                    .size(28.dp)
                    .rotate(rotation)
            )
        }
    }
}

/**
 * Inner Team View
 */
@Composable
private fun InnerTeamView(
    team: PremiumTeam,
    onBack: () -> Unit,
    viewModel: TeamUpViewModel = viewModel()
) {
    var messageInput by remember { mutableStateOf("") }
    
    // Load real messages from database
    val messages by viewModel.groupMessages.collectAsState()
    val isLoading by viewModel.messagesLoading.collectAsState()
    
    // Load messages when view opens
    LaunchedEffect(team.id) {
        viewModel.loadGroupMessages(team.id)
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
                .background(Color.White.copy(alpha = 0.03f))
        )
        
        Column(modifier = Modifier.fillMaxSize()) {
            // Header
            InnerTeamHeader(team = team, onBack = onBack)
            
            // Messages
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Welcome banner
                item {
                    WelcomeBanner(team = team)
                }
                
                // Loading indicator
                if (isLoading) {
                    item {
                        Box(
                            modifier = Modifier.fillMaxWidth().padding(16.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(color = IndigoAccent, modifier = Modifier.size(24.dp))
                        }
                    }
                }
                
                // Real messages from database
                items(
                    items = messages,
                    key = { it.id }
                ) { message ->
                    TeamMessageBubble(
                        message = TeamMessage(
                            id = message.id,
                            text = message.content,
                            sender = if (message.isMe) "me" else message.senderName,
                            avatar = message.senderAvatar,
                            time = message.createdAt,
                            role = if (message.isMe) "me" else "member",
                            media = message.media.map { m -> MessageMedia(type = m.type, url = m.url) }
                        )
                    )
                }
                
                // Empty state if no messages
                if (!isLoading && messages.isEmpty()) {
                    item {
                        Box(
                            modifier = Modifier.fillMaxWidth().padding(32.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "No messages yet. Start the conversation!",
                                color = Zinc500,
                                fontSize = 14.sp
                            )
                        }
                    }
                }
                
                item {
                    Spacer(modifier = Modifier.height(80.dp))
                }
            }
        }
        
        // Input bar
        TeamInputBar(
            value = messageInput,
            onValueChange = { messageInput = it },
            onSend = { 
                if (messageInput.isNotBlank()) {
                    viewModel.sendGroupMessage(team.id, messageInput)
                    messageInput = "" 
                }
            },
            teamName = team.name,
            isChannel = team.type == "channel",
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .windowInsetsPadding(WindowInsets.ime)
                .windowInsetsPadding(WindowInsets.navigationBars)
                .padding(horizontal = 16.dp, vertical = 16.dp)
        )
    }
}

@Composable
private fun InnerTeamHeader(
    team: PremiumTeam,
    onBack: () -> Unit
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
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(12.dp))
            ) {
                AsyncImage(
                    model = team.avatar,
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
            }
            
            Spacer(modifier = Modifier.width(12.dp))
            
            // Name and info
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = team.name,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    if (team.type == "channel") {
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "CH",
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold,
                            color = PurpleAccent,
                            modifier = Modifier
                                .background(PurpleAccent.copy(alpha = 0.2f), RoundedCornerShape(4.dp))
                                .padding(horizontal = 4.dp, vertical = 2.dp)
                        )
                    }
                }
                Text(
                    text = "${formatMembers(team.members)} members • ${team.active} online",
                    fontSize = 12.sp,
                    color = Zinc400,
                    fontWeight = FontWeight.Medium
                )
            }
            
            IconButton(onClick = { }) {
                Icon(Icons.Outlined.Search, null, tint = Zinc400, modifier = Modifier.size(20.dp))
            }
            IconButton(onClick = { }) {
                Icon(Icons.Outlined.Settings, null, tint = Zinc400, modifier = Modifier.size(20.dp))
            }
        }
        
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .align(Alignment.BottomCenter)
                .background(Color.White.copy(alpha = 0.1f))
        )
    }
}

@Composable
private fun WelcomeBanner(team: PremiumTeam) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            modifier = Modifier
                .size(64.dp)
                .clip(RoundedCornerShape(16.dp))
                .background(Zinc800)
                .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(16.dp)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = if (team.type == "group") Icons.Filled.People else Icons.Filled.Radio,
                contentDescription = null,
                tint = if (team.type == "group") CyanAccent else PurpleAccent,
                modifier = Modifier.size(32.dp)
            )
        }
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "Welcome to the beginning of the",
            fontSize = 14.sp,
            color = Zinc400
        )
        Text(
            text = team.name,
            fontSize = 14.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        Text(
            text = "history.",
            fontSize = 14.sp,
            color = Zinc400
        )
    }
}

@Composable
private fun TeamMessageBubble(message: TeamMessage) {
    val isMine = message.sender == "me"
    
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isMine) Arrangement.End else Arrangement.Start
    ) {
        // Avatar for others
        if (!isMine && message.avatar != null) {
            AsyncImage(
                model = message.avatar,
                contentDescription = null,
                modifier = Modifier
                    .size(32.dp)
                    .clip(CircleShape)
                    .border(1.dp, Color.White.copy(alpha = 0.1f), CircleShape),
                contentScale = ContentScale.Crop
            )
            Spacer(modifier = Modifier.width(8.dp))
        }
        
        Column(
            horizontalAlignment = if (isMine) Alignment.End else Alignment.Start,
            modifier = Modifier.widthIn(max = 280.dp)
        ) {
            // Sender name
            if (!isMine) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = message.sender,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = Zinc400
                    )
                    if (message.role == "admin") {
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "ADMIN",
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold,
                            color = CyanAccent,
                            modifier = Modifier
                                .background(CyanAccent.copy(alpha = 0.2f), RoundedCornerShape(4.dp))
                                .padding(horizontal = 4.dp, vertical = 1.dp)
                        )
                    }
                }
            }
            
            // Bubble with media and text
            Column(
                modifier = Modifier
                    .clip(
                        RoundedCornerShape(
                            topStart = 20.dp,
                            topEnd = 20.dp,
                            bottomStart = if (isMine) 20.dp else 0.dp,
                            bottomEnd = if (isMine) 0.dp else 20.dp
                        )
                    )
                    .background(
                        if (isMine) CyanAccent.copy(alpha = 0.9f)
                        else Color(0xFF1E1E22)
                    )
                    .then(
                        if (!isMine) Modifier.border(
                            1.dp,
                            Color.White.copy(alpha = 0.05f),
                            RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp, bottomEnd = 20.dp)
                        ) else Modifier
                    )
            ) {
                // Display images/videos from S3
                if (message.media.isNotEmpty()) {
                    Column(
                        modifier = Modifier
                            .padding(4.dp)
                            .clip(RoundedCornerShape(16.dp))
                    ) {
                        message.media.forEachIndexed { index, media ->
                            if (media.type == "image") {
                                AsyncImage(
                                    model = media.url,
                                    contentDescription = "Image ${index + 1}",
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .heightIn(max = 300.dp)
                                        .clip(RoundedCornerShape(12.dp)),
                                    contentScale = ContentScale.Crop
                                )
                            } else if (media.type == "video") {
                                // Video Player with pooled ExoPlayer (no re-buffering!)
                                val context = LocalContext.current
                                var isBuffering by remember { mutableStateOf(true) }
                                var hasError by remember { mutableStateOf(false) }
                                
                                // Initialize and get player from pool
                                FeedPlayerManager.init(context)
                                
                                @androidx.annotation.OptIn(UnstableApi::class)
                                val exoPlayer = remember(media.url) {
                                    FeedPlayerManager.getPlayer(
                                        url = media.url,
                                        onBufferingChange = { buffering -> isBuffering = buffering },
                                        onError = { hasError = true }
                                    )
                                }
                                
                                // Pause when scrolled away (keep in pool for instant resume)
                                DisposableEffect(media.url) {
                                    onDispose { FeedPlayerManager.pausePlayer(media.url) }
                                }
                                
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(180.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(Zinc800)
                                ) {
                                    AndroidView(
                                        factory = { ctx ->
                                            PlayerView(ctx).apply {
                                                player = exoPlayer
                                                useController = false
                                                resizeMode = AspectRatioFrameLayout.RESIZE_MODE_ZOOM
                                                layoutParams = FrameLayout.LayoutParams(
                                                    ViewGroup.LayoutParams.MATCH_PARENT,
                                                    ViewGroup.LayoutParams.MATCH_PARENT
                                                )
                                            }
                                        },
                                        modifier = Modifier.fillMaxSize()
                                    )
                                    
                                    // Loading indicator
                                    if (isBuffering) {
                                        Box(
                                            modifier = Modifier.fillMaxSize(),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            CircularProgressIndicator(
                                                color = CyanAccent,
                                                modifier = Modifier.size(32.dp)
                                            )
                                        }
                                    }
                                    
                                    // Video badge
                                    Box(
                                        modifier = Modifier
                                            .align(Alignment.TopEnd)
                                            .padding(8.dp)
                                            .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(4.dp))
                                            .padding(4.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Outlined.PlayCircle,
                                            contentDescription = null,
                                            tint = Color.White,
                                            modifier = Modifier.size(16.dp)
                                        )
                                    }
                                }
                            }
                            if (index < message.media.size - 1) {
                                Spacer(modifier = Modifier.height(4.dp))
                            }
                        }
                    }
                }
                
                // Text content (if any)
                if (message.text.isNotBlank()) {
                    Box(
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp)
                    ) {
                        Text(
                            text = message.text,
                            fontSize = 14.sp,
                            lineHeight = 20.sp,
                            color = if (isMine) Color.Black else Zinc200
                        )
                    }
                }
            }
            
            // Time
            Text(
                text = message.time,
                fontSize = 9.sp,
                color = Zinc600,
                fontWeight = FontWeight.Medium,
                modifier = Modifier.padding(horizontal = 4.dp, vertical = 4.dp)
            )
        }
    }
}

@Composable
private fun TeamInputBar(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    teamName: String,
    isChannel: Boolean,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .background(Color(0xFF121212))
            .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(24.dp))
            .padding(6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Plus button
        IconButton(
            onClick = { },
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
        
        // Input
        BasicTextField(
            value = value,
            onValueChange = onValueChange,
            enabled = !isChannel,
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 12.dp),
            textStyle = TextStyle(
                color = Color.White,
                fontSize = 14.sp
            ),
            cursorBrush = SolidColor(CyanAccent),
            decorationBox = { innerTextField ->
                Box {
                    if (value.isEmpty()) {
                        Text(
                            text = if (isChannel) "Message (Admin Only)" else "Message $teamName...",
                            color = Zinc600,
                            fontSize = 14.sp
                        )
                    }
                    innerTextField()
                }
            }
        )
        
        if (value.isEmpty()) {
            IconButton(onClick = { }) {
                Icon(
                    imageVector = Icons.Outlined.EmojiEmotions,
                    contentDescription = "Emoji",
                    tint = Zinc500,
                    modifier = Modifier.size(20.dp)
                )
            }
        } else {
            IconButton(
                onClick = onSend,
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(CyanAccent)
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

private fun formatMembers(count: Int): String {
    return when {
        count >= 1000000 -> "${count / 1000000}M"
        count >= 1000 -> "${count / 1000}K"
        else -> count.toString()
    }
}

// Data classes
private data class PremiumTeam(
    val id: String,
    val name: String,
    val type: String, // "group" or "channel"
    val members: Int,
    val active: Int,
    val avatar: String,
    val lastMsg: String,
    val time: String,
    val unread: Int
)

// Media item for posts
private data class MessageMedia(
    val type: String,  // "image" or "video"
    val url: String
)

private data class TeamMessage(
    val id: String,
    val text: String,
    val sender: String,
    val avatar: String?,
    val time: String,
    val role: String,
    val status: String = "",
    val media: List<MessageMedia> = emptyList()
)
