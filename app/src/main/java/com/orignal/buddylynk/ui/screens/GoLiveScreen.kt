package com.orignal.buddylynk.ui.screens

import android.Manifest
import android.content.Context
import android.util.Log
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.*
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale
import com.orignal.buddylynk.ui.theme.*
import kotlinx.coroutines.delay
import java.util.concurrent.Executors

import androidx.lifecycle.viewmodel.compose.viewModel
import com.orignal.buddylynk.ui.viewmodel.LiveViewModel

/**
 * GoLiveScreen - Start a live stream with camera preview
 */
@OptIn(ExperimentalPermissionsApi::class, ExperimentalMaterial3Api::class)
@Composable
fun GoLiveScreen(
    onNavigateBack: () -> Unit,
    viewModel: LiveViewModel = viewModel()
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    
    // ViewModel State
    val isStreaming by viewModel.isStreaming.collectAsState()
    val viewerCount by viewModel.viewerCount.collectAsState()
    val streamDuration by viewModel.streamDuration.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    
    // UI State
    var streamTitle by remember { mutableStateOf("") }
    var selectedCategory by remember { mutableStateOf("Just Chatting") }
    var isFrontCamera by remember { mutableStateOf(true) }
    var showCategorySheet by remember { mutableStateOf(false) }
    var showExitDialog by remember { mutableStateOf(false) }
    
    // Exit confirmation dialog
    if (showExitDialog) {
        AlertDialog(
            onDismissRequest = { showExitDialog = false },
            containerColor = DarkSurface,
            titleContentColor = TextPrimary,
            textContentColor = TextSecondary,
            title = {
                Text(
                    text = "End Live Stream?",
                    fontWeight = FontWeight.Bold
                )
            },
            text = {
                Text("If you go back, your live stream will be cancelled and viewers will be disconnected.")
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showExitDialog = false
                        viewModel.stopStream()
                        onNavigateBack()
                    }
                ) {
                    Text("End Stream", color = LikeRed)
                }
            },
            dismissButton = {
                TextButton(onClick = { showExitDialog = false }) {
                    Text("Continue", color = GradientCyan)
                }
            }
        )
    }
    
    val categories = listOf(
        "Just Chatting", "Gaming", "Music", "Art", "Sports",
        "Cooking", "Technology", "Education", "Fitness", "Travel"
    )
    
    // Camera permission
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)
    
    
    Box(modifier = Modifier.fillMaxSize()) {
        // Camera Preview or Permission Request
        if (cameraPermissionState.status.isGranted) {
            CameraPreview(
                modifier = Modifier.fillMaxSize(),
                isFrontCamera = isFrontCamera,
                context = context,
                lifecycleOwner = lifecycleOwner
            )
        } else {
            // Permission not granted - show request
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(DarkBackground),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.Videocam,
                        contentDescription = null,
                        tint = GradientPink,
                        modifier = Modifier.size(80.dp)
                    )
                    
                    Text(
                        text = "Camera Permission Required",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                    
                    Text(
                        text = if (cameraPermissionState.status.shouldShowRationale) {
                            "Camera access is needed to go live.\nPlease grant permission to continue."
                        } else {
                            "Tap below to enable camera access\nand start streaming!"
                        },
                        style = MaterialTheme.typography.bodyMedium,
                        color = TextSecondary,
                        textAlign = TextAlign.Center
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Button(
                        onClick = { cameraPermissionState.launchPermissionRequest() },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = GradientPink
                        ),
                        shape = RoundedCornerShape(24.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Camera,
                            contentDescription = null,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Enable Camera")
                    }
                }
            }
        }
        
        // Live Chat State
        var chatMessages by remember { mutableStateOf(listOf<ChatMessage>()) }
        var commentText by remember { mutableStateOf("") }
        var floatingReactions by remember { mutableStateOf(listOf<FloatingReaction>()) }
        
        // Simulate incoming chat messages when streaming
        LaunchedEffect(isStreaming) {
            if (isStreaming) {
                val sampleMessages = listOf(
                    "🔥 This is awesome!",
                    "Hey! Just joined!",
                    "Love the stream! 💜",
                    "Can you say hi?",
                    "First time here 👋",
                    "Great content!",
                    "Where are you from?",
                    "This is so cool!",
                    "Keep it up! 🙌",
                    "Hello from India! 🇮🇳"
                )
                val sampleUsers = listOf("gamer_pro", "music_fan", "cool_viewer", "newbie_01", "fan_2024", "viewer123")
                
                while (isStreaming) {
                    delay((2000..5000).random().toLong())
                    if (chatMessages.size < 50) {
                        val newMessage = ChatMessage(
                            id = System.currentTimeMillis(),
                            username = sampleUsers.random(),
                            message = sampleMessages.random(),
                            timestamp = System.currentTimeMillis()
                        )
                        chatMessages = (chatMessages + newMessage).takeLast(30)
                    }
                }
            } else {
                chatMessages = emptyList()
            }
        }
        
        // Auto-remove old floating reactions
        LaunchedEffect(floatingReactions) {
            if (floatingReactions.isNotEmpty()) {
                delay(3000)
                floatingReactions = floatingReactions.drop(1)
            }
        }
        
        // Top Controls Overlay
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Back Button
                IconButton(
                    onClick = {
                        if (isStreaming) {
                            // Show confirmation dialog when streaming
                            showExitDialog = true
                        } else {
                            onNavigateBack()
                        }
                    },
                    modifier = Modifier
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(Color.Black.copy(alpha = 0.5f))
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = Color.White
                    )
                }
                
                // Live Indicator (when streaming)
                AnimatedVisibility(
                    visible = isStreaming,
                    enter = fadeIn() + scaleIn(),
                    exit = fadeOut() + scaleOut()
                ) {
                    LiveIndicator(viewerCount = viewerCount, duration = streamDuration)
                }
                
                // Camera Flip Button
                IconButton(
                    onClick = { isFrontCamera = !isFrontCamera },
                    modifier = Modifier
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(Color.Black.copy(alpha = 0.5f))
                ) {
                    Icon(
                        imageVector = Icons.Filled.Cameraswitch,
                        contentDescription = "Switch Camera",
                        tint = Color.White
                    )
                }
            }
        }
        
        // Floating Reactions Overlay (when streaming)
        AnimatedVisibility(
            visible = isStreaming && floatingReactions.isNotEmpty(),
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 16.dp)
        ) {
            Column(
                modifier = Modifier.height(300.dp),
                verticalArrangement = Arrangement.Bottom
            ) {
                floatingReactions.takeLast(5).forEach { reaction ->
                    FloatingReactionItem(emoji = reaction.emoji)
                }
            }
        }
        
        // Live Chat Overlay (when streaming)
        AnimatedVisibility(
            visible = isStreaming,
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(start = 16.dp, bottom = 180.dp)
                .width(280.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxWidth()
            ) {
                // Chat Messages
                Column(
                    modifier = Modifier
                        .heightIn(max = 200.dp)
                        .verticalScroll(rememberScrollState()), 
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    chatMessages.takeLast(8).forEach { message ->
                        ChatMessageItem(message = message)
                    }
                }
            }
        }
        
        // Reaction Buttons (when streaming)
        AnimatedVisibility(
            visible = isStreaming,
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 8.dp, bottom = 250.dp)
        ) {
            Column(
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                val reactions = listOf("❤️", "🔥", "😂", "👏", "🎉")
                reactions.forEach { emoji ->
                    Box(
                        modifier = Modifier
                            .size(44.dp)
                            .clip(CircleShape)
                            .background(Color.Black.copy(alpha = 0.5f))
                            .clickable {
                                floatingReactions = floatingReactions + FloatingReaction(
                                    id = System.currentTimeMillis(),
                                    emoji = emoji
                                )
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(text = emoji, style = MaterialTheme.typography.titleMedium)
                    }
                }
            }
        }
        
        // Bottom Controls
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(Color.Transparent, Color.Black.copy(alpha = 0.8f))
                    )
                )
                .padding(horizontal = 16.dp, vertical = 12.dp)
                .navigationBarsPadding(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Comment Input (when streaming)
            AnimatedVisibility(visible = isStreaming) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = commentText,
                        onValueChange = { if (it.length <= 150) commentText = it },
                        placeholder = { Text("Say something...", color = TextTertiary) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(24.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = GradientPink,
                            unfocusedBorderColor = GlassWhite,
                            focusedContainerColor = Color.Black.copy(alpha = 0.5f),
                            unfocusedContainerColor = Color.Black.copy(alpha = 0.5f),
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White
                        ),
                        singleLine = true,
                        trailingIcon = {
                            if (commentText.isNotBlank()) {
                                IconButton(
                                    onClick = {
                                        val newMessage = ChatMessage(
                                            id = System.currentTimeMillis(),
                                            username = "You",
                                            message = commentText,
                                            timestamp = System.currentTimeMillis()
                                        )
                                        chatMessages = (chatMessages + newMessage).takeLast(30)
                                        commentText = ""
                                    }
                                ) {
                                    Icon(
                                        imageVector = Icons.Filled.Send,
                                        contentDescription = "Send",
                                        tint = GradientPink
                                    )
                                }
                            }
                        }
                    )
                }
            }
            
            // Stream Settings (when not streaming)
            AnimatedVisibility(visible = !isStreaming) {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Title Input
                    OutlinedTextField(
                        value = streamTitle,
                        onValueChange = { if (it.length <= 100) streamTitle = it },
                        placeholder = { Text("Enter stream title...", color = TextTertiary) },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = GradientPink,
                            unfocusedBorderColor = GlassWhite,
                            focusedContainerColor = Color.Black.copy(alpha = 0.3f),
                            unfocusedContainerColor = Color.Black.copy(alpha = 0.3f),
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White
                        ),
                        singleLine = true,
                        leadingIcon = {
                            Icon(
                                imageVector = Icons.Outlined.Title,
                                contentDescription = null,
                                tint = TextSecondary
                            )
                        }
                    )
                    
                    // Category Selector
                    Surface(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { showCategorySheet = true },
                        shape = RoundedCornerShape(16.dp),
                        color = Color.Black.copy(alpha = 0.3f),
                        border = BorderStroke(1.dp, GlassWhite)
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(12.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.Category,
                                    contentDescription = null,
                                    tint = TextSecondary
                                )
                                Text(
                                    text = selectedCategory,
                                    color = Color.White,
                                    style = MaterialTheme.typography.bodyLarge
                                )
                            }
                            Icon(
                                imageVector = Icons.Filled.KeyboardArrowDown,
                                contentDescription = null,
                                tint = TextSecondary
                            )
                        }
                    }
                }
            }
            
            // Go Live / End Stream Button
            Box(
                modifier = Modifier
                    .size(if (isStreaming) 64.dp else 80.dp)
                    .clip(CircleShape)
                    .background(
                        if (isStreaming) {
                            Brush.linearGradient(listOf(Color.Red, Color.Red.copy(alpha = 0.8f)))
                        } else {
                            Brush.linearGradient(listOf(GradientPink, GradientOrange))
                        }
                    )
                    .clickable {
                        if (cameraPermissionState.status.isGranted) {
                            if (isStreaming) {
                                viewModel.stopStream()
                            } else {
                                viewModel.startStream(streamTitle, selectedCategory)
                            }
                        } else {
                            cameraPermissionState.launchPermissionRequest()
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                if (isStreaming) {
                    // Pulsing animation when live
                    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
                    val scale by infiniteTransition.animateFloat(
                        initialValue = 1f,
                        targetValue = 1.1f,
                        animationSpec = infiniteRepeatable(
                            animation = tween(500),
                            repeatMode = RepeatMode.Reverse
                        ),
                        label = "scale"
                    )
                    
                    Box(
                        modifier = Modifier
                            .size((24 * scale).dp)
                            .clip(RoundedCornerShape(4.dp))
                            .background(Color.White)
                    )
                } else {
                    Icon(
                        imageVector = Icons.Filled.Videocam,
                        contentDescription = "Go Live",
                        tint = Color.White,
                        modifier = Modifier.size(36.dp)
                    )
                }
            }
            
            Text(
                text = if (isStreaming) "Tap to End Stream" else "Tap to Go Live",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )
        }
        
        // Category Bottom Sheet
        if (showCategorySheet) {
            ModalBottomSheet(
                onDismissRequest = { showCategorySheet = false },
                containerColor = DarkSurface
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 32.dp)
                ) {
                    Text(
                        text = "Select Category",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary,
                        modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp)
                    )
                    
                    categories.forEach { category ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    selectedCategory = category
                                    showCategorySheet = false
                                }
                                .padding(horizontal = 24.dp, vertical = 12.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = category,
                                style = MaterialTheme.typography.bodyLarge,
                                color = if (selectedCategory == category) GradientPink else TextPrimary
                            )
                            if (selectedCategory == category) {
                                Icon(
                                    imageVector = Icons.Filled.Check,
                                    contentDescription = null,
                                    tint = GradientPink
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

// Data classes for chat
private data class ChatMessage(
    val id: Long,
    val username: String,
    val message: String,
    val timestamp: Long
)

private data class FloatingReaction(
    val id: Long,
    val emoji: String
)

@Composable
private fun ChatMessageItem(message: ChatMessage) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .background(Color.Black.copy(alpha = 0.5f))
            .padding(horizontal = 8.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Text(
            text = message.username,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Bold,
            color = if (message.username == "You") GradientPink else GradientCyan
        )
        Text(
            text = message.message,
            style = MaterialTheme.typography.labelSmall,
            color = Color.White
        )
    }
}

@Composable
private fun FloatingReactionItem(emoji: String) {
    val infiniteTransition = rememberInfiniteTransition(label = "float")
    val offsetY by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = -50f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "y"
    )
    val alpha by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 0f,
        animationSpec = infiniteRepeatable(
            animation = tween(2500),
            repeatMode = RepeatMode.Restart
        ),
        label = "alpha"
    )
    
    Text(
        text = emoji,
        style = MaterialTheme.typography.headlineMedium,
        modifier = Modifier
            .offset(y = offsetY.dp)
            .graphicsLayer(alpha = alpha)
    )
}

@Composable
private fun LiveIndicator(
    viewerCount: Int,
    duration: Long
) {
    val minutes = duration / 60
    val seconds = duration % 60
    
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(Color.Black.copy(alpha = 0.6f))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Live Badge
        Row(
            modifier = Modifier
                .clip(RoundedCornerShape(4.dp))
                .background(Color.Red)
                .padding(horizontal = 8.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Pulsing dot
            val infiniteTransition = rememberInfiniteTransition(label = "dot")
            val alpha by infiniteTransition.animateFloat(
                initialValue = 0.5f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    animation = tween(500),
                    repeatMode = RepeatMode.Reverse
                ),
                label = "alpha"
            )
            
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(Color.White.copy(alpha = alpha))
            )
            
            Text(
                text = "LIVE",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
        }
        
        // Viewer Count
        Row(
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Filled.Visibility,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(14.dp)
            )
            Text(
                text = viewerCount.toString(),
                style = MaterialTheme.typography.labelMedium,
                color = Color.White
            )
        }
        
        // Duration
        Text(
            text = String.format("%02d:%02d", minutes, seconds),
            style = MaterialTheme.typography.labelMedium,
            color = Color.White.copy(alpha = 0.8f)
        )
    }
}

@Composable
private fun CameraPreview(
    modifier: Modifier = Modifier,
    isFrontCamera: Boolean,
    context: Context,
    lifecycleOwner: LifecycleOwner
) {
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    var cameraProvider by remember { mutableStateOf<ProcessCameraProvider?>(null) }
    
    LaunchedEffect(Unit) {
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
        }, ContextCompat.getMainExecutor(context))
    }
    
    val cameraSelector = remember(isFrontCamera) {
        if (isFrontCamera) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
    }
    
    AndroidView(
        factory = { ctx ->
            PreviewView(ctx).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
        },
        modifier = modifier,
        update = { previewView ->
            cameraProvider?.let { provider ->
                try {
                    provider.unbindAll()
                    
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                    
                    provider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview
                    )
                } catch (e: Exception) {
                    Log.e("GoLiveScreen", "Camera bind failed", e)
                }
            }
        }
    )
}
