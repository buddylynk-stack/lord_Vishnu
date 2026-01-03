package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Link
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.ui.theme.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

// =============================================================================
// MODERN SPLASH SCREEN - CLEAN & PREMIUM
// =============================================================================

@Composable
fun SplashScreen(
    onNavigateToHome: () -> Unit,
    onNavigateToLogin: () -> Unit
) {
    var showLogo by remember { mutableStateOf(false) }
    var showName by remember { mutableStateOf(false) }
    var showTagline by remember { mutableStateOf(false) }
    var showRings by remember { mutableStateOf(false) }
    
    // Server health check state
    var isServerOnline by remember { mutableStateOf<Boolean?>(null) } // null = checking
    var isCheckingServer by remember { mutableStateOf(true) }
    var animationComplete by remember { mutableStateOf(false) }
    var showServerDownScreen by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    
    // Run splash animation AND check server in parallel
    LaunchedEffect(Unit) {
        // Start server check in background
        launch {
            try {
                val isHealthy = com.orignal.buddylynk.data.network.ServerHealthObserver.checkServerHealth()
                isServerOnline = isHealthy
                android.util.Log.d("SplashScreen", "Server health check: $isHealthy")
            } catch (e: Exception) {
                isServerOnline = false
                android.util.Log.e("SplashScreen", "Server health check failed: ${e.message}")
            }
            isCheckingServer = false
        }
        
        // Run splash animation
        delay(200)
        showRings = true
        delay(300)
        showLogo = true
        delay(400)
        showName = true
        delay(300)
        showTagline = true
        delay(1500)
        animationComplete = true
    }
    
    // After animation completes, decide what to show
    LaunchedEffect(animationComplete, isServerOnline) {
        if (animationComplete && isServerOnline != null) {
            if (isServerOnline == true) {
                // Server is online - navigate to home/login
                if (AuthManager.isUserLoggedIn()) {
                    onNavigateToHome()
                } else {
                    onNavigateToLogin()
                }
            } else {
                // Server is offline - show ServerDownScreen
                showServerDownScreen = true
            }
        }
    }
    
    // Show ServerDownScreen if server is offline after animation
    if (showServerDownScreen) {
        ServerDownScreen(
            onRetry = {
                // Re-check server health
                isCheckingServer = true
                showServerDownScreen = false
                isServerOnline = null
                
                // Check again
                scope.launch {
                    try {
                        val isHealthy = com.orignal.buddylynk.data.network.ServerHealthObserver.checkServerHealth()
                        isServerOnline = isHealthy
                        if (isHealthy) {
                            // Server is back - navigate
                            if (AuthManager.isUserLoggedIn()) {
                                onNavigateToHome()
                            } else {
                                onNavigateToLogin()
                            }
                        } else {
                            showServerDownScreen = true
                        }
                    } catch (e: Exception) {
                        isServerOnline = false
                        showServerDownScreen = true
                    }
                    isCheckingServer = false
                }
            },
            isRetrying = isCheckingServer
        )
        return
    }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        Color(0xFF0D0D0F),
                        Color(0xFF1A1A2E),
                        Color(0xFF0D0D0F)
                    )
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        // Animated background circles
        AnimatedBackgroundCircles(showRings)
        
        // Main content
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp)
        ) {
            // Logo
            AnimatedVisibility(
                visible = showLogo,
                enter = scaleIn(
                    initialScale = 0.3f,
                    animationSpec = spring(
                        dampingRatio = Spring.DampingRatioMediumBouncy,
                        stiffness = Spring.StiffnessLow
                    )
                ) + fadeIn(tween(500))
            ) {
                LogoWithRing()
            }
            
            Spacer(modifier = Modifier.height(32.dp))
            
            // Brand Name
            AnimatedVisibility(
                visible = showName,
                enter = fadeIn(tween(600)) + slideInVertically(
                    initialOffsetY = { 30 },
                    animationSpec = tween(600, easing = EaseOutCubic)
                )
            ) {
                BrandName()
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Tagline
            AnimatedVisibility(
                visible = showTagline,
                enter = fadeIn(tween(800)) + slideInVertically(
                    initialOffsetY = { 20 },
                    animationSpec = tween(600, easing = EaseOutCubic)
                )
            ) {
                Text(
                    text = "Share, Connect and Collab",
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextSecondary.copy(alpha = 0.8f),
                    letterSpacing = 2.sp,
                    textAlign = TextAlign.Center
                )
            }
        }
        
        // Bottom loader
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(bottom = 60.dp),
            contentAlignment = Alignment.BottomCenter
        ) {
            AnimatedVisibility(
                visible = showTagline,
                enter = fadeIn(tween(500))
            ) {
                LoadingDots()
            }
        }
    }
}

// =============================================================================
// ANIMATED BACKGROUND CIRCLES
// =============================================================================

@Composable
private fun AnimatedBackgroundCircles(visible: Boolean) {
    val infiniteTransition = rememberInfiniteTransition(label = "bg")
    
    val scale1 by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = EaseInOutSine),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale1"
    )
    
    val scale2 by infiniteTransition.animateFloat(
        initialValue = 1.1f,
        targetValue = 0.9f,
        animationSpec = infiniteRepeatable(
            animation = tween(5000, easing = EaseInOutSine),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale2"
    )
    
    val alpha by animateFloatAsState(
        targetValue = if (visible) 0.15f else 0f,
        animationSpec = tween(1000),
        label = "alpha"
    )
    
    Canvas(modifier = Modifier.fillMaxSize()) {
        val centerX = size.width / 2
        val centerY = size.height / 2
        
        // Large gradient circle - top
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    GradientPink.copy(alpha = alpha),
                    Color.Transparent
                ),
                center = Offset(centerX + 150f, centerY - 300f),
                radius = 400f * scale1
            ),
            center = Offset(centerX + 150f, centerY - 300f),
            radius = 400f * scale1
        )
        
        // Large gradient circle - bottom
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    GradientPurple.copy(alpha = alpha),
                    Color.Transparent
                ),
                center = Offset(centerX - 100f, centerY + 350f),
                radius = 350f * scale2
            ),
            center = Offset(centerX - 100f, centerY + 350f),
            radius = 350f * scale2
        )
        
        // Cyan accent
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    GradientCyan.copy(alpha = alpha * 0.7f),
                    Color.Transparent
                ),
                center = Offset(centerX - 180f, centerY - 100f),
                radius = 200f * scale1
            ),
            center = Offset(centerX - 180f, centerY - 100f),
            radius = 200f * scale1
        )
    }
}

// =============================================================================
// STATIC LOGO (PNG)
// =============================================================================

@Composable
private fun LogoWithRing() {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    
    // Subtle pulse effect
    val pulse by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.03f,
        animationSpec = infiniteRepeatable(
            animation = tween(2500, easing = EaseInOutSine),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )
    
    // Container with fixed size for proper layout
    Box(
        modifier = Modifier
            .size(160.dp)
            .scale(pulse),
        contentAlignment = Alignment.Center
    ) {
        // Static PNG logo - no animation
        com.orignal.buddylynk.ui.components.BuddyLynkLogo(
            modifier = Modifier,
            size = 150.dp
        )
    }
}

// =============================================================================
// BRAND NAME WITH GRADIENT
// =============================================================================

@Composable
private fun BrandName() {
    val infiniteTransition = rememberInfiniteTransition(label = "shimmer")
    
    val shimmer by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(3000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "shimmer"
    )
    
    Text(
        text = "Buddylynk",
        style = MaterialTheme.typography.headlineLarge.copy(
            fontSize = 36.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = (-1).sp,
            brush = Brush.linearGradient(
                colors = listOf(
                    GradientPink,
                    GradientPurple,
                    GradientCyan,
                    GradientPurple,
                    GradientPink
                ),
                start = Offset(shimmer * 400 - 100, 0f),
                end = Offset(shimmer * 400 + 200, 0f)
            )
        )
    )
}

// =============================================================================
// LOADING DOTS
// =============================================================================

@Composable
private fun LoadingDots() {
    val infiniteTransition = rememberInfiniteTransition(label = "dots")
    
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        repeat(3) { index ->
            val delay = index * 200
            
            val scale by infiniteTransition.animateFloat(
                initialValue = 0.5f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    animation = keyframes {
                        durationMillis = 1200
                        0.5f at 0
                        1f at 400
                        0.5f at 800
                        0.5f at 1200
                    },
                    repeatMode = RepeatMode.Restart,
                    initialStartOffset = StartOffset(delay)
                ),
                label = "dot$index"
            )
            
            val color = when (index) {
                0 -> GradientPink
                1 -> GradientPurple
                else -> GradientCyan
            }
            
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .scale(scale)
                    .clip(CircleShape)
                    .background(color)
            )
        }
    }
}
