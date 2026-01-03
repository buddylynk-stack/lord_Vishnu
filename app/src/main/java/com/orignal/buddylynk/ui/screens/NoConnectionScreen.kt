package com.orignal.buddylynk.ui.screens

import android.content.Intent
import android.provider.Settings
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.WifiOff
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.orignal.buddylynk.ui.components.AnimatedGradientBackground
import com.orignal.buddylynk.ui.theme.*
import kotlinx.coroutines.delay

/**
 * No Connection Screen - Beautiful offline state
 * Responsive design that adapts to all screen sizes
 */
@Composable
fun NoConnectionScreen(
    onRetry: () -> Unit,
    isRetrying: Boolean = false,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    
    // Pulsing animation for wifi rings
    val infiniteTransition = rememberInfiniteTransition(label = "wifi_pulse")
    
    val ring1Scale by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1.3f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "ring1"
    )
    
    val ring1Alpha by infiniteTransition.animateFloat(
        initialValue = 0.5f,
        targetValue = 0f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "ring1_alpha"
    )
    
    val ring2Scale by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1.3f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = FastOutSlowInEasing, delayMillis = 500),
            repeatMode = RepeatMode.Restart
        ),
        label = "ring2"
    )
    
    val ring2Alpha by infiniteTransition.animateFloat(
        initialValue = 0.5f,
        targetValue = 0f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = FastOutSlowInEasing, delayMillis = 500),
            repeatMode = RepeatMode.Restart
        ),
        label = "ring2_alpha"
    )
    
    // Shake animation for failed retry
    var triggerShake by remember { mutableStateOf(false) }
    val shakeOffset by animateFloatAsState(
        targetValue = if (triggerShake) 1f else 0f,
        animationSpec = spring(
            dampingRatio = 0.3f,
            stiffness = 600f
        ),
        finishedListener = { triggerShake = false },
        label = "shake"
    )
    
    // Spinning icon for loading
    val spinRotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "spin"
    )
    
    AnimatedGradientBackground(modifier = modifier.fillMaxSize()) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding(),
            contentAlignment = Alignment.Center
        ) {
            Column(
                modifier = Modifier
                    .widthIn(max = 320.dp)
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Animated WiFi Icon with Pulsing Rings
                Box(
                    modifier = Modifier
                        .size(120.dp)
                        .padding(bottom = 24.dp),
                    contentAlignment = Alignment.Center
                ) {
                    // Ring 1
                    Box(
                        modifier = Modifier
                            .size(80.dp)
                            .scale(ring1Scale)
                            .graphicsLayer { alpha = ring1Alpha }
                            .border(
                                width = 2.dp,
                                color = Color.White.copy(alpha = 0.3f),
                                shape = CircleShape
                            )
                    )
                    
                    // Ring 2
                    Box(
                        modifier = Modifier
                            .size(80.dp)
                            .scale(ring2Scale)
                            .graphicsLayer { alpha = ring2Alpha }
                            .border(
                                width = 2.dp,
                                color = Color.White.copy(alpha = 0.3f),
                                shape = CircleShape
                            )
                    )
                    
                    // Center Icon
                    Box(
                        modifier = Modifier
                            .size(64.dp)
                            .clip(CircleShape)
                            .background(GlassWhite.copy(alpha = 0.1f))
                            .border(
                                width = 1.dp,
                                color = GlassBorder,
                                shape = CircleShape
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Filled.WifiOff,
                            contentDescription = "No Connection",
                            tint = Color.White,
                            modifier = Modifier.size(32.dp)
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Title
                Text(
                    text = "You're Offline",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary,
                    textAlign = TextAlign.Center
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Subtitle
                Text(
                    text = "Your feed is ghosting you. Check your signal and we'll get you back in the loop.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextSecondary,
                    textAlign = TextAlign.Center,
                    lineHeight = 22.sp,
                    modifier = Modifier.padding(horizontal = 8.dp)
                )
                
                Spacer(modifier = Modifier.height(32.dp))
                
                // Try Again Button
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .offset(x = (shakeOffset * 4).dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(Color.White)
                        .clickable(enabled = !isRetrying) { 
                            onRetry()
                        }
                        .padding(vertical = 16.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        if (isRetrying) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(20.dp),
                                strokeWidth = 2.dp,
                                color = DarkBackground
                            )
                            Text(
                                text = "Connecting...",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = DarkBackground
                            )
                        } else {
                            Text(
                                text = "Try Again",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = DarkBackground
                            )
                            Icon(
                                imageVector = Icons.Filled.Refresh,
                                contentDescription = null,
                                tint = DarkBackground,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Open Settings Button
                TextButton(
                    onClick = {
                        val intent = Intent(Settings.ACTION_WIRELESS_SETTINGS)
                        context.startActivity(intent)
                    }
                ) {
                    Text(
                        text = "Open Settings",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                        color = TextTertiary
                    )
                }
            }
        }
    }
}

/**
 * Preview / Demo composable showing NoConnectionScreen usage
 */
@Composable
fun NoConnectionScreenPreview() {
    var isRetrying by remember { mutableStateOf(false) }
    
    LaunchedEffect(isRetrying) {
        if (isRetrying) {
            delay(2000)
            isRetrying = false
        }
    }
    
    NoConnectionScreen(
        onRetry = { isRetrying = true },
        isRetrying = isRetrying
    )
}
