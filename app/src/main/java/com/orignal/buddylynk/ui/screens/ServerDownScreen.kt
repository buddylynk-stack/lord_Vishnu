package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * Server Down Screen - Beautiful Space-themed "Satellite Signal Lost" screen
 * Shows when the server is not responding
 */
@Composable
fun ServerDownScreen(
    onRetry: () -> Unit,
    isRetrying: Boolean = false,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition(label = "space_anim")
    val scope = rememberCoroutineScope()
    
    // Slider state
    var sliderProgress by remember { mutableStateOf(0f) }
    var isSlideComplete by remember { mutableStateOf(false) }
    val density = LocalDensity.current
    
    // Earth floating animation
    val earthFloat by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 10f,
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "earth_float"
    )
    
    // Satellite orbit animation
    val satelliteAngle by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(8000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "satellite_orbit"
    )
    
    // Red ping animation
    val pingScale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 3f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "ping_scale"
    )
    
    val pingAlpha by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 0f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "ping_alpha"
    )
    
    // Star twinkle
    val starTwinkle by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "star_twinkle"
    )
    
    // Glitch effect for 404
    val glitchOffset by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "glitch"
    )

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(
                Brush.radialGradient(
                    colors = listOf(
                        Color(0xFF0F172A),
                        Color(0xFF020617),
                        Color(0xFF000000)
                    ),
                    radius = 1500f
                )
            )
    ) {
        // Stars background
        val stars = remember { 
            List(50) { 
                Triple(
                    Random.nextFloat(),
                    Random.nextFloat(),
                    Random.nextFloat() * 2 + 1
                )
            }
        }
        Canvas(modifier = Modifier.fillMaxSize()) {
            stars.forEach { (xRatio, yRatio, radius) ->
                drawCircle(
                    color = Color.White.copy(alpha = starTwinkle * Random.nextFloat().coerceIn(0.3f, 1f)),
                    radius = radius,
                    center = Offset(xRatio * size.width, yRatio * size.height)
                )
            }
        }
        
        // Giant 404 in background
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "404",
                fontSize = 200.sp,
                fontWeight = FontWeight.Black,
                color = Color.White.copy(alpha = 0.03f),
                modifier = Modifier
                    .offset(
                        x = if (glitchOffset < 0.02f) (-2).dp else if (glitchOffset < 0.04f) 2.dp else 0.dp,
                        y = if (glitchOffset < 0.02f) 1.dp else 0.dp
                    )
            )
        }
        
        // Main Scene
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(80.dp))
            
            // Space scene container
            Box(
                modifier = Modifier.size(300.dp),
                contentAlignment = Alignment.Center
            ) {
                // Orbit rings
                Canvas(modifier = Modifier.size(280.dp)) {
                    drawCircle(
                        color = Color(0xFF1E293B).copy(alpha = 0.4f),
                        radius = size.minDimension / 2 * 1.4f,
                        style = Stroke(width = 1f)
                    )
                    drawCircle(
                        color = Color(0xFF1E293B).copy(alpha = 0.4f),
                        radius = size.minDimension / 2 * 1.8f,
                        style = Stroke(width = 1f)
                    )
                }
                
                // Moon (top right)
                Box(
                    modifier = Modifier
                        .offset(x = 80.dp, y = (-80).dp)
                        .size(70.dp)
                ) {
                    Moon()
                }
                
                // Earth (center)
                Box(
                    modifier = Modifier
                        .size(140.dp)
                        .offset(y = earthFloat.dp)
                ) {
                    Earth()
                }
                
                // Satellite orbiting
                val orbitRadius = 120.dp
                val satelliteX = cos(Math.toRadians(satelliteAngle.toDouble())).toFloat() * with(density) { orbitRadius.toPx() }
                val satelliteY = sin(Math.toRadians(satelliteAngle.toDouble())).toFloat() * with(density) { orbitRadius.toPx() } * 0.4f
                
                Box(
                    modifier = Modifier
                        .offset(
                            x = with(density) { (satelliteX / density.density).dp },
                            y = with(density) { (satelliteY / density.density).dp }
                        )
                ) {
                    // Satellite with red ping
                    Box(contentAlignment = Alignment.Center) {
                        // Red ping ring
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .scale(pingScale)
                                .alpha(pingAlpha)
                                .border(2.dp, Color(0xFFEF4444), CircleShape)
                        )
                        
                        // Satellite
                        Satellite(modifier = Modifier.rotate(satelliteAngle + 45f))
                        
                        // NO SIGNAL badge
                        Box(
                            modifier = Modifier
                                .offset(x = 25.dp, y = (-20).dp)
                                .background(Color(0xFFDC2626), RoundedCornerShape(4.dp))
                                .padding(horizontal = 4.dp, vertical = 2.dp)
                        ) {
                            Text(
                                text = "NO SIGNAL",
                                fontSize = 6.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White
                            )
                        }
                    }
                }
            }
            
            Spacer(modifier = Modifier.weight(1f))
        }
        
        // Bottom Sheet
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .clip(RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp))
                .background(Color(0xFF0F0F1A))
                .border(
                    width = 1.dp,
                    color = Color.White.copy(alpha = 0.1f),
                    shape = RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp)
                )
                .padding(horizontal = 24.dp, vertical = 28.dp)
                .navigationBarsPadding()
        ) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Drag handle
                Box(
                    modifier = Modifier
                        .width(48.dp)
                        .height(4.dp)
                        .background(Color.White.copy(alpha = 0.2f), RoundedCornerShape(2.dp))
                )
                
                Spacer(modifier = Modifier.height(24.dp))
                
                // Title
                Text(
                    text = "Satellite Left Us on Read",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Description
                Text(
                    text = "Server isn't responding. Please try again or check your connection.",
                    fontSize = 14.sp,
                    color = Color(0xFF94A3B8),
                    textAlign = TextAlign.Center,
                    lineHeight = 22.sp,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 32.dp)
                )
                
                Spacer(modifier = Modifier.height(32.dp))
                
                // Swipe to retry slider
                SwipeToRetrySlider(
                    onComplete = {
                        isSlideComplete = true
                        onRetry()
                        scope.launch {
                            delay(2000)
                            isSlideComplete = false
                        }
                    },
                    isLoading = isRetrying || isSlideComplete
                )
                
                Spacer(modifier = Modifier.height(24.dp))
                
                // Error code
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.alpha(0.3f)
                ) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .background(Color(0xFFEF4444), CircleShape)
                    )
                    Text(
                        text = "ERR_SAT_LINK_DOWN",
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Medium,
                        color = Color.White,
                        letterSpacing = 2.sp
                    )
                }
                
                Spacer(modifier = Modifier.height(16.dp))
            }
        }
    }
}

@Composable
private fun Earth() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .clip(CircleShape)
            .background(
                Brush.radialGradient(
                    colors = listOf(
                        Color(0xFF1E40AF),
                        Color(0xFF172554),
                        Color(0xFF020617)
                    ),
                    center = Offset(0.3f, 0.3f)
                )
            )
    ) {
        // Atmosphere glow
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.radialGradient(
                        colors = listOf(
                            Color.Transparent,
                            Color(0xFF3B82F6).copy(alpha = 0.2f)
                        )
                    )
                )
        )
        
        // Continents
        Box(
            modifier = Modifier
                .offset(x = 20.dp, y = 30.dp)
                .size(width = 50.dp, height = 25.dp)
                .blur(3.dp)
                .background(Color(0xFF166534).copy(alpha = 0.4f), CircleShape)
        )
        Box(
            modifier = Modifier
                .offset(x = 70.dp, y = 80.dp)
                .size(40.dp)
                .blur(3.dp)
                .background(Color(0xFF166534).copy(alpha = 0.4f), CircleShape)
        )
    }
}

@Composable
private fun Moon() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .clip(CircleShape)
            .background(
                Brush.radialGradient(
                    colors = listOf(
                        Color(0xFFE2E8F0),
                        Color(0xFF94A3B8)
                    ),
                    center = Offset(0.3f, 0.3f)
                )
            )
    ) {
        // Craters
        Box(
            modifier = Modifier
                .offset(x = 15.dp, y = 10.dp)
                .size(12.dp)
                .background(Color(0xFF94A3B8).copy(alpha = 0.5f), CircleShape)
        )
        Box(
            modifier = Modifier
                .offset(x = 40.dp, y = 35.dp)
                .size(18.dp)
                .background(Color(0xFF94A3B8).copy(alpha = 0.5f), CircleShape)
        )
        Box(
            modifier = Modifier
                .offset(x = 25.dp, y = 45.dp)
                .size(8.dp)
                .background(Color(0xFF94A3B8).copy(alpha = 0.5f), CircleShape)
        )
    }
}

@Composable
private fun Satellite(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier.size(50.dp)) {
        val centerX = size.width / 2
        val centerY = size.height / 2
        
        // Solar panels
        drawRect(
            color = Color(0xFF1E293B),
            topLeft = Offset(centerX - 35, centerY - 8),
            size = androidx.compose.ui.geometry.Size(20f, 16f)
        )
        drawRect(
            color = Color(0xFF1E293B),
            topLeft = Offset(centerX + 15, centerY - 8),
            size = androidx.compose.ui.geometry.Size(20f, 16f)
        )
        
        // Main body
        drawRect(
            color = Color(0xFFCBD5E1),
            topLeft = Offset(centerX - 10, centerY - 12),
            size = androidx.compose.ui.geometry.Size(20f, 24f)
        )
        
        // Center lens
        drawCircle(
            color = Color(0xFF3B82F6),
            radius = 6f,
            center = Offset(centerX, centerY)
        )
        drawCircle(
            color = Color.White.copy(alpha = 0.5f),
            radius = 3f,
            center = Offset(centerX, centerY)
        )
    }
}

@Composable
private fun SwipeToRetrySlider(
    onComplete: () -> Unit,
    isLoading: Boolean
) {
    var progress by remember { mutableStateOf(0f) }
    val density = LocalDensity.current
    
    val animatedProgress by animateFloatAsState(
        targetValue = if (isLoading) 1f else progress,
        animationSpec = spring(dampingRatio = 0.8f),
        label = "slider_progress"
    )
    
    // Reset progress when loading completes
    LaunchedEffect(isLoading) {
        if (!isLoading) {
            progress = 0f
        }
    }
    
    BoxWithConstraints(
        modifier = Modifier
            .fillMaxWidth()
            .height(72.dp)
            .clip(RoundedCornerShape(36.dp))
            .background(Color.White.copy(alpha = 0.08f))
            .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(36.dp))
    ) {
        val containerWidth = constraints.maxWidth
        val knobSize = 60.dp
        val knobSizePx = with(density) { knobSize.toPx() }
        val padding = 6.dp
        val paddingPx = with(density) { padding.toPx() }
        val maxSlideDistance = containerWidth - knobSizePx - (paddingPx * 2)
        
        // Fill
        Box(
            modifier = Modifier
                .fillMaxHeight()
                .fillMaxWidth(animatedProgress)
                .background(
                    Brush.horizontalGradient(
                        colors = listOf(
                            Color(0xFFEF4444).copy(alpha = 0.2f),
                            Color.White.copy(alpha = 0.2f)
                        )
                    )
                )
        )
        
        // Text
        if (!isLoading) {
            Text(
                text = "SLIDE TO RETRY",
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White.copy(alpha = 0.6f * (1f - animatedProgress)),
                letterSpacing = 1.5.sp,
                modifier = Modifier.align(Alignment.Center)
            )
        } else {
            Text(
                text = "CONNECTING...",
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White.copy(alpha = 0.6f),
                letterSpacing = 1.5.sp,
                modifier = Modifier.align(Alignment.Center)
            )
        }
        
        // Knob
        val knobOffset = with(density) { (animatedProgress * maxSlideDistance).toDp() }
        Box(
            modifier = Modifier
                .padding(padding)
                .offset(x = knobOffset)
                .size(knobSize)
                .clip(CircleShape)
                .background(Color.White)
                .pointerInput(isLoading) {
                    if (!isLoading) {
                        detectHorizontalDragGestures(
                            onDragEnd = {
                                if (progress > 0.8f) {
                                    progress = 1f
                                    onComplete()
                                } else {
                                    progress = 0f
                                }
                            },
                            onHorizontalDrag = { _, dragAmount ->
                                val delta = dragAmount / maxSlideDistance
                                progress = (progress + delta).coerceIn(0f, 1f)
                            }
                        )
                    }
                },
            contentAlignment = Alignment.Center
        ) {
            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    strokeWidth = 3.dp,
                    color = Color(0xFF991B1B)
                )
            } else {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.Send,
                    contentDescription = "Retry",
                    tint = Color(0xFF991B1B),
                    modifier = Modifier
                        .size(24.dp)
                        .rotate(-45f)
                )
            }
        }
    }
}
