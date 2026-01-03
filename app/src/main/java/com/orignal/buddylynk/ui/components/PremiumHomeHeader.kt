package com.orignal.buddylynk.ui.components

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import kotlinx.coroutines.delay

// Premium Colors
private val RoyalBlue = Color(0xFF2563EB)
private val DeepViolet = Color(0xFF7C3AED)
private val RichPink = Color(0xFFDB2777)
private val IndigoAccent = Color(0xFF6366F1)
private val DarkBg = Color(0xFF050505)
private val BorderWhite10 = Color.White.copy(alpha = 0.1f)

/**
 * Premium Header matching React design
 * Features:
 * - Animated flowing logo
 * - Gradient brand text "Buddylynk"
 * - Upload button with glow
 * - Notification bell with indicator
 * - Profile avatar with letter fallback
 */
@Composable
fun PremiumHomeHeader(
    userAvatar: String?,
    username: String = "User",
    onCreateClick: () -> Unit,
    onNotificationClick: () -> Unit = {},
    onProfileClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var showLogo by remember { mutableStateOf(true) }
    var animationKey by remember { mutableStateOf(0) }

    // Logo animation collapse after 4.8s
    LaunchedEffect(animationKey) {
        showLogo = true
        delay(4800)
        showLogo = false
    }
    
    // Smooth alpha animation for logo (no layout recalculations)
    val logoAlpha by animateFloatAsState(
        targetValue = if (showLogo) 1f else 0f,
        animationSpec = tween(durationMillis = 300, easing = FastOutSlowInEasing),
        label = "logoAlpha"
    )
    
    // Smooth logo width animation
    val logoWidth by animateDpAsState(
        targetValue = if (showLogo) 48.dp else 0.dp,
        animationSpec = tween(durationMillis = 300, easing = FastOutSlowInEasing),
        label = "logoWidth"
    )
    
    // Smooth font size animation
    val fontSize by animateFloatAsState(
        targetValue = if (showLogo) 18f else 24f,
        animationSpec = tween(durationMillis = 300, easing = FastOutSlowInEasing),
        label = "fontSize"
    )

    Row(
        modifier = modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .padding(horizontal = 20.dp, vertical = 16.dp),
            // No background - seamless with screen background
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Left: Animated Logo + Brand Name
        Row(
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Animated Logo (fades out smoothly - no layout jumps)
            if (logoWidth > 0.dp) {
                Box(
                    modifier = Modifier
                        .width(logoWidth)
                        .graphicsLayer { alpha = logoAlpha }
                ) {
                    AnimatedBuddyLynkLogo(
                        modifier = Modifier.padding(end = 8.dp),
                        size = 40.dp
                    )
                }
            }

            // Brand Name with Gradient (always visible, expands when logo hides)
            Text(
                text = "Buddylynk",
                fontSize = fontSize.sp,
                fontWeight = FontWeight.Bold,
                style = LocalTextStyle.current.copy(
                    brush = Brush.linearGradient(
                        colors = listOf(RoyalBlue, DeepViolet, RichPink)
                    )
                )
            )
        }

        // Right: Action Buttons
        Row(
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // 1. Upload Button - Unique Plus icon with animated gradient ring
            val infiniteTransition = rememberInfiniteTransition(label = "headerAnim")
            val ringRotation by infiniteTransition.animateFloat(
                initialValue = 0f,
                targetValue = 360f,
                animationSpec = infiniteRepeatable(
                    animation = tween(4000, easing = LinearEasing),
                    repeatMode = RepeatMode.Restart
                ),
                label = "ringRotation"
            )
            
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.sweepGradient(
                            colors = listOf(
                                Color(0xFF6366F1),
                                Color(0xFFEC4899),
                                Color(0xFF8B5CF6),
                                Color(0xFF06B6D4),
                                Color(0xFF6366F1)
                            )
                        )
                    )
                    .padding(2.dp)
                    .clip(CircleShape)
                    .background(DarkBg)
                    .clickable { onCreateClick() },
                contentAlignment = Alignment.Center
            ) {
                // Custom Plus icon with gradient
                Box(
                    modifier = Modifier.size(20.dp),
                    contentAlignment = Alignment.Center
                ) {
                    // Horizontal line
                    Box(
                        modifier = Modifier
                            .width(16.dp)
                            .height(2.5.dp)
                            .clip(CircleShape)
                            .background(
                                Brush.horizontalGradient(
                                    colors = listOf(
                                        Color(0xFF6366F1),
                                        Color(0xFFEC4899)
                                    )
                                )
                            )
                    )
                    // Vertical line
                    Box(
                        modifier = Modifier
                            .width(2.5.dp)
                            .height(16.dp)
                            .clip(CircleShape)
                            .background(
                                Brush.verticalGradient(
                                    colors = listOf(
                                        Color(0xFF6366F1),
                                        Color(0xFFEC4899)
                                    )
                                )
                            )
                    )
                }
            }

            // 2. Notification Bell - Custom bell with animated ring and badge
            val bellPulse by infiniteTransition.animateFloat(
                initialValue = 1f,
                targetValue = 1.1f,
                animationSpec = infiniteRepeatable(
                    animation = tween(1000, easing = FastOutSlowInEasing),
                    repeatMode = RepeatMode.Reverse
                ),
                label = "bellPulse"
            )
            
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.linearGradient(
                            colors = listOf(
                                Color(0xFF1A1A2E),
                                Color(0xFF12121C)
                            )
                        )
                    )
                    .border(
                        1.dp,
                        Brush.linearGradient(
                            colors = listOf(
                                Color.White.copy(alpha = 0.15f),
                                Color.White.copy(alpha = 0.05f)
                            )
                        ),
                        CircleShape
                    )
                    .clickable { onNotificationClick() },
                contentAlignment = Alignment.Center
            ) {
                // Custom bell icon
                Icon(
                    imageVector = Icons.Outlined.Notifications,
                    contentDescription = "Notifications",
                    tint = Color.White,
                    modifier = Modifier.size(20.dp)
                )
                
                // Simple red dot notification indicator
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .offset(x = (-6).dp, y = 6.dp)
                        .size(10.dp)
                        .clip(CircleShape)
                        .background(Color(0xFFEF4444))
                        .border(2.dp, DarkBg, CircleShape)
                )
            }

            // 3. Profile Avatar - Same style as profile page
            val letter = if (username.isNotBlank()) username.first().uppercaseChar() else 'U'
            val letterColor = when (letter) {
                'A' -> Color(0xFFE91E63); 'B' -> Color(0xFF9C27B0); 'C' -> Color(0xFF673AB7)
                'D' -> Color(0xFF3F51B5); 'E' -> Color(0xFF2196F3); 'F' -> Color(0xFF03A9F4)
                'G' -> Color(0xFF00BCD4); 'H' -> Color(0xFF009688); 'I' -> Color(0xFF4CAF50)
                'J' -> Color(0xFF8BC34A); 'K' -> Color(0xFFCDDC39); 'L' -> Color(0xFFFFEB3B)
                'M' -> Color(0xFFFFC107); 'N' -> Color(0xFFFF9800); 'O' -> Color(0xFFFF5722)
                'P' -> Color(0xFF795548); 'Q' -> Color(0xFF607D8B); 'R' -> Color(0xFFF44336)
                'S' -> Color(0xFF9C27B0); 'T' -> Color(0xFF3F51B5); 'U' -> Color(0xFF00BCD4)
                'V' -> Color(0xFF4CAF50); 'W' -> Color(0xFFFF9800); 'X' -> Color(0xFFE91E63)
                'Y' -> Color(0xFF673AB7); 'Z' -> Color(0xFF2196F3)
                else -> Color(0xFF757575)
            }
            val hasValidImage = userAvatar != null && userAvatar.isNotBlank() && userAvatar != "null"
            
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.sweepGradient(
                            colors = listOf(
                                Color(0xFF6366F1),
                                Color(0xFF8B5CF6),
                                Color(0xFFA855F7),
                                Color(0xFFFBBF24),
                                Color(0xFF6366F1)
                            )
                        )
                    )
                    .padding(2.dp)
                    .clip(CircleShape)
                    .clickable { onProfileClick() }
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(CircleShape)
                        .background(letterColor),
                    contentAlignment = Alignment.Center
                ) {
                    if (hasValidImage) {
                        AsyncImage(
                            model = userAvatar,
                            contentDescription = "Profile",
                            modifier = Modifier.fillMaxSize().clip(CircleShape),
                            contentScale = ContentScale.Crop
                        )
                    } else {
                        Text(
                            text = letter.toString(),
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }
                }
            }
        }
    }
}

/**
 * Animated Buddylynk Logo with flowing lines
 * Matches the React SVG animation
 */
@Composable
fun AnimatedBuddyLogo(
    modifier: Modifier = Modifier,
    key: Int = 0
) {
    // Animation progress (0 to 1 to 2)
    // 0-1: Draw in, 1-2: Flow out
    var animationProgress by remember { mutableStateOf(0f) }
    
    LaunchedEffect(key) {
        animationProgress = 0f
        // Animate from 0 to 2 over 5 seconds
        val startTime = System.currentTimeMillis()
        val duration = 5000L
        
        while (animationProgress < 2f) {
            val elapsed = System.currentTimeMillis() - startTime
            animationProgress = (elapsed.toFloat() / duration) * 2f
            delay(16) // ~60fps
        }
    }

    Canvas(modifier = modifier) {
        val strokeWidth = 8.dp.toPx()
        val gradient = Brush.linearGradient(
            colors = listOf(RoyalBlue, DeepViolet, RichPink)
        )

        // Calculate dash offset based on animation progress
        val totalLength = 250f
        val drawProgress = when {
            animationProgress < 1f -> animationProgress // Drawing in
            else -> 1f // Fully drawn
        }
        val eraseProgress = when {
            animationProgress > 1f -> animationProgress - 1f // Erasing
            else -> 0f
        }

        // Draw the "B" shape with animated stroke
        // This is a simplified version - adjust paths as needed
        
        // Path 1 - Left curve
        drawArc(
            brush = gradient,
            startAngle = 180f,
            sweepAngle = 180f * drawProgress * (1f - eraseProgress),
            useCenter = false,
            topLeft = Offset(size.width * 0.1f, size.height * 0.1f),
            size = androidx.compose.ui.geometry.Size(size.width * 0.4f, size.height * 0.4f),
            style = Stroke(width = strokeWidth, cap = StrokeCap.Round)
        )

        // Path 2 - Top curve  
        drawArc(
            brush = gradient,
            startAngle = 0f,
            sweepAngle = 180f * drawProgress * (1f - eraseProgress),
            useCenter = false,
            topLeft = Offset(size.width * 0.3f, size.height * 0.1f),
            size = androidx.compose.ui.geometry.Size(size.width * 0.5f, size.height * 0.4f),
            style = Stroke(width = strokeWidth, cap = StrokeCap.Round)
        )

        // Path 3 - Bottom curve
        drawArc(
            brush = gradient,
            startAngle = 0f,
            sweepAngle = 180f * drawProgress * (1f - eraseProgress),
            useCenter = false,
            topLeft = Offset(size.width * 0.3f, size.height * 0.5f),
            size = androidx.compose.ui.geometry.Size(size.width * 0.5f, size.height * 0.4f),
            style = Stroke(width = strokeWidth, cap = StrokeCap.Round)
        )
    }
}

/**
 * Story Status Full Screen Overlay
 * With progress bar and user info
 */
@Composable
fun StoryStatusOverlay(
    username: String,
    avatar: String?,
    statusImage: String,
    onClose: () -> Unit,
    modifier: Modifier = Modifier
) {
    var progress by remember { mutableStateOf(0f) }

    // Auto-progress timer
    LaunchedEffect(Unit) {
        val startTime = System.currentTimeMillis()
        val duration = 5000L // 5 seconds

        while (progress < 1f) {
            val elapsed = System.currentTimeMillis() - startTime
            progress = (elapsed.toFloat() / duration).coerceAtMost(1f)
            
            if (progress >= 1f) {
                onClose()
            }
            delay(50)
        }
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // Status Image
        AsyncImage(
            model = statusImage,
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // Progress Bar
        Row(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .padding(16.dp)
                .padding(top = 32.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .height(2.dp)
                    .clip(CircleShape)
                    .background(Color.White.copy(alpha = 0.3f))
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxHeight()
                        .fillMaxWidth(progress)
                        .background(Color.White)
                )
            }
        }

        // User Info
        Row(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(start = 16.dp, top = 56.dp, end = 16.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                AsyncImage(
                    model = avatar,
                    contentDescription = null,
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                        .border(1.dp, Color.White.copy(alpha = 0.2f), CircleShape),
                    contentScale = ContentScale.Crop
                )
                Text(
                    text = username,
                    color = Color.White,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 14.sp
                )
            }

            IconButton(
                onClick = onClose,
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(Color.Black.copy(alpha = 0.2f))
            ) {
                Icon(
                    imageVector = Icons.Filled.Close,
                    contentDescription = "Close",
                    tint = Color.White
                )
            }
        }
    }
}
