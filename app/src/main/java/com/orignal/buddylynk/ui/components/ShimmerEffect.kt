package com.orignal.buddylynk.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.orignal.buddylynk.ui.theme.*

// =============================================================================
// SHIMMER EFFECT - Premium loading animation
// =============================================================================

@Composable
fun ShimmerEffect(
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 12.dp
) {
    val shimmerColors = listOf(
        GlassWhite.copy(alpha = 0.3f),
        GlassWhiteLight.copy(alpha = 0.5f),
        GlassWhite.copy(alpha = 0.3f)
    )
    
    val transition = rememberInfiniteTransition(label = "shimmer")
    val translateAnimation by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1000f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 1200,
                easing = FastOutSlowInEasing
            ),
            repeatMode = RepeatMode.Restart
        ),
        label = "translate"
    )
    
    val brush = Brush.linearGradient(
        colors = shimmerColors,
        start = Offset(translateAnimation - 500f, translateAnimation - 500f),
        end = Offset(translateAnimation, translateAnimation)
    )
    
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(cornerRadius))
            .background(brush)
    )
}

// =============================================================================
// SHIMMER CARD - Pre-built shimmer placeholder for cards
// =============================================================================

@Composable
fun ShimmerCard(
    modifier: Modifier = Modifier
) {
    GlassCard(
        modifier = modifier.fillMaxWidth(),
        glassOpacity = 0.05f
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Avatar + name row
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                ShimmerEffect(
                    modifier = Modifier.size(48.dp),
                    cornerRadius = 24.dp
                )
                Column(
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    ShimmerEffect(
                        modifier = Modifier
                            .width(120.dp)
                            .height(14.dp)
                    )
                    ShimmerEffect(
                        modifier = Modifier
                            .width(80.dp)
                            .height(10.dp)
                    )
                }
            }
            // Content lines
            ShimmerEffect(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(12.dp)
            )
            ShimmerEffect(
                modifier = Modifier
                    .fillMaxWidth(0.8f)
                    .height(12.dp)
            )
            ShimmerEffect(
                modifier = Modifier
                    .fillMaxWidth(0.6f)
                    .height(12.dp)
            )
        }
    }
}

// =============================================================================
// SHIMMER POST - For feed loading
// =============================================================================

@Composable
fun ShimmerPost(
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            ShimmerEffect(
                modifier = Modifier.size(44.dp),
                cornerRadius = 22.dp
            )
            Column(
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                ShimmerEffect(
                    modifier = Modifier
                        .width(140.dp)
                        .height(12.dp)
                )
                ShimmerEffect(
                    modifier = Modifier
                        .width(90.dp)
                        .height(10.dp)
                )
            }
        }
        
        // Image placeholder
        ShimmerEffect(
            modifier = Modifier
                .fillMaxWidth()
                .height(280.dp),
            cornerRadius = 20.dp
        )
        
        // Action bar
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            ShimmerEffect(
                modifier = Modifier.size(28.dp),
                cornerRadius = 14.dp
            )
            ShimmerEffect(
                modifier = Modifier.size(28.dp),
                cornerRadius = 14.dp
            )
            ShimmerEffect(
                modifier = Modifier.size(28.dp),
                cornerRadius = 14.dp
            )
        }
        
        // Caption
        ShimmerEffect(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .height(10.dp)
        )
    }
}

// =============================================================================
// PULSING DOT - For online status, notifications
// =============================================================================

@Composable
fun PulsingDot(
    modifier: Modifier = Modifier,
    color: Color = StatusOnline,
    size: Dp = 10.dp
) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val scale by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulseScale"
    )
    
    val alpha by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 0.6f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulseAlpha"
    )
    
    Box(
        modifier = modifier
            .size(size * scale)
            .clip(CircleShape)
            .background(color.copy(alpha = alpha))
    )
}
