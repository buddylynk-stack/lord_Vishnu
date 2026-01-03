package com.orignal.buddylynk.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.orignal.buddylynk.ui.theme.*

// =============================================================================
// GLASS CARD - Premium Glassmorphism Component
// Unique design with animated gradient border
// =============================================================================

@Composable
fun GlassCard(
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 24.dp,
    glassOpacity: Float = 0.1f,
    borderWidth: Dp = 1.dp,
    animatedBorder: Boolean = false,
    content: @Composable BoxScope.() -> Unit
) {
    val shape = RoundedCornerShape(cornerRadius)
    
    // Animated gradient rotation for border
    val infiniteTransition = rememberInfiniteTransition(label = "border")
    val animatedAngle by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(8000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "angle"
    )
    
    val borderGradient = if (animatedBorder) {
        Brush.sweepGradient(
            0f to GradientPurple.copy(alpha = 0.6f),
            0.25f to GradientCyan.copy(alpha = 0.6f),
            0.5f to GradientPink.copy(alpha = 0.6f),
            0.75f to GradientBlue.copy(alpha = 0.6f),
            1f to GradientPurple.copy(alpha = 0.6f)
        )
    } else {
        Brush.linearGradient(
            colors = listOf(
                GlassBorder,
                GlassHighlight.copy(alpha = 0.1f)
            )
        )
    }
    
    Box(
        modifier = modifier
            .clip(shape)
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        GlassWhiteLight.copy(alpha = glassOpacity),
                        GlassWhite.copy(alpha = glassOpacity * 0.5f)
                    )
                )
            )
            .border(
                width = borderWidth,
                brush = borderGradient,
                shape = shape
            ),
        content = content
    )
}

// =============================================================================
// GLASS SURFACE - Lighter variant for nested elements
// =============================================================================

@Composable
fun GlassSurface(
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 16.dp,
    content: @Composable BoxScope.() -> Unit
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(cornerRadius))
            .background(GlassWhite),
        content = content
    )
}

// =============================================================================
// FLOATING GLASS CARD - With subtle shadow and elevation feel
// =============================================================================

@Composable
fun FloatingGlassCard(
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 28.dp,
    content: @Composable BoxScope.() -> Unit
) {
    Box(
        modifier = modifier
            .drawBehind {
                // Soft glow shadow
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            GradientPurple.copy(alpha = 0.15f),
                            Color.Transparent
                        ),
                        radius = size.maxDimension * 0.8f
                    ),
                    center = Offset(size.width / 2, size.height + 20f)
                )
            }
    ) {
        GlassCard(
            modifier = Modifier.fillMaxWidth(),
            cornerRadius = cornerRadius,
            glassOpacity = 0.12f,
            animatedBorder = true,
            content = content
        )
    }
}
