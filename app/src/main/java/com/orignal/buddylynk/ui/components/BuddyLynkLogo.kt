package com.orignal.buddylynk.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.*
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.orignal.buddylynk.R

// Premium Gradient Colors (Royal Blue → Deep Violet → Rich Pink)
private val RoyalBlue = Color(0xFF2563EB)
private val DeepViolet = Color(0xFF7C3AED)
private val RichPink = Color(0xFFDB2777)

/**
 * Animated BuddyLynk Logo - EXACT match to SVG animation
 * 
 * SVG Animation Logic:
 * - 0-40% (0-2s): Draw in - stroke-dashoffset from 250 to 0
 * - 40-60% (2-3s): Pause - fully visible
 * - 60-100% (3-5s): Flow out - stroke-dashoffset from 0 to -250
 * 
 * Paths with delays: Path1 = 0s, Path2 = 0.15s, Path3 = 0.3s
 */
@Composable
fun AnimatedBuddyLynkLogo(
    modifier: Modifier = Modifier,
    size: Dp = 120.dp
) {
    // Animation progress (0 to 1 for 5 second cycle)
    val infiniteTransition = rememberInfiniteTransition(label = "logoAnim")
    
    // Use CubicBezier easing matching SVG: cubic-bezier(0.65, 0, 0.35, 1)
    val progress by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 5000,
                easing = CubicBezierEasing(0.65f, 0f, 0.35f, 1f)
            ),
            repeatMode = RepeatMode.Restart
        ),
        label = "progress"
    )
    
    // Pre-calculate values
    val sizePx = with(androidx.compose.ui.platform.LocalDensity.current) { size.toPx() }
    val scale = sizePx / 100f  // SVG viewBox is 100x100
    val strokeWidthPx = with(androidx.compose.ui.platform.LocalDensity.current) { 
        (12.dp * (size / 120.dp)).toPx()  // Scale stroke with size
    }
    val pathLength = 250f
    
    // Pre-cache paths (exact SVG paths)
    val path1 = remember(scale) {
        Path().apply {
            // M35 30 C 25 30, 20 40, 25 50 L 35 60
            moveTo(35f * scale, 30f * scale)
            cubicTo(
                25f * scale, 30f * scale,  // control point 1
                20f * scale, 40f * scale,  // control point 2
                25f * scale, 50f * scale   // end point
            )
            lineTo(35f * scale, 60f * scale)
        }
    }
    
    val path2 = remember(scale) {
        Path().apply {
            // M35 30 L 65 30 C 75 30, 80 40, 75 50 L 65 60
            moveTo(35f * scale, 30f * scale)
            lineTo(65f * scale, 30f * scale)
            cubicTo(
                75f * scale, 30f * scale,
                80f * scale, 40f * scale,
                75f * scale, 50f * scale
            )
            lineTo(65f * scale, 60f * scale)
        }
    }
    
    val path3 = remember(scale) {
        Path().apply {
            // M25 50 L 35 60 L 65 60 C 75 60, 80 70, 75 80 C 70 90, 60 90, 50 90 L 35 90
            moveTo(25f * scale, 50f * scale)
            lineTo(35f * scale, 60f * scale)
            lineTo(65f * scale, 60f * scale)
            cubicTo(
                75f * scale, 60f * scale,
                80f * scale, 70f * scale,
                75f * scale, 80f * scale
            )
            cubicTo(
                70f * scale, 90f * scale,
                60f * scale, 90f * scale,
                50f * scale, 90f * scale
            )
            lineTo(35f * scale, 90f * scale)
        }
    }
    
    // Pre-cache gradient brush
    val gradientBrush = remember(sizePx) {
        Brush.linearGradient(
            colors = listOf(RoyalBlue, DeepViolet, RichPink),
            start = Offset.Zero,
            end = Offset(sizePx, sizePx)
        )
    }

    Canvas(modifier = modifier.size(size)) {
        /**
         * Calculate dash offset matching SVG animation exactly:
         * - 0% to 40%: offset goes from pathLength to 0 (draw in)
         * - 40% to 60%: offset stays at 0 (fully visible)
         * - 60% to 100%: offset goes from 0 to -pathLength (flow out)
         * 
         * delayRatio: 0.15s/5s = 0.03 for path2, 0.3s/5s = 0.06 for path3
         */
        fun calculateOffset(animProgress: Float, delayRatio: Float): Float {
            // Apply delay by adjusting progress
            val adjustedProgress = ((animProgress - delayRatio) / (1f - delayRatio)).coerceIn(0f, 1f)
            
            return when {
                adjustedProgress < 0.4f -> {
                    // Draw in phase: offset from pathLength to 0
                    val drawProgress = adjustedProgress / 0.4f
                    pathLength * (1f - drawProgress)
                }
                adjustedProgress < 0.6f -> {
                    // Pause phase: fully visible
                    0f
                }
                else -> {
                    // Flow out phase: offset from 0 to -pathLength
                    val outProgress = (adjustedProgress - 0.6f) / 0.4f
                    -pathLength * outProgress
                }
            }
        }
        
        val offset1 = calculateOffset(progress, 0f)      // No delay
        val offset2 = calculateOffset(progress, 0.03f)   // 0.15s delay
        val offset3 = calculateOffset(progress, 0.06f)   // 0.3s delay

        // Draw Path 1 (left curve)
        drawPath(
            path = path1,
            brush = gradientBrush,
            style = Stroke(
                width = strokeWidthPx,
                cap = StrokeCap.Round,
                join = StrokeJoin.Round,
                pathEffect = PathEffect.dashPathEffect(
                    floatArrayOf(pathLength, pathLength),
                    offset1
                )
            )
        )

        // Draw Path 2 (top right curve)
        drawPath(
            path = path2,
            brush = gradientBrush,
            style = Stroke(
                width = strokeWidthPx,
                cap = StrokeCap.Round,
                join = StrokeJoin.Round,
                pathEffect = PathEffect.dashPathEffect(
                    floatArrayOf(pathLength, pathLength),
                    offset2
                )
            )
        )

        // Draw Path 3 (bottom curve)
        drawPath(
            path = path3,
            brush = gradientBrush,
            style = Stroke(
                width = strokeWidthPx,
                cap = StrokeCap.Round,
                join = StrokeJoin.Round,
                pathEffect = PathEffect.dashPathEffect(
                    floatArrayOf(pathLength, pathLength),
                    offset3
                )
            )
        )
    }
}

/**
 * Static BuddyLynk Logo using the PNG resource
 */
@Composable
fun BuddyLynkLogo(
    modifier: Modifier = Modifier,
    size: Dp = 80.dp
) {
    Image(
        painter = painterResource(id = R.drawable.ic_buddylynk_logo),
        contentDescription = "BuddyLynk Logo",
        modifier = modifier.size(size)
    )
}
