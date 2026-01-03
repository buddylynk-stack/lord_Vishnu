package com.orignal.buddylynk.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import com.orignal.buddylynk.ui.theme.*

// =============================================================================
// STATIC GRADIENT BACKGROUND - Performance optimized, no animations
// =============================================================================

@Composable
fun AnimatedGradientBackground(
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit
) {
    // Static gradient for maximum performance - no animations
    Box(
        modifier = modifier.background(
            brush = Brush.verticalGradient(
                colors = listOf(
                    DarkBackground,
                    Color(0xFF0D0D1A), // Slightly lighter
                    DarkBackground
                )
            )
        )
    ) {
        // Subtle static gradient overlay for depth
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            GradientPurple.copy(alpha = 0.06f),
                            Color.Transparent
                        ),
                        center = Offset(0.3f, 0.2f),
                        radius = 1500f
                    )
                )
        )
        
        // Second subtle overlay
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            GradientCyan.copy(alpha = 0.04f),
                            Color.Transparent
                        ),
                        center = Offset(0.8f, 0.7f),
                        radius = 1200f
                    )
                )
        )
        
        content()
    }
}

// =============================================================================
// MESH GRADIENT BACKGROUND - Lightweight alternative
// =============================================================================

@Composable
fun MeshGradientBackground(
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit
) {
    Box(
        modifier = modifier.background(
            brush = Brush.verticalGradient(
                colors = listOf(
                    DarkBackground,
                    DarkSurface,
                    DarkBackground
                )
            )
        )
    ) {
        content()
    }
}

// =============================================================================
// SIMPLE DARK BACKGROUND - Maximum performance
// =============================================================================

@Composable
fun SimpleBackground(
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit
) {
    Box(
        modifier = modifier.background(DarkBackground)
    ) {
        content()
    }
}

// =============================================================================
// AURORA BACKGROUND - Static version for performance
// =============================================================================

@Composable
fun AuroraBackground(
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit
) {
    // Static aurora-like gradient - no animations for performance
    Box(
        modifier = modifier.background(
            brush = Brush.verticalGradient(
                colors = listOf(
                    DarkBackground,
                    Color(0xFF0A0A14),
                    DarkBackground
                )
            )
        )
    ) {
        // Aurora gradient overlay
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    brush = Brush.linearGradient(
                        colors = listOf(
                            GradientPurple.copy(alpha = 0.08f),
                            GradientCyan.copy(alpha = 0.05f),
                            GradientPink.copy(alpha = 0.06f)
                        ),
                        start = Offset(0f, 0f),
                        end = Offset(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY)
                    )
                )
        )
        
        content()
    }
}
