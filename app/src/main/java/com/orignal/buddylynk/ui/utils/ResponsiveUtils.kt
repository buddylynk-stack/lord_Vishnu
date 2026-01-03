package com.orignal.buddylynk.ui.utils

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Window Size Classes for responsive design
 */
enum class WindowSizeClass {
    COMPACT,    // Phone portrait (< 600dp)
    MEDIUM,     // Phone landscape / Small tablet (600-840dp)
    EXPANDED    // Tablet / Desktop (> 840dp)
}

/**
 * Device type helper
 */
enum class DeviceType {
    PHONE,
    TABLET,
    FOLDABLE
}

/**
 * Responsive dimensions based on screen size
 */
data class ResponsiveDimensions(
    // Padding
    val screenPadding: Dp,
    val cardPadding: Dp,
    val itemSpacing: Dp,
    
    // Component sizes
    val avatarSize: Dp,
    val iconSize: Dp,
    val buttonHeight: Dp,
    val cardCornerRadius: Dp,
    
    // Bottom nav
    val bottomNavHeight: Dp,
    val bottomNavIconSize: Dp,
    
    // Text sizes
    val titleSize: TextUnit,
    val bodySize: TextUnit,
    val labelSize: TextUnit,
    
    // Grid columns
    val gridColumns: Int,
    
    // Media heights
    val storySize: Dp,
    val postMediaHeight: Dp,
    val feedCardMaxWidth: Dp
)

/**
 * Responsive Screen Info
 */
data class ScreenInfo(
    val widthDp: Dp,
    val heightDp: Dp,
    val widthPx: Int,
    val heightPx: Int,
    val density: Float,
    val windowSizeClass: WindowSizeClass,
    val deviceType: DeviceType,
    val isLandscape: Boolean,
    val dimensions: ResponsiveDimensions
)

/**
 * Get current screen info with responsive dimensions
 */
@Composable
fun rememberScreenInfo(): ScreenInfo {
    val configuration = LocalConfiguration.current
    val density = LocalDensity.current
    
    return remember(configuration) {
        val widthDp = configuration.screenWidthDp.dp
        val heightDp = configuration.screenHeightDp.dp
        val widthPx = with(density) { widthDp.roundToPx() }
        val heightPx = with(density) { heightDp.roundToPx() }
        val densityValue = density.density
        val isLandscape = configuration.screenWidthDp > configuration.screenHeightDp
        
        val windowSizeClass = when {
            configuration.screenWidthDp < 600 -> WindowSizeClass.COMPACT
            configuration.screenWidthDp < 840 -> WindowSizeClass.MEDIUM
            else -> WindowSizeClass.EXPANDED
        }
        
        val deviceType = when {
            configuration.screenWidthDp >= 600 && configuration.screenHeightDp >= 600 -> DeviceType.TABLET
            configuration.smallestScreenWidthDp >= 600 -> DeviceType.FOLDABLE
            else -> DeviceType.PHONE
        }
        
        val dimensions = getResponsiveDimensions(windowSizeClass, isLandscape)
        
        ScreenInfo(
            widthDp = widthDp,
            heightDp = heightDp,
            widthPx = widthPx,
            heightPx = heightPx,
            density = densityValue,
            windowSizeClass = windowSizeClass,
            deviceType = deviceType,
            isLandscape = isLandscape,
            dimensions = dimensions
        )
    }
}

/**
 * Get dimensions based on window size class
 */
private fun getResponsiveDimensions(
    sizeClass: WindowSizeClass,
    isLandscape: Boolean
): ResponsiveDimensions {
    return when (sizeClass) {
        WindowSizeClass.COMPACT -> ResponsiveDimensions(
            screenPadding = 16.dp,
            cardPadding = 12.dp,
            itemSpacing = 12.dp,
            avatarSize = 44.dp,
            iconSize = 24.dp,
            buttonHeight = 48.dp,
            cardCornerRadius = 16.dp,
            bottomNavHeight = 72.dp,
            bottomNavIconSize = 24.dp,
            titleSize = 20.sp,
            bodySize = 14.sp,
            labelSize = 12.sp,
            gridColumns = if (isLandscape) 3 else 2,
            storySize = 68.dp,
            postMediaHeight = 200.dp,
            feedCardMaxWidth = Dp.Infinity
        )
        
        WindowSizeClass.MEDIUM -> ResponsiveDimensions(
            screenPadding = 24.dp,
            cardPadding = 16.dp,
            itemSpacing = 16.dp,
            avatarSize = 52.dp,
            iconSize = 28.dp,
            buttonHeight = 52.dp,
            cardCornerRadius = 20.dp,
            bottomNavHeight = 80.dp,
            bottomNavIconSize = 28.dp,
            titleSize = 24.sp,
            bodySize = 16.sp,
            labelSize = 14.sp,
            gridColumns = if (isLandscape) 4 else 3,
            storySize = 80.dp,
            postMediaHeight = 280.dp,
            feedCardMaxWidth = 600.dp
        )
        
        WindowSizeClass.EXPANDED -> ResponsiveDimensions(
            screenPadding = 32.dp,
            cardPadding = 20.dp,
            itemSpacing = 20.dp,
            avatarSize = 60.dp,
            iconSize = 32.dp,
            buttonHeight = 56.dp,
            cardCornerRadius = 24.dp,
            bottomNavHeight = 88.dp,
            bottomNavIconSize = 32.dp,
            titleSize = 28.sp,
            bodySize = 18.sp,
            labelSize = 16.sp,
            gridColumns = if (isLandscape) 5 else 4,
            storySize = 96.dp,
            postMediaHeight = 350.dp,
            feedCardMaxWidth = 700.dp
        )
    }
}

/**
 * Adaptive value that changes based on screen width
 */
@Composable
fun <T> adaptiveValue(
    compact: T,
    medium: T = compact,
    expanded: T = medium
): T {
    val screenInfo = rememberScreenInfo()
    return when (screenInfo.windowSizeClass) {
        WindowSizeClass.COMPACT -> compact
        WindowSizeClass.MEDIUM -> medium
        WindowSizeClass.EXPANDED -> expanded
    }
}

/**
 * Adaptive Dp based on screen size
 */
@Composable
fun adaptiveDp(
    compact: Dp,
    medium: Dp = compact,
    expanded: Dp = medium
): Dp = adaptiveValue(compact, medium, expanded)

/**
 * Adaptive TextUnit based on screen size
 */
@Composable
fun adaptiveSp(
    compact: TextUnit,
    medium: TextUnit = compact,
    expanded: TextUnit = medium
): TextUnit = adaptiveValue(compact, medium, expanded)

/**
 * Adaptive Int based on screen size
 */
@Composable
fun adaptiveInt(
    compact: Int,
    medium: Int = compact,
    expanded: Int = medium
): Int = adaptiveValue(compact, medium, expanded)

/**
 * Scale Dp based on screen width (relative to 360dp baseline)
 */
@Composable
fun Dp.scaled(): Dp {
    val configuration = LocalConfiguration.current
    val scaleFactor = (configuration.screenWidthDp / 360f).coerceIn(0.8f, 1.5f)
    return (this.value * scaleFactor).dp
}

/**
 * Scale TextUnit based on screen width
 */
@Composable
fun TextUnit.scaled(): TextUnit {
    val configuration = LocalConfiguration.current
    val scaleFactor = (configuration.screenWidthDp / 360f).coerceIn(0.85f, 1.3f)
    return (this.value * scaleFactor).sp
}
