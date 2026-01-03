package com.orignal.buddylynk.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// =============================================================================
// BUDDYLYNK THEME COLORS
// =============================================================================

data class BuddyLynkColors(
    val background: Color,
    val surface: Color,
    val surfaceVariant: Color,
    val surfaceElevated: Color,
    val border: Color,
    val primaryAccent: Color,
    val secondaryAccent: Color,
    val likeButton: Color,
    val success: Color,
    val warning: Color,
    val textPrimary: Color,
    val textSecondary: Color,
    val textTertiary: Color,
    val iconDefault: Color,
    val iconActive: Color,
    val iconMuted: Color,
    val glassOverlay: Color,
    val glassBorder: Color,
    val isDark: Boolean
)

val DarkBuddyLynkColors = BuddyLynkColors(
    background = DarkColors.Background,
    surface = DarkColors.Surface,
    surfaceVariant = DarkColors.SurfaceVariant,
    surfaceElevated = DarkColors.SurfaceElevated,
    border = DarkColors.Border,
    primaryAccent = DarkColors.PrimaryAccent,
    secondaryAccent = DarkColors.SecondaryAccent,
    likeButton = DarkColors.LikeButton,
    success = DarkColors.Success,
    warning = DarkColors.Warning,
    textPrimary = DarkColors.TextPrimary,
    textSecondary = DarkColors.TextSecondary,
    textTertiary = DarkColors.TextTertiary,
    iconDefault = DarkColors.IconDefault,
    iconActive = DarkColors.IconActive,
    iconMuted = DarkColors.IconMuted,
    glassOverlay = DarkColors.GlassOverlay,
    glassBorder = DarkColors.GlassBorder,
    isDark = true
)

val LightBuddyLynkColors = BuddyLynkColors(
    background = LightColors.Background,
    surface = LightColors.Surface,
    surfaceVariant = LightColors.SurfaceVariant,
    surfaceElevated = LightColors.SurfaceElevated,
    border = LightColors.Border,
    primaryAccent = LightColors.PrimaryAccent,
    secondaryAccent = LightColors.SecondaryAccent,
    likeButton = LightColors.LikeButton,
    success = LightColors.Success,
    warning = LightColors.Warning,
    textPrimary = LightColors.TextPrimary,
    textSecondary = LightColors.TextSecondary,
    textTertiary = LightColors.TextTertiary,
    iconDefault = LightColors.IconDefault,
    iconActive = LightColors.IconActive,
    iconMuted = LightColors.IconMuted,
    glassOverlay = LightColors.GlassOverlay,
    glassBorder = LightColors.GlassBorder,
    isDark = false
)

val LocalBuddyLynkColors = staticCompositionLocalOf { DarkBuddyLynkColors }

// =============================================================================
// MATERIAL 3 SCHEMES
// =============================================================================

private val DarkColorScheme = darkColorScheme(
    primary = PrimaryDark,
    onPrimary = OnPrimaryDark,
    primaryContainer = PrimaryContainerDark,
    onPrimaryContainer = OnPrimaryContainerDark,
    secondary = SecondaryDark,
    onSecondary = OnSecondaryDark,
    secondaryContainer = SecondaryContainerDark,
    onSecondaryContainer = OnSecondaryContainerDark,
    tertiary = TertiaryDark,
    onTertiary = OnTertiaryDark,
    tertiaryContainer = TertiaryContainerDark,
    onTertiaryContainer = OnTertiaryContainerDark,
    background = BackgroundDark,
    onBackground = OnBackgroundDark,
    surface = SurfaceDark,
    onSurface = OnSurfaceDark,
    surfaceVariant = SurfaceVariantDark,
    onSurfaceVariant = OnSurfaceVariantDark,
    error = ErrorDark,
    onError = OnErrorDark,
    errorContainer = ErrorContainerDark,
    onErrorContainer = OnErrorContainerDark,
    outline = OutlineDark,
    outlineVariant = OutlineVariantDark,
    scrim = ScrimDark
)

private val LightColorScheme = lightColorScheme(
    primary = PrimaryLight,
    onPrimary = OnPrimaryLight,
    primaryContainer = PrimaryContainerLight,
    onPrimaryContainer = OnPrimaryContainerLight,
    secondary = SecondaryLight,
    onSecondary = OnSecondaryLight,
    secondaryContainer = SecondaryContainerLight,
    onSecondaryContainer = OnSecondaryContainerLight,
    tertiary = TertiaryLight,
    onTertiary = OnTertiaryLight,
    tertiaryContainer = TertiaryContainerLight,
    onTertiaryContainer = OnTertiaryContainerLight,
    background = BackgroundLight,
    onBackground = OnBackgroundLight,
    surface = SurfaceLight,
    onSurface = OnSurfaceLight,
    surfaceVariant = SurfaceVariantLight,
    onSurfaceVariant = OnSurfaceVariantLight,
    error = ErrorLight,
    onError = OnErrorLight,
    errorContainer = ErrorContainerLight,
    onErrorContainer = OnErrorContainerLight,
    outline = OutlineLight,
    outlineVariant = OutlineVariantLight,
    scrim = ScrimLight
)

// =============================================================================
// THEME COMPOSABLE
// =============================================================================

@Composable
fun BuddylynkTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }
    
    val buddyLynkColors = if (darkTheme) DarkBuddyLynkColors else LightBuddyLynkColors
    
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = Color.Transparent.toArgb()
            window.navigationBarColor = Color.Transparent.toArgb()
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = !darkTheme
                isAppearanceLightNavigationBars = !darkTheme
            }
        }
    }

    CompositionLocalProvider(LocalBuddyLynkColors provides buddyLynkColors) {
        MaterialTheme(
            colorScheme = colorScheme,
            typography = Typography,
            content = content
        )
    }
}

object BuddyLynkTheme {
    val colors: BuddyLynkColors
        @Composable
        get() = LocalBuddyLynkColors.current
}