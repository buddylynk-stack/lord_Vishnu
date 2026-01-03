package com.orignal.buddylynk.ui.theme

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.Color

/**
 * Theme Mode Options - Professional Social App Themes
 */
enum class ThemeMode {
    DARK,           // 🌙 Dark Mode - Default, eye-comfort
    FROSTED_GLASS,  // 🧊 Frosted Glass - Premium glassmorphism
    NATURE          // 🌄 Nature Background - Warm nature vibes
}

/**
 * Theme Manager - Singleton to manage app-wide theme
 */
object ThemeManager {
    var currentTheme by mutableStateOf(ThemeMode.DARK)
        private set
    
    fun setTheme(mode: ThemeMode) {
        currentTheme = mode
    }
    
    fun toggleTheme() {
        currentTheme = when (currentTheme) {
            ThemeMode.DARK -> ThemeMode.FROSTED_GLASS
            ThemeMode.FROSTED_GLASS -> ThemeMode.NATURE
            ThemeMode.NATURE -> ThemeMode.DARK
        }
    }
}

// =============================================================================
// THEME-SPECIFIC COLOR PALETTES
// =============================================================================

/**
 * 🌙 Dark Mode Colors - Default, rich and premium
 */
object DarkModeColors {
    val background = Color(0xFF0F0F13)
    val surface = Color(0xFF18181D)
    val surfaceVariant = Color(0xFF242429)
    val accent = Color(0xFFFF6B6B)
    val accentSecondary = Color(0xFF4ECDC4)
    val textPrimary = Color(0xFFF5F5F7)
    val textSecondary = Color(0xFF9898A0)
    val glass = Color(0x12FFFFFF)
    val glassBorder = Color(0x18FFFFFF)
}

/**
 * 🧊 Frosted Glass Mode - Premium glassmorphism with blue tint
 */
object FrostedGlassColors {
    val background = Color(0xFF0A1628)  // Deep blue-black
    val surface = Color(0x20FFFFFF)     // Transparent white
    val surfaceVariant = Color(0x30FFFFFF)
    val accent = Color(0xFF00D4FF)      // Cyan glow
    val accentSecondary = Color(0xFF7B68EE)  // Purple
    val textPrimary = Color(0xFFFFFFFF)
    val textSecondary = Color(0xFFB0C4DE)
    val glass = Color(0x25FFFFFF)       // More visible glass
    val glassBorder = Color(0x40FFFFFF)
}

/**
 * 🌄 Nature Mode - Warm, earthy tones
 */
object NatureModeColors {
    val background = Color(0xFF1A1512)  // Warm dark brown
    val surface = Color(0xFF2D2520)     // Warm surface
    val surfaceVariant = Color(0xFF3D342D)
    val accent = Color(0xFFE8A87C)      // Warm orange
    val accentSecondary = Color(0xFF85DCBA) // Mint green
    val textPrimary = Color(0xFFFAF3E0)  // Warm white
    val textSecondary = Color(0xFFC4B7A6)
    val glass = Color(0x15FFFFFF)
    val glassBorder = Color(0x20FFE4C4)  // Warm border
}

// =============================================================================
// GET CURRENT THEME COLORS
// =============================================================================

fun getThemeBackground(): Color = when (ThemeManager.currentTheme) {
    ThemeMode.DARK -> DarkModeColors.background
    ThemeMode.FROSTED_GLASS -> FrostedGlassColors.background
    ThemeMode.NATURE -> NatureModeColors.background
}

fun getThemeSurface(): Color = when (ThemeManager.currentTheme) {
    ThemeMode.DARK -> DarkModeColors.surface
    ThemeMode.FROSTED_GLASS -> FrostedGlassColors.surface
    ThemeMode.NATURE -> NatureModeColors.surface
}

fun getThemeAccent(): Color = when (ThemeManager.currentTheme) {
    ThemeMode.DARK -> DarkModeColors.accent
    ThemeMode.FROSTED_GLASS -> FrostedGlassColors.accent
    ThemeMode.NATURE -> NatureModeColors.accent
}

fun getThemeTextPrimary(): Color = when (ThemeManager.currentTheme) {
    ThemeMode.DARK -> DarkModeColors.textPrimary
    ThemeMode.FROSTED_GLASS -> FrostedGlassColors.textPrimary
    ThemeMode.NATURE -> NatureModeColors.textPrimary
}

fun getThemeTextSecondary(): Color = when (ThemeManager.currentTheme) {
    ThemeMode.DARK -> DarkModeColors.textSecondary
    ThemeMode.FROSTED_GLASS -> FrostedGlassColors.textSecondary
    ThemeMode.NATURE -> NatureModeColors.textSecondary
}

fun getThemeGlass(): Color = when (ThemeManager.currentTheme) {
    ThemeMode.DARK -> DarkModeColors.glass
    ThemeMode.FROSTED_GLASS -> FrostedGlassColors.glass
    ThemeMode.NATURE -> NatureModeColors.glass
}

fun getThemeGlassBorder(): Color = when (ThemeManager.currentTheme) {
    ThemeMode.DARK -> DarkModeColors.glassBorder
    ThemeMode.FROSTED_GLASS -> FrostedGlassColors.glassBorder
    ThemeMode.NATURE -> NatureModeColors.glassBorder
}
