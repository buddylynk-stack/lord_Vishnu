package com.orignal.buddylynk.ui.theme

import androidx.compose.ui.graphics.Color

// =============================================================================
// BUDDYLYNK - PROFESSIONAL PREMIUM PALETTE
// Inspired by Discord, Spotify, and premium social apps
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// 🌙 DARK MODE - Deep, Rich, Premium Feel
// ─────────────────────────────────────────────────────────────────────────────
object DarkColors {
    // Core backgrounds - Rich deep tones, not pure black
    val Background = Color(0xFF0F0F13)        // Rich dark (Discord-like)
    val Surface = Color(0xFF18181D)           // Cards, containers
    val SurfaceVariant = Color(0xFF242429)    // Elevated elements
    val SurfaceElevated = Color(0xFF2C2C33)   // Modals, popups
    val Border = Color(0xFF303038)            // Subtle borders
    
    // Brand accent - Unique coral/salmon (not generic blue)
    val PrimaryAccent = Color(0xFFFF6B6B)     // Warm coral - unique!
    val SecondaryAccent = Color(0xFF4ECDC4)   // Teal complement
    
    // Interaction colors
    val LikeButton = Color(0xFFFF4757)        // Punchy red
    val Success = Color(0xFF26DE81)           // Fresh green
    val Warning = Color(0xFFFFC048)           // Warm amber
    
    // Text hierarchy - Proper contrast
    val TextPrimary = Color(0xFFF5F5F7)       // Almost white
    val TextSecondary = Color(0xFF9898A0)     // Muted gray
    val TextTertiary = Color(0xFF5C5C66)      // Subtle gray
    val TextDisabled = Color(0xFF404048)      // Very muted
    
    // Icons
    val IconDefault = Color(0xFF8888A0)       // Soft gray-purple tint
    val IconActive = Color(0xFFFF6B6B)        // Brand coral
    val IconMuted = Color(0xFF55555F)         // Inactive
    
    // Glass/Overlay
    val GlassOverlay = Color(0x12FFFFFF)      // Subtle frosted
    val GlassBorder = Color(0x18FFFFFF)       // Glass edge
    val Scrim = Color(0xCC000000)             // Modal background
}

// ─────────────────────────────────────────────────────────────────────────────
// ☀️ LIGHT MODE - Clean, Crisp, Modern
// ─────────────────────────────────────────────────────────────────────────────
object LightColors {
    // Core backgrounds - Warm whites, not cold
    val Background = Color(0xFFFAFAFC)        // Warm off-white
    val Surface = Color(0xFFFFFFFF)           // Pure white cards
    val SurfaceVariant = Color(0xFFF0F0F5)    // Subtle gray
    val SurfaceElevated = Color(0xFFFFFFFF)   // Cards
    val Border = Color(0xFFE8E8ED)            // Soft borders
    
    // Brand accent - Same coral for consistency
    val PrimaryAccent = Color(0xFFE85A5A)     // Deeper coral for light mode
    val SecondaryAccent = Color(0xFF3DBDB5)   // Darker teal
    
    // Interaction colors
    val LikeButton = Color(0xFFE84545)        // Punchy red
    val Success = Color(0xFF20C073)           // Rich green
    val Warning = Color(0xFFE6A800)           // Deep amber
    
    // Text hierarchy
    val TextPrimary = Color(0xFF1A1A1F)       // Near black
    val TextSecondary = Color(0xFF6B6B78)     // Medium gray
    val TextTertiary = Color(0xFF9898A5)      // Light gray
    val TextDisabled = Color(0xFFC5C5CD)      // Very light
    
    // Icons
    val IconDefault = Color(0xFF6E6E80)       // Neutral gray
    val IconActive = Color(0xFFE85A5A)        // Brand coral
    val IconMuted = Color(0xFFB0B0BC)         // Inactive
    
    // Glass/Overlay
    val GlassOverlay = Color(0x08000000)      // Subtle shadow
    val GlassBorder = Color(0x12000000)       // Soft edge
    val Scrim = Color(0x80000000)             // Modal background
}

// ─────────────────────────────────────────────────────────────────────────────
// GRADIENT ACCENTS - Premium feel
// ─────────────────────────────────────────────────────────────────────────────
val GradientCoral = Color(0xFFFF6B6B)
val GradientPeach = Color(0xFFFF8E72)
val GradientTeal = Color(0xFF4ECDC4)
val GradientMint = Color(0xFF7BE0D5)
val GradientPurple = Color(0xFF9B51E0)
val GradientViolet = Color(0xFFB76EF4)
val GradientBlue = Color(0xFF5B8DEF)
val GradientSky = Color(0xFF70A1FF)
val GradientPink = Color(0xFFFF6B81)
val GradientOrange = Color(0xFFFF9F43)

// Premium gradient combinations
val PremiumGradient = listOf(GradientCoral, GradientPeach)
val VibrantGradient = listOf(GradientPink, GradientCoral, GradientPeach)
val CoolGradient = listOf(GradientTeal, GradientMint)
val SunsetGradient = listOf(GradientOrange, GradientCoral, GradientPink)

// ─────────────────────────────────────────────────────────────────────────────
// LEGACY COMPATIBILITY (keep old names working)
// ─────────────────────────────────────────────────────────────────────────────
val LikeRed = DarkColors.LikeButton
val GradientNeonBlue = GradientBlue
val GradientCyan = GradientTeal
val DarkBackground = DarkColors.Background
val DarkSurface = DarkColors.Surface
val DarkSurfaceVariant = DarkColors.SurfaceVariant
val DarkCard = DarkColors.Surface

val GlassWhite = DarkColors.GlassOverlay
val GlassWhiteLight = Color(0x1AFFFFFF)
val GlassBorder = DarkColors.GlassBorder
val GlassHighlight = Color(0x30FFFFFF)
val GlassDark = Color(0x1A000000)

val TextPrimary = DarkColors.TextPrimary
val TextSecondary = DarkColors.TextSecondary
val TextTertiary = DarkColors.TextTertiary
val TextMuted = DarkColors.TextDisabled

val AccentPurple = GradientPurple
val AccentCyan = GradientTeal
val AccentPink = GradientPink
val AccentGreen = DarkColors.Success
val AccentOrange = GradientOrange

// Status indicators
val StatusOnline = Color(0xFF26DE81)
val StatusAway = Color(0xFFFFC048)
val StatusOffline = Color(0xFF5C5C66)
val StatusBusy = Color(0xFFFF4757)

// ─────────────────────────────────────────────────────────────────────────────
// MATERIAL 3 - DARK THEME
// ─────────────────────────────────────────────────────────────────────────────
val PrimaryDark = DarkColors.PrimaryAccent
val OnPrimaryDark = Color(0xFFFFFFFF)
val PrimaryContainerDark = Color(0xFF5C2626)
val OnPrimaryContainerDark = Color(0xFFFFDADA)

val SecondaryDark = DarkColors.SecondaryAccent
val OnSecondaryDark = Color(0xFF003735)
val SecondaryContainerDark = Color(0xFF00504D)
val OnSecondaryContainerDark = Color(0xFF70F7F0)

val TertiaryDark = GradientPurple
val OnTertiaryDark = Color(0xFF320056)
val TertiaryContainerDark = Color(0xFF4A007A)
val OnTertiaryContainerDark = Color(0xFFF2DAFF)

val BackgroundDark = DarkColors.Background
val OnBackgroundDark = DarkColors.TextPrimary
val SurfaceDark = DarkColors.Surface
val OnSurfaceDark = DarkColors.TextPrimary
val SurfaceVariantDark = DarkColors.SurfaceVariant
val OnSurfaceVariantDark = DarkColors.TextSecondary

val ErrorDark = Color(0xFFFFB4AB)
val OnErrorDark = Color(0xFF690005)
val ErrorContainerDark = Color(0xFF93000A)
val OnErrorContainerDark = Color(0xFFFFDAD6)

val OutlineDark = DarkColors.Border
val OutlineVariantDark = Color(0xFF404048)
val ScrimDark = Color(0xFF000000)

// ─────────────────────────────────────────────────────────────────────────────
// MATERIAL 3 - LIGHT THEME
// ─────────────────────────────────────────────────────────────────────────────
val PrimaryLight = LightColors.PrimaryAccent
val OnPrimaryLight = Color(0xFFFFFFFF)
val PrimaryContainerLight = Color(0xFFFFDADA)
val OnPrimaryContainerLight = Color(0xFF410002)

val SecondaryLight = LightColors.SecondaryAccent
val OnSecondaryLight = Color(0xFFFFFFFF)
val SecondaryContainerLight = Color(0xFFB8FFF7)
val OnSecondaryContainerLight = Color(0xFF00201E)

val TertiaryLight = GradientPurple
val OnTertiaryLight = Color(0xFFFFFFFF)
val TertiaryContainerLight = Color(0xFFF2DAFF)
val OnTertiaryContainerLight = Color(0xFF2C0050)

val BackgroundLight = LightColors.Background
val OnBackgroundLight = LightColors.TextPrimary
val SurfaceLight = LightColors.Surface
val OnSurfaceLight = LightColors.TextPrimary
val SurfaceVariantLight = LightColors.SurfaceVariant
val OnSurfaceVariantLight = LightColors.TextSecondary

val ErrorLight = Color(0xFFBA1A1A)
val OnErrorLight = Color(0xFFFFFFFF)
val ErrorContainerLight = Color(0xFFFFDAD6)
val OnErrorContainerLight = Color(0xFF410002)

val OutlineLight = LightColors.Border
val OutlineVariantLight = Color(0xFFD0D0D8)
val ScrimLight = Color(0xFF000000)