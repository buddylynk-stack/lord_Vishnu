package com.orignal.buddylynk.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage

/**
 * Color palette for letter avatars (A-Z)
 * Each letter gets a consistent, vibrant color
 */
private val letterColors = mapOf(
    'A' to Color(0xFFE91E63), // Pink
    'B' to Color(0xFF9C27B0), // Purple
    'C' to Color(0xFF673AB7), // Deep Purple
    'D' to Color(0xFF3F51B5), // Indigo
    'E' to Color(0xFF2196F3), // Blue
    'F' to Color(0xFF03A9F4), // Light Blue
    'G' to Color(0xFF00BCD4), // Cyan
    'H' to Color(0xFF009688), // Teal
    'I' to Color(0xFF4CAF50), // Green
    'J' to Color(0xFF8BC34A), // Light Green
    'K' to Color(0xFFCDDC39), // Lime
    'L' to Color(0xFFFFEB3B), // Yellow
    'M' to Color(0xFFFFC107), // Amber
    'N' to Color(0xFFFF9800), // Orange
    'O' to Color(0xFFFF5722), // Deep Orange
    'P' to Color(0xFF795548), // Brown
    'Q' to Color(0xFF607D8B), // Blue Grey
    'R' to Color(0xFFF44336), // Red
    'S' to Color(0xFF9C27B0), // Purple
    'T' to Color(0xFF3F51B5), // Indigo
    'U' to Color(0xFF00BCD4), // Cyan
    'V' to Color(0xFF4CAF50), // Green
    'W' to Color(0xFFFF9800), // Orange
    'X' to Color(0xFFE91E63), // Pink
    'Y' to Color(0xFF673AB7), // Deep Purple
    'Z' to Color(0xFF2196F3)  // Blue
)

/**
 * Get color for a letter (defaults to grey for non-letters)
 */
private fun getColorForLetter(letter: Char): Color {
    return letterColors[letter.uppercaseChar()] ?: Color(0xFF757575)
}

/**
 * Default Avatar with Letter
 * Shows a colorful circle with the first letter of the name
 * Used when user has no profile picture
 */
@Composable
fun DefaultAvatar(
    name: String,
    modifier: Modifier = Modifier,
    size: Dp = 48.dp,
    fontSize: TextUnit = 20.sp
) {
    val firstLetter = name.firstOrNull()?.uppercaseChar() ?: '?'
    val backgroundColor = getColorForLetter(firstLetter)
    
    Box(
        modifier = modifier
            .size(size)
            .clip(CircleShape)
            .background(backgroundColor),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = firstLetter.toString(),
            color = Color.White,
            fontSize = fontSize,
            fontWeight = FontWeight.Bold
        )
    }
}

/**
 * Smart Avatar - Shows image if available, otherwise shows letter avatar
 * Use this everywhere you display user avatars
 */
@Composable
fun SmartAvatar(
    imageUrl: String?,
    name: String,
    modifier: Modifier = Modifier,
    size: Dp = 48.dp,
    fontSize: TextUnit = 20.sp
) {
    if (imageUrl.isNullOrBlank()) {
        // No image - show letter avatar
        DefaultAvatar(
            name = name,
            modifier = modifier,
            size = size,
            fontSize = fontSize
        )
    } else {
        // Has image - show it
        AsyncImage(
            model = imageUrl,
            contentDescription = "Avatar",
            modifier = modifier
                .size(size)
                .clip(CircleShape),
            contentScale = ContentScale.Crop
        )
    }
}

/**
 * Get avatar URL or null for default
 * Helper function to use in data layer
 */
fun getAvatarUrlOrDefault(avatarUrl: String?, username: String): String? {
    return if (avatarUrl.isNullOrBlank()) null else avatarUrl
}

/**
 * Get color hex code for a name (for storing in database)
 * Used to sync avatar colors across web and app
 */
fun getColorHexForName(name: String): String {
    val letter = name.firstOrNull()?.uppercaseChar() ?: 'A'
    val colorMap = mapOf(
        'A' to "#E91E63", 'B' to "#9C27B0", 'C' to "#673AB7", 'D' to "#3F51B5",
        'E' to "#2196F3", 'F' to "#03A9F4", 'G' to "#00BCD4", 'H' to "#009688",
        'I' to "#4CAF50", 'J' to "#8BC34A", 'K' to "#CDDC39", 'L' to "#FFEB3B",
        'M' to "#FFC107", 'N' to "#FF9800", 'O' to "#FF5722", 'P' to "#795548",
        'Q' to "#607D8B", 'R' to "#F44336", 'S' to "#9C27B0", 'T' to "#3F51B5",
        'U' to "#00BCD4", 'V' to "#4CAF50", 'W' to "#FF9800", 'X' to "#E91E63",
        'Y' to "#673AB7", 'Z' to "#2196F3"
    )
    return colorMap[letter] ?: "#757575"
}
