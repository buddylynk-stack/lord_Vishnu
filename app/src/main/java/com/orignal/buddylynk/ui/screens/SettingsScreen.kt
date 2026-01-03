package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Logout
import androidx.compose.material.icons.automirrored.outlined.Help
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.ui.components.*
import com.orignal.buddylynk.ui.theme.*

// =============================================================================
// SETTINGS SCREEN - All app settings in one place
// =============================================================================

@Composable
fun SettingsScreen(
    onNavigateBack: () -> Unit,
    onLogout: () -> Unit = {},
    onNavigateToNotifications: () -> Unit = {},
    onNavigateToPrivacy: () -> Unit = {},
    onNavigateToAppearance: () -> Unit = {},
    onNavigateToHelp: () -> Unit = {},
    onNavigateToAccount: () -> Unit = {},
    onNavigateToAbout: () -> Unit = {},
    onNavigateToBlockedUsers: () -> Unit = {},
    onNavigateToSavedPosts: () -> Unit = {}
) {
    val currentUser by AuthManager.currentUser.collectAsState()
    
    AnimatedGradientBackground(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            // Header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onNavigateBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back", tint = TextPrimary)
                }
                Text(
                    "Settings",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                Spacer(Modifier.size(48.dp))
            }
            
            // Settings content
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Account section
                Text(
                    "Account",
                    style = MaterialTheme.typography.titleSmall,
                    color = TextTertiary,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
                
                SettingsItem(
                    icon = Icons.Outlined.Person,
                    title = "Account Settings",
                    subtitle = "Manage your account",
                    iconColor = GradientPurple,
                    onClick = onNavigateToAccount
                )
                
                SettingsItem(
                    icon = Icons.Outlined.Notifications,
                    title = "Notifications",
                    subtitle = "Manage alerts & sounds",
                    iconColor = GradientPink,
                    onClick = onNavigateToNotifications
                )
                
                SettingsItem(
                    icon = Icons.Outlined.Lock,
                    title = "Privacy & Security",
                    subtitle = "Control visibility & data",
                    iconColor = GradientOrange,
                    onClick = onNavigateToPrivacy
                )
                
                SettingsItem(
                    icon = Icons.Outlined.Bookmark,
                    title = "Saved Posts",
                    subtitle = "View your saved posts",
                    iconColor = GradientCyan,
                    onClick = onNavigateToSavedPosts
                )
                
                SettingsItem(
                    icon = Icons.Outlined.Block,
                    title = "Blocked Users",
                    subtitle = "Manage blocked accounts",
                    iconColor = LikeRed,
                    onClick = onNavigateToBlockedUsers
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Preferences section
                Text(
                    "Preferences",
                    style = MaterialTheme.typography.titleSmall,
                    color = TextTertiary,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
                
                // Sensitive Content setting
                var showSensitiveDialog by remember { mutableStateOf(false) }
                val sensitiveMode by com.orignal.buddylynk.data.settings.SensitiveContentManager.contentMode.collectAsState()
                
                SettingsItem(
                    icon = Icons.Outlined.VisibilityOff,
                    title = "Sensitive Content",
                    subtitle = when (sensitiveMode) {
                        com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.SHOW -> "Show all content"
                        com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.BLUR -> "Blur sensitive content"
                        com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.HIDE -> "Hide sensitive content"
                    },
                    iconColor = GradientPink,
                    onClick = { showSensitiveDialog = true }
                )
                
                // Sensitive Content Dialog
                if (showSensitiveDialog) {
                    AlertDialog(
                        onDismissRequest = { showSensitiveDialog = false },
                        containerColor = Color(0xFF1A1A2E),
                        title = {
                            Text(
                                "Sensitive Content",
                                style = MaterialTheme.typography.titleMedium,
                                color = TextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                        },
                        text = {
                            Column(
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                // Show option
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(if (sensitiveMode == com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.SHOW) GradientPurple.copy(alpha = 0.2f) else Color.Transparent)
                                        .clickable {
                                            com.orignal.buddylynk.data.settings.SensitiveContentManager.setMode(com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.SHOW)
                                            showSensitiveDialog = false
                                        }
                                        .padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    Icon(Icons.Outlined.Visibility, null, tint = GradientMint, modifier = Modifier.size(24.dp))
                                    Column {
                                        Text("Show All", color = TextPrimary, fontWeight = FontWeight.Medium)
                                        Text("Show all content without blur", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                                    }
                                }
                                
                                // Blur option
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(if (sensitiveMode == com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.BLUR) GradientPurple.copy(alpha = 0.2f) else Color.Transparent)
                                        .clickable {
                                            com.orignal.buddylynk.data.settings.SensitiveContentManager.setMode(com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.BLUR)
                                            showSensitiveDialog = false
                                        }
                                        .padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    Icon(Icons.Outlined.BlurOn, null, tint = GradientPink, modifier = Modifier.size(24.dp))
                                    Column {
                                        Text("Blur", color = TextPrimary, fontWeight = FontWeight.Medium)
                                        Text("Blur sensitive content, tap to reveal", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                                    }
                                }
                                
                                // Hide option
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(if (sensitiveMode == com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.HIDE) GradientPurple.copy(alpha = 0.2f) else Color.Transparent)
                                        .clickable {
                                            com.orignal.buddylynk.data.settings.SensitiveContentManager.setMode(com.orignal.buddylynk.data.settings.SensitiveContentManager.ContentMode.HIDE)
                                            showSensitiveDialog = false
                                        }
                                        .padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    Icon(Icons.Outlined.VisibilityOff, null, tint = LikeRed, modifier = Modifier.size(24.dp))
                                    Column {
                                        Text("Hide", color = TextPrimary, fontWeight = FontWeight.Medium)
                                        Text("Completely hide sensitive content", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                                    }
                                }
                            }
                        },
                        confirmButton = {
                            TextButton(onClick = { showSensitiveDialog = false }) {
                                Text("Cancel", color = GradientPurple)
                            }
                        }
                    )
                }
                
                SettingsItem(
                    icon = Icons.Outlined.Palette,
                    title = "Appearance",
                    subtitle = "Theme, colors & display",
                    iconColor = GradientCyan,
                    onClick = onNavigateToAppearance
                )
                
                SettingsItem(
                    icon = Icons.Outlined.Language,
                    title = "Language",
                    subtitle = "English",
                    iconColor = GradientBlue,
                    onClick = {}
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Support section
                Text(
                    "Support",
                    style = MaterialTheme.typography.titleSmall,
                    color = TextTertiary,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
                
                SettingsItem(
                    icon = Icons.AutoMirrored.Outlined.Help,
                    title = "Help & Support",
                    subtitle = "Get assistance",
                    iconColor = GradientMint,
                    onClick = onNavigateToHelp
                )
                
                SettingsItem(
                    icon = Icons.Outlined.Info,
                    title = "About",
                    subtitle = "App info & version",
                    iconColor = GradientTeal,
                    onClick = onNavigateToAbout
                )
                
                SettingsItem(
                    icon = Icons.Outlined.Star,
                    title = "Rate App",
                    subtitle = "Leave a review",
                    iconColor = GradientOrange,
                    onClick = {}
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Logout button
                GlassCard(
                    Modifier
                        .fillMaxWidth()
                        .clickable {
                            AuthManager.logout()
                            onLogout()
                        },
                    cornerRadius = 16.dp,
                    glassOpacity = 0.08f
                ) {
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.AutoMirrored.Filled.Logout,
                            "Logout",
                            tint = LikeRed,
                            modifier = Modifier.size(22.dp)
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            "Logout",
                            style = MaterialTheme.typography.titleSmall,
                            color = LikeRed,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }
                
                // App version
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 24.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        "Buddylynk v1.0.0",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextTertiary
                    )
                }
                
                Spacer(modifier = Modifier.height(80.dp))
            }
        }
    }
}

@Composable
private fun SettingsItem(
    icon: ImageVector,
    title: String,
    subtitle: String,
    iconColor: Color = GradientPurple,
    onClick: () -> Unit
) {
    GlassCard(
        Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        cornerRadius = 16.dp,
        glassOpacity = 0.06f
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(14.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(14.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(iconColor.copy(alpha = 0.15f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(icon, null, tint = iconColor, modifier = Modifier.size(22.dp))
                }
                Column {
                    Text(
                        title,
                        style = MaterialTheme.typography.titleSmall,
                        color = TextPrimary
                    )
                    Text(
                        subtitle,
                        style = MaterialTheme.typography.bodySmall,
                        color = TextTertiary
                    )
                }
            }
            Icon(
                Icons.Filled.ChevronRight,
                null,
                tint = TextTertiary,
                modifier = Modifier.size(22.dp)
            )
        }
    }
}
