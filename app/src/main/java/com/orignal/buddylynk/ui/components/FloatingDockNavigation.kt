package com.orignal.buddylynk.ui.components

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp

// Premium Colors
private val DockBg = Color(0xFF121212).copy(alpha = 0.9f)
private val BorderWhite10 = Color.White.copy(alpha = 0.1f)
private val IndigoAccent = Color(0xFF6366F1)
private val CyanAccent = Color(0xFF00D9FF)

/**
 * Floating Dock Navigation matching React design
 * Centered floating pill with 5 nav items + menu
 */
@Composable
fun FloatingDockNavigation(
    currentRoute: String,
    onNavigate: (String) -> Unit,
    onMenuClick: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val navItems = listOf(
        DockNavItem("home", Icons.Outlined.Home, Icons.Filled.Home),
        DockNavItem("search", Icons.Outlined.Search, Icons.Filled.Search),
        DockNavItem("shorts", Icons.Outlined.Bolt, Icons.Filled.Bolt), // Zap equivalent
        DockNavItem("teamup", Icons.Outlined.Groups, Icons.Filled.Groups), // TeamUp
        DockNavItem("menu", Icons.Outlined.Menu, Icons.Filled.Menu) // Menu drawer
    )

    Box(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 32.dp)
            .padding(bottom = 24.dp)
            .navigationBarsPadding(),
        contentAlignment = Alignment.Center
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(28.dp))
                .background(DockBg)
                .border(1.dp, BorderWhite10, RoundedCornerShape(28.dp))
                .padding(horizontal = 8.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            navItems.forEach { item ->
                val isSelected = currentRoute == item.route ||
                    (currentRoute == "messages" && item.route == "chatlist")
                
                DockNavButton(
                    item = item,
                    isSelected = isSelected && item.route != "menu", // Menu never stays selected
                    onClick = { 
                        if (item.route == "menu") {
                            onMenuClick()
                        } else {
                            onNavigate(item.route)
                        }
                    }
                )
            }
        }
    }
}

@Composable
private fun DockNavButton(
    item: DockNavItem,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val scale by animateFloatAsState(
        targetValue = if (isSelected) 1f else 0.9f,
        animationSpec = spring(dampingRatio = 0.6f),
        label = "scale"
    )

    Box(
        modifier = Modifier
            .size(44.dp)
            .scale(scale)
            .clip(CircleShape)
            .background(
                if (isSelected) Color.White.copy(alpha = 0.1f) else Color.Transparent
            )
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = if (isSelected) item.selectedIcon else item.icon,
            contentDescription = item.route,
            tint = if (isSelected) Color.White else Color(0xFF71717A),
            modifier = Modifier.size(22.dp)
        )
        
        // Active indicator dot
        if (isSelected) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .offset(y = 4.dp)
                    .size(4.dp)
                    .clip(CircleShape)
                    .background(IndigoAccent)
            )
        }
    }
}

private data class DockNavItem(
    val route: String,
    val icon: ImageVector,
    val selectedIcon: ImageVector
)
