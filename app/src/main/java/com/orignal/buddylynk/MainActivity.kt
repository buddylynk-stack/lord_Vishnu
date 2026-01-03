package com.orignal.buddylynk

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.network.NetworkObserver
import com.orignal.buddylynk.navigation.BuddyLynkBottomBar
import com.orignal.buddylynk.navigation.BuddyLynkNavHost
import com.orignal.buddylynk.navigation.Screen
import com.orignal.buddylynk.ui.screens.NoConnectionScreen
import com.orignal.buddylynk.ui.theme.BuddylynkTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    
    companion object {
        // Pending deep link to be processed after login
        var pendingDeepLink: android.net.Uri? = null
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize AuthManager for session persistence
        AuthManager.init(this)
        
        // Initialize SensitiveContentManager for blur/hide preferences
        com.orignal.buddylynk.data.settings.SensitiveContentManager.init(this)
        
        // Initialize LikedPostsManager for local like persistence
        com.orignal.buddylynk.data.settings.LikedPostsManager.init(this)
        
        // Handle deep link from launch intent
        handleDeepLink(intent)
        
        enableEdgeToEdge()
        setContent {
            BuddylynkTheme(darkTheme = true) {
                BuddyLynkApp()
            }
        }
    }
    
    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleDeepLink(intent)
    }
    
    private fun handleDeepLink(intent: android.content.Intent?) {
        val data = intent?.data
        if (data != null) {
            android.util.Log.d("MainActivity", "Deep link received: $data")
            android.util.Log.d("MainActivity", "Host: ${data.host}, Path: ${data.path}")
            pendingDeepLink = data
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BuddyLynkApp() {
    val context = LocalContext.current
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route
    
    // Network connectivity state
    val isConnected by NetworkObserver.observeConnectivity(context).collectAsState(initial = true)
    var isRetrying by remember { mutableStateOf(false) }
    
    // Server status observer
    val serverStatus by com.orignal.buddylynk.data.network.ServerHealthObserver.serverStatus.collectAsState()
    val isServerOnline = serverStatus.isOnline()
    
    // Team inner view state (to hide bottom nav when inside team/channel)
    var isTeamInnerView by remember { mutableStateOf(false) }
    
    // Menu bottom sheet state
    var showMenuSheet by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    
    // Handle retry
    LaunchedEffect(isRetrying) {
        if (isRetrying) {
            delay(2000)
            isRetrying = false
        }
    }
    
    // Hide bottom bar on splash, login, register, shorts, and fullscreen screens
    val showBottomBar = currentRoute != null && 
        currentRoute != Screen.Splash.route &&
        currentRoute != Screen.Login.route &&
        currentRoute != Screen.Register.route &&
        currentRoute != Screen.GoLive.route &&
        currentRoute != Screen.Events.route &&
        currentRoute != Screen.AddStory.route &&
        currentRoute != Screen.Chat.route &&
        currentRoute != Screen.Shorts.route &&
        !isTeamInnerView
    
    Box(modifier = Modifier.fillMaxSize()) {
        // Main App Content
        Scaffold(
            modifier = Modifier.fillMaxSize(),
            containerColor = Color.Transparent,
            bottomBar = { } // No scaffold bottom bar - we use floating dock
        ) { innerPadding ->
            Box(
                modifier = Modifier.fillMaxSize()
            ) {
                BuddyLynkNavHost(
                    navController = navController,
                    startDestination = Screen.Splash.route,
                    onTeamInnerViewChanged = { isTeamInnerView = it }
                )
                
                // Floating Dock Navigation (Premium Design)
                // Hide when: not connected OR server is down
                if (showBottomBar && isConnected && isServerOnline) {
                    com.orignal.buddylynk.ui.components.FloatingDockNavigation(
                        currentRoute = currentRoute ?: "",
                        onNavigate = { route ->
                            if (route != currentRoute) {
                                navController.navigate(route) {
                                    // Pop up to home for clean navigation
                                    popUpTo(Screen.Home.route) {
                                        inclusive = (route == Screen.Home.route)
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = (route != Screen.Home.route)
                                }
                            }
                        },
                        onMenuClick = { showMenuSheet = true },
                        modifier = Modifier.align(androidx.compose.ui.Alignment.BottomCenter)
                    )
                }
            }
        }
        
        // No Connection Overlay
        AnimatedVisibility(
            visible = !isConnected,
            enter = fadeIn(),
            exit = fadeOut()
        ) {
            NoConnectionScreen(
                onRetry = { isRetrying = true },
                isRetrying = isRetrying
            )
        }
    }
    
    // Menu Bottom Sheet
    if (showMenuSheet) {
        ModalBottomSheet(
            onDismissRequest = { showMenuSheet = false },
            containerColor = Color(0xFF1A1A2E),
            contentColor = Color.White,
            dragHandle = {
                Box(
                    modifier = Modifier
                        .padding(vertical = 12.dp)
                        .width(40.dp)
                        .height(4.dp)
                        .background(Color.White.copy(alpha = 0.3f), androidx.compose.foundation.shape.RoundedCornerShape(2.dp))
                )
            }
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp)
                    .padding(bottom = 32.dp)
            ) {
                // Header with title and close button
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = androidx.compose.ui.Alignment.CenterVertically
                ) {
                    Text(
                        text = "Menu",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = Color.White
                    )
                    IconButton(onClick = { showMenuSheet = false }) {
                        Icon(
                            Icons.Outlined.Close,
                            contentDescription = "Close",
                            tint = Color.White.copy(alpha = 0.7f)
                        )
                    }
                }
                
                Spacer(Modifier.height(24.dp))
                
                // Chat Option
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            showMenuSheet = false
                            navController.navigate(Screen.Messages.route)
                        }
                        .padding(vertical = 16.dp),
                    verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        Icons.Outlined.Forum,
                        contentDescription = "Chat",
                        tint = Color(0xFF6B7280),
                        modifier = Modifier.size(24.dp)
                    )
                    Text(
                        text = "Chat",
                        fontSize = 16.sp,
                        color = Color.White
                    )
                }
                
                // Events Option
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            showMenuSheet = false
                            navController.navigate(Screen.Events.route)
                        }
                        .padding(vertical = 16.dp),
                    verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        Icons.Outlined.CalendarMonth,
                        contentDescription = "Events",
                        tint = Color(0xFF6B7280),
                        modifier = Modifier.size(24.dp)
                    )
                    Text(
                        text = "Events",
                        fontSize = 16.sp,
                        color = Color.White
                    )
                }
                
                // Settings Option
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            showMenuSheet = false
                            navController.navigate(Screen.Settings.route)
                        }
                        .padding(vertical = 16.dp),
                    verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        Icons.Outlined.Settings,
                        contentDescription = "Settings",
                        tint = Color(0xFF6B7280),
                        modifier = Modifier.size(24.dp)
                    )
                    Text(
                        text = "Settings",
                        fontSize = 16.sp,
                        color = Color.White
                    )
                }
            }
        }
    }
}