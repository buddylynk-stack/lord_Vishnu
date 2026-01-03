package com.orignal.buddylynk.navigation

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.orignal.buddylynk.ui.screens.*

// =============================================================================
// NAVIGATION ROUTES
// =============================================================================

sealed class Screen(val route: String) {
    object Splash : Screen("splash")
    object Login : Screen("login")
    object Register : Screen("register")
    object Home : Screen("home")
    object Search : Screen("search")
    object Create : Screen("create")
    object Messages : Screen("messages")
    object Profile : Screen("profile")
    
    // User profile - takes userId argument
    object UserProfile : Screen("user/{userId}") {
        fun createRoute(userId: String) = "user/$userId"
    }
    
    // Chat with user
    object Chat : Screen("chat/{conversationId}") {
        fun createRoute(conversationId: String) = "chat/$conversationId"
    }
    
    // Comments screen - takes postId argument
    object Comments : Screen("comments/{postId}") {
        fun createRoute(postId: String) = "comments/$postId"
    }
    
    // Activity feed
    object Activity : Screen("activity")
    
    // Settings screens
    object Settings : Screen("settings")
    object NotificationsSettings : Screen("settings/notifications")
    object PrivacySettings : Screen("settings/privacy")
    object AppearanceSettings : Screen("settings/appearance")
    object HelpSettings : Screen("settings/help")
    
    // Edit profile
    object EditProfile : Screen("profile/edit")
    
    // Feature screens
    object TeamUp : Screen("teamup")
    object Live : Screen("live")
    object Events : Screen("events")
    object GoLive : Screen("golive")
    object ChatList : Screen("chatlist")
    object AddStory : Screen("addstory")
    object BlockedUsers : Screen("settings/blocked")
    object SavedPosts : Screen("settings/saved")
    object Shorts : Screen("shorts")
    object Call : Screen("call")
    object CreatePost : Screen("createpost")
    
    // Group/Channel creation
    object CreateGroup : Screen("creategroup/{isChannel}") {
        fun createRoute(isChannel: Boolean) = "creategroup/$isChannel"
    }
    
    // Group chat
    object GroupChat : Screen("groupchat/{groupId}") {
        fun createRoute(groupId: String) = "groupchat/$groupId"
    }
}

// =============================================================================
// NAVIGATION HOST - With premium transitions
// =============================================================================

@Composable
fun BuddyLynkNavHost(
    navController: NavHostController,
    startDestination: String = Screen.Splash.route,
    onTeamInnerViewChanged: (Boolean) -> Unit = {}
) {
    NavHost(
        navController = navController,
        startDestination = startDestination,
        enterTransition = {
            fadeIn(
                animationSpec = tween(300, easing = FastOutSlowInEasing)
            ) + slideInHorizontally(
                initialOffsetX = { it / 4 },
                animationSpec = tween(300, easing = FastOutSlowInEasing)
            )
        },
        exitTransition = {
            fadeOut(
                animationSpec = tween(200, easing = FastOutSlowInEasing)
            ) + slideOutHorizontally(
                targetOffsetX = { -it / 4 },
                animationSpec = tween(200, easing = FastOutSlowInEasing)
            )
        },
        popEnterTransition = {
            fadeIn(
                animationSpec = tween(300, easing = FastOutSlowInEasing)
            ) + slideInHorizontally(
                initialOffsetX = { -it / 4 },
                animationSpec = tween(300, easing = FastOutSlowInEasing)
            )
        },
        popExitTransition = {
            fadeOut(
                animationSpec = tween(200, easing = FastOutSlowInEasing)
            ) + slideOutHorizontally(
                targetOffsetX = { it / 4 },
                animationSpec = tween(200, easing = FastOutSlowInEasing)
            )
        }
    ) {
        composable(Screen.Splash.route) {
            SplashScreen(
                onNavigateToHome = {
                    navController.navigate(Screen.Home.route) {
                        popUpTo(Screen.Splash.route) { inclusive = true }
                    }
                },
                onNavigateToLogin = {
                    navController.navigate(Screen.Login.route) {
                        popUpTo(Screen.Splash.route) { inclusive = true }
                    }
                }
            )
        }
        
        composable(Screen.Login.route) {
            LoginScreen(
                onNavigateToHome = {
                    navController.navigate(Screen.Home.route) {
                        popUpTo(Screen.Login.route) { inclusive = true }
                    }
                },
                onNavigateToRegister = {
                    navController.navigate(Screen.Register.route)
                }
            )
        }
        
        composable(Screen.Register.route) {
            RegisterScreen(
                onNavigateToHome = {
                    navController.navigate(Screen.Home.route) {
                        popUpTo(Screen.Register.route) { inclusive = true }
                        popUpTo(Screen.Login.route) { inclusive = true }
                    }
                },
                onNavigateToLogin = {
                    navController.popBackStack()
                }
            )
        }
        
        composable(Screen.Home.route) {
            HomeScreen(
                onNavigateToProfile = {
                    navController.navigate(Screen.Profile.route)
                },
                onNavigateToSearch = {
                    navController.navigate(Screen.Search.route)
                },
                onNavigateToCreate = {
                    navController.navigate(Screen.Create.route)
                },
                onNavigateToTeamUp = {
                    navController.navigate(Screen.TeamUp.route)
                },
                onNavigateToLive = {
                    navController.navigate(Screen.Live.route)
                },
                onNavigateToEvents = {
                    navController.navigate(Screen.Events.route)
                },
                onNavigateToChat = {
                    navController.navigate(Screen.ChatList.route)
                },
                onNavigateToUserProfile = { userId ->
                    navController.navigate(Screen.UserProfile.createRoute(userId))
                },
                onNavigateToShorts = {
                    navController.navigate(Screen.Shorts.route)
                }
            )
        }
        
        composable(Screen.Search.route) {
            PremiumSearchScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToProfile = { userId ->
                    navController.navigate("user/$userId")
                }
            )
        }
        
        // Shorts screen - TikTok/Reels style video feed
        composable(Screen.Shorts.route) {
            ShortsScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToProfile = { userId -> navController.navigate(Screen.UserProfile.createRoute(userId)) },
                onNavigateToComments = { postId -> navController.navigate(Screen.Comments.createRoute(postId)) }
            )
        }
        
        // Comments screen - view and add comments to a post
        composable(
            route = Screen.Comments.route,
            arguments = listOf(
                androidx.navigation.navArgument("postId") {
                    type = androidx.navigation.NavType.StringType
                }
            )
        ) { backStackEntry ->
            val postId = backStackEntry.arguments?.getString("postId") ?: return@composable
            CommentsScreen(
                postId = postId,
                onNavigateBack = { navController.popBackStack() },
                onNavigateToProfile = { userId -> navController.navigate(Screen.UserProfile.createRoute(userId)) }
            )
        }
        
        composable(Screen.Profile.route) {
            ProfileScreen(
                onNavigateBack = { navController.popBackStack() },
                onLogout = {
                    navController.navigate(Screen.Login.route) {
                        popUpTo(0) { inclusive = true }
                    }
                },
                onNavigateToEditProfile = { navController.navigate(Screen.EditProfile.route) },
                onNavigateToSettings = { navController.navigate(Screen.Settings.route) },
                onNavigateToNotifications = { navController.navigate(Screen.NotificationsSettings.route) },
                onNavigateToPrivacy = { navController.navigate(Screen.PrivacySettings.route) },
                onNavigateToAppearance = { navController.navigate(Screen.AppearanceSettings.route) },
                onNavigateToHelp = { navController.navigate(Screen.HelpSettings.route) }
            )
        }
        
        // User Profile screen (other users)
        composable(
            route = Screen.UserProfile.route,
            arguments = listOf(
                androidx.navigation.navArgument("userId") {
                    type = androidx.navigation.NavType.StringType
                }
            )
        ) { backStackEntry ->
            val userId = backStackEntry.arguments?.getString("userId") ?: return@composable
            UserProfileScreen(
                userId = userId,
                onNavigateBack = { navController.popBackStack() },
                onNavigateToChat = { conversationId ->
                    navController.navigate(Screen.Chat.createRoute(conversationId))
                }
            )
        }
        // Settings screen (main)
        composable(Screen.Settings.route) {
            SettingsScreen(
                onNavigateBack = { navController.popBackStack() },
                onLogout = {
                    navController.navigate(Screen.Login.route) {
                        popUpTo(0) { inclusive = true }
                    }
                },
                onNavigateToNotifications = { navController.navigate(Screen.NotificationsSettings.route) },
                onNavigateToPrivacy = { navController.navigate(Screen.PrivacySettings.route) },
                onNavigateToAppearance = { navController.navigate(Screen.AppearanceSettings.route) },
                onNavigateToHelp = { navController.navigate(Screen.HelpSettings.route) },
                onNavigateToBlockedUsers = { navController.navigate(Screen.BlockedUsers.route) },
                onNavigateToSavedPosts = { navController.navigate(Screen.SavedPosts.route) }
            )
        }
        
        // Edit Profile screen
        composable(Screen.EditProfile.route) {
            EditProfileScreen(
                onNavigateBack = { navController.popBackStack() },
                onSaveSuccess = { navController.popBackStack() }
            )
        }
        
        composable(Screen.Messages.route) {
            MessagesScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        composable(Screen.Create.route) {
            CreatePostScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        // Settings screens
        composable(Screen.NotificationsSettings.route) {
            NotificationsSettingsScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        composable(Screen.PrivacySettings.route) {
            PrivacySettingsScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToBlockedUsers = { navController.navigate(Screen.BlockedUsers.route) }
            )
        }
        
        composable(Screen.AppearanceSettings.route) {
            AppearanceSettingsScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        composable(Screen.HelpSettings.route) {
            HelpScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        // Blocked Users screen
        composable(Screen.BlockedUsers.route) {
            BlockedUsersScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        // Saved Posts screen
        composable(Screen.SavedPosts.route) {
            SavedPostsScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToComments = { postId ->
                    navController.navigate(Screen.Comments.createRoute(postId))
                },
                onNavigateToProfile = { userId ->
                    navController.navigate(Screen.UserProfile.createRoute(userId))
                }
            )
        }
        
        // User Profile screen - view other users
        composable(
            route = Screen.UserProfile.route,
            arguments = listOf(
                androidx.navigation.navArgument("userId") {
                    type = androidx.navigation.NavType.StringType
                }
            )
        ) { backStackEntry ->
            val userId = backStackEntry.arguments?.getString("userId") ?: ""
            UserProfileScreen(
                userId = userId,
                onNavigateBack = { navController.popBackStack() },
                onNavigateToChat = { chatUserId ->
                    navController.navigate(Screen.Chat.createRoute(chatUserId))
                }
            )
        }
        
        // Chat screen - real-time messaging
        composable(
            route = Screen.Chat.route,
            arguments = listOf(
                androidx.navigation.navArgument("conversationId") {
                    type = androidx.navigation.NavType.StringType
                }
            )
        ) { backStackEntry ->
            val conversationId = backStackEntry.arguments?.getString("conversationId") ?: ""
            PremiumInnerChatScreen(
                conversationId = conversationId,
                onNavigateBack = { navController.popBackStack() },
                onNavigateToCall = { navController.navigate(Screen.Call.route) }
            )
        }
        
        // Activity screen
        composable(Screen.Activity.route) {
            ActivityScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToProfile = { userId ->
                    navController.navigate(Screen.UserProfile.createRoute(userId))
                },
                onNavigateToPost = { postId ->
                    // TODO: Navigate to post detail
                }
            )
        }
        
        // Edit Profile screen
        composable(Screen.EditProfile.route) {
            EditProfileScreen(
                onNavigateBack = { navController.popBackStack() },
                onSaveSuccess = { navController.popBackStack() }
            )
        }
        
        // Feature screens
        composable(Screen.TeamUp.route) {
            PremiumTeamUpScreen(
                onNavigateBack = { navController.popBackStack() },
                onInnerViewChanged = onTeamInnerViewChanged,
                onCreateGroup = { isChannel ->
                    navController.navigate(Screen.CreateGroup.createRoute(isChannel))
                }
            )
        }
        
        composable(Screen.Live.route) {
            LiveScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToGoLive = { navController.navigate(Screen.GoLive.route) }
            )
        }
        
        composable(Screen.Events.route) {
            EventsScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        // Go Live Screen
        composable(Screen.GoLive.route) {
            GoLiveScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        // Chat List Screen
        composable(Screen.ChatList.route) {
            PremiumChatListScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToChat = { userId ->
                    navController.navigate(Screen.Chat.createRoute(userId))
                },
                onCreateGroup = { isChannel ->
                    navController.navigate(Screen.CreateGroup.createRoute(isChannel))
                }
            )
        }
        
        // Add Story Screen
        composable(Screen.AddStory.route) {
            AddStoryScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        // Blocked Users Screen
        composable(Screen.BlockedUsers.route) {
            BlockedUsersScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }

        // Call Screen
        composable(Screen.Call.route) {
            CallScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }

        // Create Post Screen
        composable(Screen.CreatePost.route) {
            CreatePostScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        // Create Group/Channel Screen
        composable(
            route = Screen.CreateGroup.route,
            arguments = listOf(
                androidx.navigation.navArgument("isChannel") {
                    type = androidx.navigation.NavType.BoolType
                }
            )
        ) { backStackEntry ->
            val isChannel = backStackEntry.arguments?.getBoolean("isChannel") ?: false
            CreateGroupScreen(
                isChannel = isChannel,
                onNavigateBack = { navController.popBackStack() },
                onGroupCreated = { groupId ->
                    navController.popBackStack()
                    // Could navigate to group chat here
                }
            )
        }
    }
}

// =============================================================================
// BOTTOM NAVIGATION BAR
// =============================================================================

@Composable
fun BuddyLynkBottomBar(
    currentRoute: String?,
    onNavigate: (String) -> Unit
) {
    val items = listOf(
        NavItem(Screen.Home.route, Icons.Outlined.Home, Icons.Filled.Home, "Home"),
        NavItem(Screen.Search.route, Icons.Outlined.Search, Icons.Filled.Search, "Search"),
        NavItem(Screen.Shorts.route, Icons.Outlined.SlowMotionVideo, Icons.Filled.SlowMotionVideo, "Shorts"),
        NavItem(Screen.Messages.route, Icons.Outlined.ChatBubbleOutline, Icons.AutoMirrored.Filled.Chat, "Chat"),
        NavItem(Screen.TeamUp.route, Icons.Outlined.Groups, Icons.Filled.Groups, "TeamUp")
    )
    
    // Full width nav bar with no gap
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        Color(0xFF1A1025).copy(alpha = 0.98f),
                        Color(0xFF0D0810).copy(alpha = 0.99f)
                    )
                )
            )
    ) {
        // Top border
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(Color(0xFF6B21A8).copy(alpha = 0.3f))
        )
        
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp)
                .padding(horizontal = 8.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            items.forEach { item ->
                val isSelected = currentRoute == item.route
                
                // Regular nav item - icons only
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clickable { onNavigate(item.route) },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (isSelected) item.selectedIcon else item.icon,
                        contentDescription = item.label,
                        tint = if (isSelected) 
                            Color(0xFF00D9FF) 
                        else 
                            Color.White.copy(alpha = 0.5f),
                        modifier = Modifier.size(28.dp)
                    )
                }
            }
        }
    }
}

private data class NavItem(
    val route: String,
    val icon: ImageVector,
    val selectedIcon: ImageVector,
    val label: String
)

// =============================================================================
// SHORTS PLACEHOLDER SCREEN
// =============================================================================

@Composable
private fun ShortsPlaceholderScreen(
    onNavigateBack: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        Color(0xFF0A0A0A),
                        Color(0xFF1A1025),
                        Color(0xFF0A0A0A)
                    )
                )
            )
            .statusBarsPadding(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Animated icon
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .clip(CircleShape)
                    .background(
                        brush = Brush.linearGradient(
                            colors = listOf(
                                Color(0xFF6366F1),
                                Color(0xFF8B5CF6),
                                Color(0xFFA855F7)
                            )
                        )
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.PlayArrow,
                    contentDescription = "Shorts",
                    tint = Color.White,
                    modifier = Modifier.size(50.dp)
                )
            }
            
            Text(
                text = "Shorts",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            
            Text(
                text = "Coming Soon",
                fontSize = 16.sp,
                color = Color(0xFF00D9FF),
                fontWeight = FontWeight.Medium
            )
            
            Text(
                text = "Short videos feature is under development",
                fontSize = 14.sp,
                color = Color.White.copy(alpha = 0.6f)
            )
        }
    }
}
