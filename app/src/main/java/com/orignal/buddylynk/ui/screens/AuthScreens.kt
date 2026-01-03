package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.auth.GoogleAuthService
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.data.model.User
import com.orignal.buddylynk.ui.components.*
import com.orignal.buddylynk.ui.theme.*
import kotlinx.coroutines.launch
import java.util.UUID

// =============================================================================
// PREMIUM LOGIN SCREEN
// =============================================================================

@Composable
fun LoginScreen(
    onNavigateToHome: () -> Unit,
    onNavigateToRegister: () -> Unit
) {
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var passwordVisible by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }
    var isGoogleLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    // Animation
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }
    
    Box(modifier = Modifier.fillMaxSize().background(DarkBackground)) {
        // Background Orbs
        AuthBackgroundOrbs()
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(24.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(60.dp))
            
            // Logo Section
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(600)) + slideInVertically(initialOffsetY = { -50 })
            ) {
                LogoSection()
            }
            
            Spacer(modifier = Modifier.height(12.dp))
            
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(600, delayMillis = 200))
            ) {
                Text(
                    text = "Welcome back!",
                    style = MaterialTheme.typography.titleMedium,
                    color = TextSecondary
                )
            }
            
            Spacer(modifier = Modifier.height(48.dp))
            
            // Form Fields
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(600, delayMillis = 300)) + slideInVertically(
                    initialOffsetY = { 50 },
                    animationSpec = tween(600, delayMillis = 300)
                )
            ) {
                Column {
                    // Email field
                    PremiumTextField(
                        value = email,
                        onValueChange = { email = it },
                        placeholder = "Email address",
                        leadingIcon = Icons.Outlined.Email,
                        keyboardType = KeyboardType.Email
                    )
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    // Password field
                    PremiumTextField(
                        value = password,
                        onValueChange = { password = it },
                        placeholder = "Password",
                        leadingIcon = Icons.Outlined.Lock,
                        isPassword = true,
                        passwordVisible = passwordVisible,
                        onPasswordToggle = { passwordVisible = !passwordVisible }
                    )
                    
                    // Error message
                    AnimatedVisibility(visible = errorMessage != null) {
                        Text(
                            text = errorMessage ?: "",
                            color = LikeRed,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(top = 8.dp)
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(32.dp))
                    
                    // Login button
                    PremiumButton(
                        text = if (isLoading) "Signing in..." else "Sign In",
                        onClick = {
                            if (email.isNotBlank() && password.isNotBlank()) {
                                isLoading = true
                                errorMessage = null
                                scope.launch {
                                    // Login via API using BackendRepository
                                    val result = BackendRepository.login(email.trim(), password)
                                    
                                    if (result != null) {
                                        val (user, token) = result
                                        // User found - login with JWT token
                                        AuthManager.login(user, token)
                                        isLoading = false
                                        onNavigateToHome()
                                    } else {
                                        // User not found
                                        isLoading = false
                                        errorMessage = "Invalid email or password. Please try again."
                                    }
                                }
                            } else {
                                errorMessage = "Please fill in all fields"
                            }
                        },
                        enabled = !isLoading,
                        isLoading = isLoading
                    )
                    
                    Spacer(modifier = Modifier.height(24.dp))
                    
                    // Divider
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(1.dp)
                                .background(
                                    Brush.horizontalGradient(
                                        listOf(Color.Transparent, GlassWhite)
                                    )
                                )
                        )
                        Text(
                            text = "  or  ",
                            color = TextTertiary,
                            style = MaterialTheme.typography.bodySmall
                        )
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(1.dp)
                                .background(
                                    Brush.horizontalGradient(
                                        listOf(GlassWhite, Color.Transparent)
                                    )
                                )
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(24.dp))
                    
                    // Google Sign-In Button
                    GoogleSignInButton(
                        onClick = {
                            isGoogleLoading = true
                            errorMessage = null
                            scope.launch {
                                val result = GoogleAuthService.signInWithGoogle(context)
                                isGoogleLoading = false
                                if (result.success) {
                                    onNavigateToHome()
                                } else {
                                    errorMessage = result.errorMessage
                                }
                            }
                        },
                        isLoading = isGoogleLoading,
                        enabled = !isLoading && !isGoogleLoading
                    )
                }
            }
            
            Spacer(modifier = Modifier.weight(1f))
            
            // Register link
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(600, delayMillis = 500))
            ) {
                Row(
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Don't have an account? ",
                        color = TextSecondary,
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Text(
                        text = "Sign up",
                        style = MaterialTheme.typography.bodyMedium.copy(
                            brush = Brush.horizontalGradient(PremiumGradient)
                        ),
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.clickable { onNavigateToRegister() }
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

// =============================================================================
// PREMIUM REGISTER SCREEN
// =============================================================================

@Composable
fun RegisterScreen(
    onNavigateToHome: () -> Unit,
    onNavigateToLogin: () -> Unit
) {
    var username by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var confirmPassword by remember { mutableStateOf("") }
    var passwordVisible by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }
    var isGoogleLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }
    
    Box(modifier = Modifier.fillMaxSize().background(DarkBackground)) {
        AuthBackgroundOrbs()
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(24.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(40.dp))
            
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(600)) + slideInVertically(initialOffsetY = { -50 })
            ) {
                LogoSection()
            }
            
            Spacer(modifier = Modifier.height(12.dp))
            
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(600, delayMillis = 200))
            ) {
                Text(
                    text = "Create your account",
                    style = MaterialTheme.typography.titleMedium,
                    color = TextSecondary
                )
            }
            
            Spacer(modifier = Modifier.height(32.dp))
            
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(600, delayMillis = 300)) + slideInVertically(
                    initialOffsetY = { 50 },
                    animationSpec = tween(600, delayMillis = 300)
                )
            ) {
                Column {
                    PremiumTextField(
                        value = username,
                        onValueChange = { username = it },
                        placeholder = "Username",
                        leadingIcon = Icons.Outlined.Person
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    PremiumTextField(
                        value = email,
                        onValueChange = { email = it },
                        placeholder = "Email address",
                        leadingIcon = Icons.Outlined.Email,
                        keyboardType = KeyboardType.Email
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    PremiumTextField(
                        value = password,
                        onValueChange = { password = it },
                        placeholder = "Password",
                        leadingIcon = Icons.Outlined.Lock,
                        isPassword = true,
                        passwordVisible = passwordVisible,
                        onPasswordToggle = { passwordVisible = !passwordVisible }
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    PremiumTextField(
                        value = confirmPassword,
                        onValueChange = { confirmPassword = it },
                        placeholder = "Confirm password",
                        leadingIcon = Icons.Outlined.Lock,
                        isPassword = true,
                        passwordVisible = passwordVisible,
                        onPasswordToggle = { passwordVisible = !passwordVisible }
                    )
                    
                    AnimatedVisibility(visible = errorMessage != null) {
                        Text(
                            text = errorMessage ?: "",
                            color = LikeRed,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(top = 8.dp)
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(24.dp))
                    
                    PremiumButton(
                        text = if (isLoading) "Creating account..." else "Create Account",
                        onClick = {
                            when {
                                username.isBlank() || email.isBlank() || password.isBlank() -> {
                                    errorMessage = "Please fill in all fields"
                                }
                                password != confirmPassword -> {
                                    errorMessage = "Passwords don't match"
                                }
                                password.length < 6 -> {
                                    errorMessage = "Password must be at least 6 characters"
                                }
                                else -> {
                                    isLoading = true
                                    errorMessage = null
                                    scope.launch {
                                        // Register user via API
                                        val result = BackendRepository.register(
                                            email = email.trim(),
                                            username = username.trim(),
                                            password = password
                                        )
                                        
                                        if (result != null) {
                                            val (user, token) = result
                                            // Registration successful - login with JWT token
                                            AuthManager.login(user, token)
                                            isLoading = false
                                            onNavigateToHome()
                                        } else {
                                            // Registration failed (likely email already exists)
                                            isLoading = false
                                            errorMessage = "Email already registered. Please login instead."
                                        }
                                    }
                                }
                            }
                        },
                        enabled = !isLoading && !isGoogleLoading,
                        isLoading = isLoading
                    )
                    
                    Spacer(modifier = Modifier.height(24.dp))
                    
                    // Divider
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(1.dp)
                                .background(
                                    Brush.horizontalGradient(
                                        listOf(Color.Transparent, GlassWhite)
                                    )
                                )
                        )
                        Text(
                            text = "  or  ",
                            color = TextTertiary,
                            style = MaterialTheme.typography.bodySmall
                        )
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(1.dp)
                                .background(
                                    Brush.horizontalGradient(
                                        listOf(GlassWhite, Color.Transparent)
                                    )
                                )
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(24.dp))
                    
                    // Google Sign-In Button
                    GoogleSignInButton(
                        onClick = {
                            isGoogleLoading = true
                            errorMessage = null
                            scope.launch {
                                val result = GoogleAuthService.signInWithGoogle(context)
                                isGoogleLoading = false
                                if (result.success) {
                                    onNavigateToHome()
                                } else {
                                    errorMessage = result.errorMessage
                                }
                            }
                        },
                        isLoading = isGoogleLoading,
                        enabled = !isLoading && !isGoogleLoading
                    )
                }
            }
            
            Spacer(modifier = Modifier.weight(1f))
            
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(600, delayMillis = 500))
            ) {
                Row(
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Already have an account? ",
                        color = TextSecondary,
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Text(
                        text = "Sign in",
                        style = MaterialTheme.typography.bodyMedium.copy(
                            brush = Brush.horizontalGradient(PremiumGradient)
                        ),
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.clickable { onNavigateToLogin() }
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

// =============================================================================
// PREMIUM COMPONENTS
// =============================================================================

@Composable
private fun LogoSection() {
    val infiniteTransition = rememberInfiniteTransition(label = "logo")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(10000, easing = LinearEasing)),
        label = "rotation"
    )
    
    Box(contentAlignment = Alignment.Center) {
        // Glow
        Box(
            modifier = Modifier
                .size(120.dp)
                .rotate(rotation)
                .blur(25.dp)
                .background(
                    brush = Brush.sweepGradient(
                        listOf(
                            Color(0xFF2563EB).copy(alpha = 0.4f),  // Royal Blue
                            Color.Transparent,
                            Color(0xFF7C3AED).copy(alpha = 0.4f),  // Deep Violet
                            Color.Transparent,
                            Color(0xFFDB2777).copy(alpha = 0.4f),  // Rich Pink
                            Color.Transparent
                        )
                    ),
                    shape = CircleShape
                )
        )
        
        // Animated Logo
        AnimatedBuddyLynkLogo(
            modifier = Modifier,
            size = 100.dp
        )
    }
    
    Spacer(modifier = Modifier.height(16.dp))
    
    Text(
        text = "Buddylynk",
        style = MaterialTheme.typography.headlineMedium.copy(
            fontWeight = FontWeight.Bold,
            brush = Brush.horizontalGradient(
                listOf(
                    Color(0xFF2563EB),  // Royal Blue
                    Color(0xFF7C3AED),  // Deep Violet
                    Color(0xFFDB2777)   // Rich Pink
                )
            )
        )
    )
}

@Composable
private fun PremiumTextField(
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String,
    leadingIcon: androidx.compose.ui.graphics.vector.ImageVector,
    keyboardType: KeyboardType = KeyboardType.Text,
    isPassword: Boolean = false,
    passwordVisible: Boolean = false,
    onPasswordToggle: (() -> Unit)? = null
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(GlassWhite.copy(alpha = 0.05f))
            .border(
                width = 1.dp,
                brush = Brush.linearGradient(
                    listOf(GlassWhite.copy(alpha = 0.2f), GlassWhite.copy(alpha = 0.05f))
                ),
                shape = RoundedCornerShape(16.dp)
            )
    ) {
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            placeholder = { Text(placeholder, color = TextTertiary) },
            leadingIcon = {
                Icon(leadingIcon, null, tint = TextSecondary)
            },
            trailingIcon = if (isPassword && onPasswordToggle != null) {
                {
                    IconButton(onClick = onPasswordToggle) {
                        Icon(
                            if (passwordVisible) Icons.Filled.Visibility else Icons.Filled.VisibilityOff,
                            null,
                            tint = TextSecondary
                        )
                    }
                }
            } else null,
            modifier = Modifier.fillMaxWidth(),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Color.Transparent,
                unfocusedBorderColor = Color.Transparent,
                focusedTextColor = TextPrimary,
                unfocusedTextColor = TextPrimary,
                cursorColor = GradientPink
            ),
            visualTransformation = if (isPassword && !passwordVisible) 
                PasswordVisualTransformation() else VisualTransformation.None,
            keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
            singleLine = true
        )
    }
}

@Composable
private fun PremiumButton(
    text: String,
    onClick: () -> Unit,
    enabled: Boolean = true,
    isLoading: Boolean = false
) {
    val scale by animateFloatAsState(
        targetValue = if (enabled && !isLoading) 1f else 0.98f,
        animationSpec = spring(dampingRatio = 0.8f),
        label = "scale"
    )
    
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
            .scale(scale)
            .clip(RoundedCornerShape(16.dp))
            .background(
                brush = if (enabled) {
                    Brush.linearGradient(PremiumGradient)
                } else {
                    Brush.linearGradient(listOf(GlassWhite, GlassWhiteLight))
                }
            )
            .clickable(enabled = enabled && !isLoading, onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        if (isLoading) {
            CircularProgressIndicator(
                modifier = Modifier.size(24.dp),
                color = Color.White,
                strokeWidth = 2.dp
            )
        } else {
            Text(
                text = text,
                style = MaterialTheme.typography.titleMedium,
                color = if (enabled) Color.White else TextTertiary,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@Composable
private fun SocialButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String
) {
    Box(
        modifier = Modifier
            .size(56.dp)
            .clip(CircleShape)
            .background(GlassWhite.copy(alpha = 0.08f))
            .border(
                width = 1.dp,
                color = GlassWhite.copy(alpha = 0.15f),
                shape = CircleShape
            )
            .clickable { /* Social login */ },
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = TextSecondary,
            modifier = Modifier.size(24.dp)
        )
    }
}

@Composable
private fun AuthBackgroundOrbs() {
    val infiniteTransition = rememberInfiniteTransition(label = "orbs")
    val offset by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(8000),
            repeatMode = RepeatMode.Reverse
        ),
        label = "offset"
    )
    
    // Pink orb
    Box(
        modifier = Modifier
            .fillMaxSize()
            .graphicsLayer { translationY = offset * 30f }
    ) {
        Box(
            modifier = Modifier
                .offset(x = (-80).dp, y = 100.dp)
                .size(300.dp)
                .blur(100.dp)
                .background(
                    brush = Brush.radialGradient(
                        listOf(GradientPink.copy(alpha = 0.3f), Color.Transparent)
                    ),
                    shape = CircleShape
                )
        )
    }
    
    // Purple orb
    Box(
        modifier = Modifier
            .fillMaxSize()
            .graphicsLayer { translationY = offset * -20f }
    ) {
        Box(
            modifier = Modifier
                .offset(x = 200.dp, y = 400.dp)
                .size(350.dp)
                .blur(120.dp)
                .background(
                    brush = Brush.radialGradient(
                        listOf(GradientPurple.copy(alpha = 0.25f), Color.Transparent)
                    ),
                    shape = CircleShape
                )
        )
    }
}

// =============================================================================
// GOOGLE SIGN-IN BUTTON
// =============================================================================

@Composable
private fun GoogleSignInButton(
    onClick: () -> Unit,
    isLoading: Boolean = false,
    enabled: Boolean = true
) {
    val scale by animateFloatAsState(
        targetValue = if (enabled && !isLoading) 1f else 0.98f,
        animationSpec = spring(dampingRatio = 0.8f),
        label = "scale"
    )
    
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
            .scale(scale)
            .clip(RoundedCornerShape(16.dp))
            .background(Color.White)
            .border(
                width = 1.dp,
                color = Color.LightGray.copy(alpha = 0.5f),
                shape = RoundedCornerShape(16.dp)
            )
            .clickable(enabled = enabled && !isLoading, onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    color = Color(0xFF4285F4),
                    strokeWidth = 2.dp
                )
            } else {
                // Google Logo
                GoogleLogo(modifier = Modifier.size(24.dp))
                
                Spacer(modifier = Modifier.width(12.dp))
                
                Text(
                    text = "Continue with Google",
                    style = MaterialTheme.typography.titleMedium,
                    color = Color.DarkGray,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }
}

/**
 * Google "G" Logo - Simple colored G icon
 */
@Composable
private fun GoogleLogo(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val size = size
        val width = size.width
        val height = size.height
        val strokeWidth = width * 0.18f
        val radius = (minOf(width, height) - strokeWidth) / 2
        val center = Offset(width / 2, height / 2)
        
        // Google colors
        val blue = Color(0xFF4285F4)
        val red = Color(0xFFEA4335)
        val yellow = Color(0xFFFBBC05)
        val green = Color(0xFF34A853)
        
        // Draw colored arcs to form the "G"
        // Red arc (top-left quadrant, from top going left)
        drawArc(
            color = red,
            startAngle = -135f,
            sweepAngle = 90f,
            useCenter = false,
            topLeft = Offset(center.x - radius, center.y - radius),
            size = Size(radius * 2, radius * 2),
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = strokeWidth)
        )
        
        // Yellow arc (bottom-left quadrant)
        drawArc(
            color = yellow,
            startAngle = 135f,
            sweepAngle = 45f,
            useCenter = false,
            topLeft = Offset(center.x - radius, center.y - radius),
            size = Size(radius * 2, radius * 2),
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = strokeWidth)
        )
        
        // Green arc (bottom quadrant)
        drawArc(
            color = green,
            startAngle = 90f,
            sweepAngle = 45f,
            useCenter = false,
            topLeft = Offset(center.x - radius, center.y - radius),
            size = Size(radius * 2, radius * 2),
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = strokeWidth)
        )
        
        // Blue arc (right side - from bottom-right going up to top-right)
        drawArc(
            color = blue,
            startAngle = -45f,
            sweepAngle = 90f,
            useCenter = false,
            topLeft = Offset(center.x - radius, center.y - radius),
            size = Size(radius * 2, radius * 2),
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = strokeWidth)
        )
        
        // Blue horizontal bar (the crossbar of G)
        drawLine(
            color = blue,
            start = Offset(center.x, center.y),
            end = Offset(center.x + radius, center.y),
            strokeWidth = strokeWidth
        )
    }
}
