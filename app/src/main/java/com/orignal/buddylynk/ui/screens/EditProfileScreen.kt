package com.orignal.buddylynk.ui.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.aws.DynamoDbService
import com.orignal.buddylynk.data.aws.S3Service
import com.orignal.buddylynk.ui.theme.*
import kotlinx.coroutines.launch

/**
 * EditProfileScreen - Futuristic edit profile UI
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EditProfileScreen(
    onNavigateBack: () -> Unit,
    onSaveSuccess: () -> Unit = {}
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val currentUser = AuthManager.currentUser.collectAsState().value
    
    // Form state
    var username by remember { mutableStateOf(TextFieldValue(currentUser?.username ?: "")) }
    var bio by remember { mutableStateOf(TextFieldValue(currentUser?.bio ?: "")) }
    var website by remember { mutableStateOf(TextFieldValue(currentUser?.website ?: "")) }
    var location by remember { mutableStateOf(TextFieldValue(currentUser?.location ?: "")) }
    var selectedAvatarUri by remember { mutableStateOf<Uri?>(null) }
    var selectedBannerUri by remember { mutableStateOf<Uri?>(null) }
    
    // Loading state
    var isSaving by remember { mutableStateOf(false) }
    var showSuccess by remember { mutableStateOf(false) }
    
    // Image pickers
    val avatarPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri -> selectedAvatarUri = uri }
    
    val bannerPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri -> selectedBannerUri = uri }
    
    // Calculate profile strength
    val profileStrength = remember(username, bio, currentUser?.avatar, selectedAvatarUri) {
        var strength = 0
        if (username.text.isNotBlank()) strength += 25
        if (bio.text.isNotBlank()) strength += 25
        if (currentUser?.avatar != null || selectedAvatarUri != null) strength += 25
        if (currentUser?.banner != null || selectedBannerUri != null) strength += 25
        strength
    }
    
    // Has changes
    val hasChanges = remember(username, bio, website, location, selectedAvatarUri, selectedBannerUri) {
        username.text != (currentUser?.username ?: "") ||
        bio.text != (currentUser?.bio ?: "") ||
        website.text != (currentUser?.website ?: "") ||
        location.text != (currentUser?.location ?: "") ||
        selectedAvatarUri != null ||
        selectedBannerUri != null
    }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF050505))
    ) {
        // Background orbs
        FuturisticEditBackground()
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            // Header Bar
            EditProfileHeader(onNavigateBack = onNavigateBack)
            
            // Main content
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
            ) {
                // Glass Card
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(RoundedCornerShape(24.dp))
                        .background(Color.Black.copy(alpha = 0.4f))
                        .border(
                            1.dp,
                            Color.White.copy(alpha = 0.1f),
                            RoundedCornerShape(24.dp)
                        )
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .verticalScroll(rememberScrollState())
                    ) {
                        // Identity Section (Left column content)
                        IdentitySection(
                            currentAvatarUrl = currentUser?.avatar,
                            selectedAvatarUri = selectedAvatarUri,
                            username = username.text,
                            profileStrength = profileStrength,
                            onAvatarClick = { avatarPicker.launch("image/*") },
                            onCoverClick = { bannerPicker.launch("image/*") }
                        )
                        
                        Divider(
                            color = Color.White.copy(alpha = 0.1f),
                            modifier = Modifier.padding(horizontal = 24.dp)
                        )
                        
                        // Form Fields Section
                        FormFieldsSection(
                            username = username,
                            onUsernameChange = { username = it },
                            bio = bio,
                            onBioChange = { bio = it },
                            website = website,
                            onWebsiteChange = { website = it },
                            location = location,
                            onLocationChange = { location = it }
                        )
                        
                        Spacer(modifier = Modifier.height(24.dp))
                        
                        // Action Buttons
                        ActionButtons(
                            onCancel = onNavigateBack,
                            onSave = {
                                if (hasChanges && !isSaving) {
                                    isSaving = true
                                    scope.launch {
                                        try {
                                            var newAvatarUrl = currentUser?.avatar
                                            var newBannerUrl = currentUser?.banner
                                            
                                            // Upload new avatar
                                            selectedAvatarUri?.let { uri ->
                                                currentUser?.userId?.let { userId ->
                                                    newAvatarUrl = S3Service.uploadProfileImage(context, userId, uri)
                                                }
                                            }
                                            
                                            // Upload new banner (using post media folder for banners)
                                            selectedBannerUri?.let { uri ->
                                                currentUser?.userId?.let { userId ->
                                                    newBannerUrl = S3Service.uploadPostMedia(context, "banner_$userId", uri)
                                                }
                                            }
                                            
                                            // Update user
                                            currentUser?.copy(
                                                username = username.text,
                                                bio = bio.text,
                                                website = website.text,
                                                location = location.text,
                                                avatar = newAvatarUrl,
                                                banner = newBannerUrl
                                            )?.let {
                                                DynamoDbService.updateUser(it)
                                                AuthManager.updateCurrentUser(it)
                                                
                                                // Update posts with new username/avatar
                                                DynamoDbService.updateUserPostsProfile(
                                                    userId = it.userId,
                                                    newUsername = it.username,
                                                    newAvatar = it.avatar
                                                )
                                            }
                                            
                                            showSuccess = true
                                            onSaveSuccess()
                                            onNavigateBack()
                                        } catch (e: Exception) {
                                            e.printStackTrace()
                                        } finally {
                                            isSaving = false
                                        }
                                    }
                                }
                            },
                            isSaving = isSaving,
                            hasChanges = hasChanges
                        )
                        
                        Spacer(modifier = Modifier.height(32.dp))
                    }
                }
            }
        }
        
        // Loading overlay
        if (isSaving) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.6f)),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CircularProgressIndicator(
                        color = GradientCyan,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Saving...",
                        color = Color.White,
                        fontSize = 14.sp
                    )
                }
            }
        }
    }
}

// =============================================================================
// BACKGROUND
// =============================================================================

@Composable
private fun FuturisticEditBackground() {
    Box(modifier = Modifier.fillMaxSize()) {
        // Purple orb
        Box(
            modifier = Modifier
                .offset(x = (-50).dp, y = (-50).dp)
                .size(400.dp)
                .blur(120.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            Color(0xFF581C87).copy(alpha = 0.25f),
                            Color.Transparent
                        )
                    ),
                    shape = CircleShape
                )
        )
        
        // Cyan orb
        Box(
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .offset(x = 50.dp, y = 50.dp)
                .size(350.dp)
                .blur(100.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            Color(0xFF164E63).copy(alpha = 0.15f),
                            Color.Transparent
                        )
                    ),
                    shape = CircleShape
                )
        )
    }
}

// =============================================================================
// HEADER
// =============================================================================

@Composable
private fun EditProfileHeader(onNavigateBack: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Back button
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.clickable { onNavigateBack() }
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(Color.White.copy(alpha = 0.05f))
                    .border(1.dp, Color.White.copy(alpha = 0.1f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint = Color.Gray,
                    modifier = Modifier.size(18.dp)
                )
            }
            Text(
                text = "Back",
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = Color.Gray,
                letterSpacing = 1.sp
            )
        }
        
        // Title
        Text(
            text = "Edit Profile",
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        
        // Spacer for alignment
        Spacer(modifier = Modifier.width(80.dp))
    }
}

// =============================================================================
// IDENTITY SECTION
// =============================================================================

@Composable
private fun IdentitySection(
    currentAvatarUrl: String?,
    selectedAvatarUri: Uri?,
    username: String,
    profileStrength: Int,
    onAvatarClick: () -> Unit,
    onCoverClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Section title
        Text(
            text = "Identity",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        Text(
            text = "UPDATE YOUR DIGITAL PERSONA",
            fontSize = 10.sp,
            color = Color.Gray,
            letterSpacing = 2.sp
        )
        
        Spacer(modifier = Modifier.height(24.dp))
        
        // Avatar Editor with glow
        Box(
            modifier = Modifier
                .size(150.dp)
                .clickable { onAvatarClick() },
            contentAlignment = Alignment.Center
        ) {
            // Glow effect
            Box(
                modifier = Modifier
                    .size(160.dp)
                    .blur(20.dp)
                    .background(
                        brush = Brush.radialGradient(
                            colors = listOf(
                                GradientCyan.copy(alpha = 0.5f),
                                Color(0xFF8B5CF6).copy(alpha = 0.3f),
                                Color.Transparent
                            )
                        ),
                        shape = CircleShape
                    )
            )
            
            // Avatar container
            Box(
                modifier = Modifier
                    .size(140.dp)
                    .clip(CircleShape)
                    .background(Color.Black)
                    .border(3.dp, Color.Black, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                if (selectedAvatarUri != null) {
                    AsyncImage(
                        model = selectedAvatarUri,
                        contentDescription = "Selected Avatar",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(CircleShape),
                        contentScale = ContentScale.Crop
                    )
                } else if (!currentAvatarUrl.isNullOrBlank()) {
                    AsyncImage(
                        model = currentAvatarUrl,
                        contentDescription = "Current Avatar",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(CircleShape),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    Text(
                        text = username.firstOrNull()?.uppercase() ?: "U",
                        fontSize = 48.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }
                
                // Hover overlay
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.5f)),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Icon(
                            imageVector = Icons.Filled.CameraAlt,
                            contentDescription = "Change",
                            tint = Color.White,
                            modifier = Modifier.size(28.dp)
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "CHANGE",
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold,
                            color = GradientCyan,
                            letterSpacing = 2.sp
                        )
                    }
                }
            }
        }
        
        Spacer(modifier = Modifier.height(24.dp))
        
        // Cover Photo Button
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(80.dp)
                .clip(RoundedCornerShape(12.dp))
                .border(
                    width = 1.dp,
                    color = Color.White.copy(alpha = 0.2f),
                    shape = RoundedCornerShape(12.dp)
                )
                .clickable { onCoverClick() },
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Outlined.CloudUpload,
                    contentDescription = "Upload",
                    tint = Color.Gray,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "CHANGE COVER ART",
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Gray,
                    letterSpacing = 2.sp
                )
            }
        }
        
        Spacer(modifier = Modifier.height(24.dp))
        
        // Profile Strength
        ProfileStrengthCard(strength = profileStrength)
    }
}

@Composable
private fun ProfileStrengthCard(strength: Int) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Color.White.copy(alpha = 0.05f))
            .border(1.dp, Color.White.copy(alpha = 0.05f), RoundedCornerShape(12.dp))
            .padding(16.dp)
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "PROFILE STRENGTH",
                    fontSize = 10.sp,
                    color = Color.Gray,
                    letterSpacing = 2.sp
                )
                Text(
                    text = "${strength}%",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (strength >= 75) Color(0xFF34D399) else GradientCyan
                )
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Progress bar
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(RoundedCornerShape(3.dp))
                    .background(Color(0xFF1F2937))
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(strength / 100f)
                        .fillMaxHeight()
                        .clip(RoundedCornerShape(3.dp))
                        .background(
                            brush = Brush.horizontalGradient(
                                colors = listOf(
                                    Color(0xFF34D399),
                                    GradientCyan
                                )
                            )
                        )
                )
            }
        }
    }
}

// =============================================================================
// FORM FIELDS
// =============================================================================

@Composable
private fun FormFieldsSection(
    username: TextFieldValue,
    onUsernameChange: (TextFieldValue) -> Unit,
    bio: TextFieldValue,
    onBioChange: (TextFieldValue) -> Unit,
    website: TextFieldValue,
    onWebsiteChange: (TextFieldValue) -> Unit,
    location: TextFieldValue,
    onLocationChange: (TextFieldValue) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(24.dp)
    ) {
        // Row 1: Display Name & Username
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            FuturisticInputField(
                modifier = Modifier.weight(1f),
                label = "Display Name",
                icon = Icons.Outlined.Person,
                value = username,
                onValueChange = onUsernameChange,
                placeholder = "Your name"
            )
        }
        
        Spacer(modifier = Modifier.height(20.dp))
        
        // Bio
        FuturisticInputField(
            modifier = Modifier.fillMaxWidth(),
            label = "Bio / Manifesto",
            icon = Icons.Outlined.Tag,
            value = bio,
            onValueChange = onBioChange,
            placeholder = "Tell the world who you are...",
            multiline = true
        )
        
        Spacer(modifier = Modifier.height(20.dp))
        
        // Row 2: Website & Location
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            FuturisticInputField(
                modifier = Modifier.weight(1f),
                label = "Website",
                icon = Icons.Outlined.Link,
                value = website,
                onValueChange = onWebsiteChange,
                placeholder = "https://your-site.com"
            )
        }
        
        Spacer(modifier = Modifier.height(20.dp))
        
        FuturisticInputField(
            modifier = Modifier.fillMaxWidth(),
            label = "Location",
            icon = Icons.Outlined.LocationOn,
            value = location,
            onValueChange = onLocationChange,
            placeholder = "City, Country"
        )
        
        Spacer(modifier = Modifier.height(24.dp))
        
        // Interests/Tags
        InterestsTags()
    }
}

@Composable
private fun FuturisticInputField(
    modifier: Modifier = Modifier,
    label: String,
    icon: ImageVector,
    value: TextFieldValue,
    onValueChange: (TextFieldValue) -> Unit,
    placeholder: String,
    multiline: Boolean = false
) {
    Column(modifier = modifier) {
        Text(
            text = label.uppercase(),
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold,
            color = Color.Gray,
            letterSpacing = 2.sp
        )
        
        Spacer(modifier = Modifier.height(8.dp))
        
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .then(if (multiline) Modifier.height(120.dp) else Modifier.height(52.dp))
                .clip(RoundedCornerShape(12.dp))
                .background(Color.White.copy(alpha = 0.05f))
                .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(12.dp))
        ) {
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp),
                verticalAlignment = if (multiline) Alignment.Top else Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = Color.Gray,
                    modifier = Modifier
                        .size(16.dp)
                        .then(if (multiline) Modifier.padding(top = 16.dp) else Modifier)
                )
                
                Spacer(modifier = Modifier.width(12.dp))
                
                BasicTextField(
                    value = value,
                    onValueChange = onValueChange,
                    modifier = Modifier
                        .weight(1f)
                        .then(if (multiline) Modifier.padding(vertical = 12.dp) else Modifier),
                    textStyle = TextStyle(
                        fontSize = 14.sp,
                        color = Color.White
                    ),
                    cursorBrush = SolidColor(GradientCyan),
                    singleLine = !multiline,
                    decorationBox = { innerTextField ->
                        Box {
                            if (value.text.isEmpty()) {
                                Text(
                                    text = placeholder,
                                    style = TextStyle(
                                        fontSize = 14.sp,
                                        color = Color(0xFF4B5563)
                                    )
                                )
                            }
                            innerTextField()
                        }
                    }
                )
            }
        }
    }
}

@Composable
private fun InterestsTags() {
    Column {
        Text(
            text = "INTERESTS",
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold,
            color = Color.Gray,
            letterSpacing = 2.sp
        )
        
        Spacer(modifier = Modifier.height(12.dp))
        
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            listOf("Creator", "Social", "Digital", "Tech").forEach { tag ->
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color.White.copy(alpha = 0.05f))
                        .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(8.dp))
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = tag,
                            fontSize = 12.sp,
                            color = Color(0xFFD1D5DB)
                        )
                        Icon(
                            imageVector = Icons.Filled.Close,
                            contentDescription = "Remove",
                            tint = Color.Gray,
                            modifier = Modifier.size(12.dp)
                        )
                    }
                }
            }
            
            // Add tag button
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .border(
                        width = 1.dp,
                        color = Color.White.copy(alpha = 0.2f),
                        shape = RoundedCornerShape(8.dp)
                    )
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                Text(
                    text = "+ Add Tag",
                    fontSize = 12.sp,
                    color = Color.Gray
                )
            }
        }
    }
}

// =============================================================================
// ACTION BUTTONS
// =============================================================================

@Composable
private fun ActionButtons(
    onCancel: () -> Unit,
    onSave: () -> Unit,
    isSaving: Boolean,
    hasChanges: Boolean
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Cancel button
        OutlinedButton(
            onClick = onCancel,
            modifier = Modifier
                .weight(1f)
                .height(52.dp),
            shape = RoundedCornerShape(12.dp),
            border = BorderStroke(1.dp, Color.White.copy(alpha = 0.1f)),
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = Color.Gray
            )
        ) {
            Text(
                text = "Cancel",
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp
            )
        }
        
        // Save button
        Button(
            onClick = onSave,
            modifier = Modifier
                .weight(1.5f)
                .height(52.dp),
            enabled = hasChanges && !isSaving,
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.Transparent,
                disabledContainerColor = Color.Gray.copy(alpha = 0.2f)
            ),
            contentPadding = PaddingValues(0.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        brush = Brush.horizontalGradient(
                            colors = if (hasChanges && !isSaving) 
                                listOf(Color(0xFF4F46E5), Color(0xFF7C3AED))
                            else 
                                listOf(Color.Gray.copy(alpha = 0.3f), Color.Gray.copy(alpha = 0.3f))
                        ),
                        shape = RoundedCornerShape(12.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.Save,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        text = "Save Changes",
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }
            }
        }
    }
}
