package com.orignal.buddylynk.ui.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.ColorMatrix
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.orignal.buddylynk.ui.viewmodel.*

// Premium Colors
private val PremiumPurple = Color(0xFF8B5CF6)
private val PremiumIndigo = Color(0xFF6366F1)
private val PremiumPink = Color(0xFFEC4899)
private val DarkBg = Color(0xFF000000)
private val CardBg = Color(0xFF18181B)
private val BorderColor = Color(0xFF27272A)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreatePostScreen(
    onNavigateBack: () -> Unit,
    viewModel: CreatePostViewModel = viewModel()
) {
    val context = LocalContext.current
    val imageUri by viewModel.selectedImageUri.collectAsState()
    val caption by viewModel.caption.collectAsState()
    val location by viewModel.location.collectAsState()
    val selectedFilter by viewModel.selectedFilter.collectAsState()
    val adjustments by viewModel.adjustments.collectAsState()
    val uploadMode by viewModel.uploadMode.collectAsState()
    val editTab by viewModel.editTab.collectAsState()
    val postState by viewModel.postState.collectAsState()

    // Image Picker
    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.setImageUri(it) }
    }

    // Success State
    if (postState == PostState.SUCCESS) {
        SuccessScreen(onDismiss = {
            viewModel.clearSelection()
            onNavigateBack()
        })
        return
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBg)
    ) {
        // Ambient Background Glow
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(300.dp)
                .offset(y = (-100).dp)
                .blur(100.dp)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            PremiumIndigo.copy(alpha = 0.2f),
                            Color.Transparent
                        )
                    )
                )
        )

        Column(modifier = Modifier.fillMaxSize()) {
            // Top Navigation Bar
            TopNavBar(
                hasImage = imageUri != null,
                isPosting = postState == PostState.UPLOADING,
                onBackClick = {
                    if (imageUri != null) viewModel.clearSelection()
                    else onNavigateBack()
                },
                onShareClick = { viewModel.createPost(context) }
            )

            // Main Content
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 16.dp)
            ) {
                if (imageUri == null) {
                    // Upload Mode Selection
                    UploadModeSelector(
                        currentMode = uploadMode,
                        onModeChange = { viewModel.setUploadMode(it) },
                        onSelectImage = { imagePicker.launch("image/*") }
                    )
                } else {
                    // Image Editor
                    ImageEditorContent(
                        imageUri = imageUri!!,
                        selectedFilter = selectedFilter,
                        adjustments = adjustments,
                        filters = viewModel.filters,
                        editTab = editTab,
                        caption = caption,
                        location = location,
                        onFilterSelect = { viewModel.setFilter(it) },
                        onTabChange = { viewModel.setEditTab(it) },
                        onBrightnessChange = { viewModel.setBrightness(it) },
                        onContrastChange = { viewModel.setContrast(it) },
                        onSaturationChange = { viewModel.setSaturation(it) },
                        onCaptionChange = { viewModel.setCaption(it) },
                        onLocationChange = { viewModel.setLocation(it) }
                    )
                }
            }
        }
    }
}

@Composable
private fun TopNavBar(
    hasImage: Boolean,
    isPosting: Boolean,
    onBackClick: () -> Unit,
    onShareClick: () -> Unit
) {
    val currentUser = com.orignal.buddylynk.data.auth.AuthManager.currentUser.collectAsState()
    
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .height(60.dp)
            .background(DarkBg.copy(alpha = 0.8f))
            .padding(horizontal = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        // Left: Back Button + User Avatar
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            IconButton(onClick = onBackClick) {
                Icon(
                    imageVector = if (hasImage) Icons.AutoMirrored.Filled.ArrowBack else Icons.Filled.Close,
                    contentDescription = "Back",
                    tint = Color.White
                )
            }
            
            // User Profile Avatar (like React design)
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.linearGradient(
                            colors = listOf(PremiumIndigo, PremiumPink)
                        )
                    )
                    .padding(1.5.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(CircleShape)
                        .background(DarkBg),
                    contentAlignment = Alignment.Center
                ) {
                    val avatar = currentUser.value?.avatar
                    if (avatar != null) {
                        AsyncImage(
                            model = avatar,
                            contentDescription = null,
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Crop
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Filled.Person,
                            contentDescription = null,
                            tint = Color.Gray,
                            modifier = Modifier.size(14.dp)
                        )
                    }
                }
            }
        }

        // Center: Title (only when no image)
        if (!hasImage) {
            Text(
                text = "New Post",
                color = Color.White,
                fontWeight = FontWeight.SemiBold,
                fontSize = 17.sp
            )
        } else {
            Spacer(modifier = Modifier.weight(1f))
        }

        // Right: Share Button
        AnimatedVisibility(
            visible = hasImage,
            enter = fadeIn() + scaleIn(),
            exit = fadeOut() + scaleOut()
        ) {
            Button(
                onClick = onShareClick,
                enabled = !isPosting,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White,
                    contentColor = Color.Black,
                    disabledContainerColor = Color.White.copy(alpha = 0.5f)
                ),
                shape = RoundedCornerShape(24.dp),
                contentPadding = PaddingValues(horizontal = 20.dp, vertical = 8.dp)
            ) {
                if (isPosting) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp,
                        color = Color.Black
                    )
                } else {
                    Text("Share", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                }
            }
        }

        // Spacer when no image
        if (!hasImage) {
            Spacer(modifier = Modifier.width(48.dp))
        }
    }
}

@Composable
private fun UploadModeSelector(
    currentMode: UploadMode,
    onModeChange: (UploadMode) -> Unit,
    onSelectImage: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Mode Tabs
        Row(
            modifier = Modifier
                .clip(RoundedCornerShape(24.dp))
                .background(CardBg)
                .border(1.dp, BorderColor, RoundedCornerShape(24.dp))
                .padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            ModeTab(
                icon = Icons.Outlined.Folder,
                isSelected = currentMode == UploadMode.FILES,
                onClick = { onModeChange(UploadMode.FILES) }
            )
            ModeTab(
                icon = Icons.Outlined.CameraAlt,
                isSelected = currentMode == UploadMode.CAMERA,
                onClick = { onModeChange(UploadMode.CAMERA) }
            )
            ModeTab(
                icon = Icons.Outlined.Event,
                isSelected = currentMode == UploadMode.EVENT,
                onClick = { onModeChange(UploadMode.EVENT) }
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Mode Content
        when (currentMode) {
            UploadMode.FILES -> FilesUploadCard(onSelectImage = onSelectImage)
            UploadMode.CAMERA -> CameraCard()
            UploadMode.EVENT -> EventCard()
        }
    }
}

@Composable
private fun ModeTab(
    icon: ImageVector,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .size(44.dp)
            .clip(CircleShape)
            .background(if (isSelected) CardBg.copy(alpha = 0.8f) else Color.Transparent)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = if (isSelected) PremiumIndigo else Color.Gray,
            modifier = Modifier.size(20.dp)
        )
    }
}

@Composable
private fun FilesUploadCard(onSelectImage: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(4f / 5f)
            .clip(RoundedCornerShape(32.dp))
            .background(CardBg.copy(alpha = 0.3f))
            .border(1.dp, BorderColor, RoundedCornerShape(32.dp))
            .clickable(onClick = onSelectImage),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                modifier = Modifier
                    .size(96.dp)
                    .clip(CircleShape)
                    .background(CardBg)
                    .border(1.dp, BorderColor, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Outlined.AddPhotoAlternate,
                    contentDescription = null,
                    tint = Color.Gray,
                    modifier = Modifier.size(40.dp)
                )
            }
            Spacer(modifier = Modifier.height(24.dp))
            Text(
                text = "Select Media",
                color = Color.White,
                fontSize = 20.sp,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Tap to browse gallery",
                color = Color.Gray,
                fontSize = 14.sp
            )
        }
    }
}

@Composable
private fun CameraCard() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(4f / 5f)
            .clip(RoundedCornerShape(32.dp))
            .background(Color.Black)
            .border(1.dp, BorderColor, RoundedCornerShape(32.dp)),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(
                imageVector = Icons.Filled.CameraAlt,
                contentDescription = null,
                tint = Color.Gray.copy(alpha = 0.5f),
                modifier = Modifier.size(48.dp)
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text("Camera Mode", color = Color.Gray, fontSize = 16.sp)
        }
    }
}

@Composable
private fun EventCard() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(4f / 5f)
            .clip(RoundedCornerShape(32.dp))
            .background(
                Brush.verticalGradient(
                    colors = listOf(CardBg.copy(alpha = 0.8f), DarkBg)
                )
            )
            .border(1.dp, BorderColor, RoundedCornerShape(32.dp))
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(PremiumIndigo.copy(alpha = 0.1f))
                    .border(1.dp, PremiumIndigo.copy(alpha = 0.2f), RoundedCornerShape(16.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.Event,
                    contentDescription = null,
                    tint = PremiumIndigo,
                    modifier = Modifier.size(36.dp)
                )
            }
            Spacer(modifier = Modifier.height(24.dp))
            Text(
                text = "New Event",
                color = Color.White,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "Host a meetup, party, or\nlive session for your community.",
                color = Color.Gray,
                fontSize = 14.sp,
                textAlign = TextAlign.Center,
                lineHeight = 20.sp
            )
            Spacer(modifier = Modifier.height(32.dp))
            Button(
                onClick = { },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White,
                    contentColor = Color.Black
                ),
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    "Create Event",
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }
        }
    }
}

@Composable
private fun ImageEditorContent(
    imageUri: Uri,
    selectedFilter: ImageFilter,
    adjustments: ImageAdjustments,
    filters: List<ImageFilter>,
    editTab: EditTab,
    caption: String,
    location: String,
    onFilterSelect: (ImageFilter) -> Unit,
    onTabChange: (EditTab) -> Unit,
    onBrightnessChange: (Float) -> Unit,
    onContrastChange: (Float) -> Unit,
    onSaturationChange: (Float) -> Unit,
    onCaptionChange: (String) -> Unit,
    onLocationChange: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 16.dp, bottom = 100.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        // Image Preview with Filters
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(4f / 5f)
                .clip(RoundedCornerShape(32.dp))
                .background(CardBg)
        ) {
            AsyncImage(
                model = imageUri,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
                colorFilter = getColorFilter(selectedFilter, adjustments)
            )
            
            // Maximize Button (like React design)
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
                    .size(44.dp)
                    .clip(CircleShape)
                    .background(Color.Black.copy(alpha = 0.3f))
                    .border(1.dp, Color.White.copy(alpha = 0.1f), CircleShape)
                    .clickable { /* TODO: Fullscreen */ },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Outlined.Fullscreen,
                    contentDescription = "Fullscreen",
                    tint = Color.White,
                    modifier = Modifier.size(20.dp)
                )
            }
        }

        // Tab Switcher
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            Row(
                modifier = Modifier
                    .clip(RoundedCornerShape(24.dp))
                    .background(CardBg.copy(alpha = 0.8f))
                    .border(1.dp, BorderColor, RoundedCornerShape(24.dp))
                    .padding(4.dp)
            ) {
                EditTabButton(
                    icon = Icons.Outlined.AutoAwesome,
                    label = "Filters",
                    isSelected = editTab == EditTab.FILTER,
                    onClick = { onTabChange(EditTab.FILTER) }
                )
                EditTabButton(
                    icon = Icons.Outlined.Tune,
                    label = "Edit",
                    isSelected = editTab == EditTab.EDIT,
                    onClick = { onTabChange(EditTab.EDIT) }
                )
            }
        }

        // Filter/Edit Content
        AnimatedContent(
            targetState = editTab,
            transitionSpec = {
                fadeIn(animationSpec = tween(200)) togetherWith fadeOut(animationSpec = tween(200))
            },
            label = "editTab"
        ) { tab ->
            when (tab) {
                EditTab.FILTER -> FilterSelector(
                    filters = filters,
                    selectedFilter = selectedFilter,
                    imageUri = imageUri,
                    adjustments = adjustments,
                    onSelect = onFilterSelect
                )
                EditTab.EDIT -> AdjustmentsPanel(
                    adjustments = adjustments,
                    onBrightnessChange = onBrightnessChange,
                    onContrastChange = onContrastChange,
                    onSaturationChange = onSaturationChange
                )
            }
        }

        // Caption Input
        CaptionInput(
            caption = caption,
            onCaptionChange = onCaptionChange
        )

        // Metadata Options
        MetadataOptions(
            location = location,
            onLocationChange = onLocationChange
        )
    }
}

@Composable
private fun EditTabButton(
    icon: ImageVector,
    label: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(if (isSelected) CardBg else Color.Transparent)
            .clickable(onClick = onClick)
            .padding(horizontal = 20.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = if (isSelected) Color.White else Color.Gray,
            modifier = Modifier.size(16.dp)
        )
        Text(
            text = label,
            color = if (isSelected) Color.White else Color.Gray,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
private fun FilterSelector(
    filters: List<ImageFilter>,
    selectedFilter: ImageFilter,
    imageUri: Uri,
    adjustments: ImageAdjustments,
    onSelect: (ImageFilter) -> Unit
) {
    LazyRow(
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(horizontal = 4.dp)
    ) {
        items(filters) { filter ->
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.clickable { onSelect(filter) }
            ) {
                Box(
                    modifier = Modifier
                        .size(80.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .border(
                            width = if (selectedFilter == filter) 2.dp else 0.dp,
                            color = if (selectedFilter == filter) PremiumIndigo else Color.Transparent,
                            shape = RoundedCornerShape(16.dp)
                        )
                ) {
                    AsyncImage(
                        model = imageUri,
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                        colorFilter = getColorFilter(filter, adjustments)
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = filter.name,
                    color = if (selectedFilter == filter) PremiumIndigo else Color.Gray,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }
    }
}

@Composable
private fun AdjustmentsPanel(
    adjustments: ImageAdjustments,
    onBrightnessChange: (Float) -> Unit,
    onContrastChange: (Float) -> Unit,
    onSaturationChange: (Float) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .background(CardBg.copy(alpha = 0.4f))
            .border(1.dp, BorderColor, RoundedCornerShape(24.dp))
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        AdjustmentSlider(
            label = "Brightness",
            icon = Icons.Outlined.WbSunny,
            value = adjustments.brightness,
            onValueChange = onBrightnessChange,
            valueRange = 0.5f..1.5f
        )
        AdjustmentSlider(
            label = "Contrast",
            icon = Icons.Outlined.Contrast,
            value = adjustments.contrast,
            onValueChange = onContrastChange,
            valueRange = 0.5f..1.5f
        )
        AdjustmentSlider(
            label = "Saturation",
            icon = Icons.Outlined.WaterDrop,
            value = adjustments.saturation,
            onValueChange = onSaturationChange,
            valueRange = 0f..2f
        )
    }
}

@Composable
private fun AdjustmentSlider(
    label: String,
    icon: ImageVector,
    value: Float,
    onValueChange: (Float) -> Unit,
    valueRange: ClosedFloatingPointRange<Float>
) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = Color.Gray,
                    modifier = Modifier.size(16.dp)
                )
                Text(label, color = Color.Gray, fontSize = 12.sp, fontWeight = FontWeight.Medium)
            }
            Text(
                text = "${(value * 100).toInt()}",
                color = Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium,
                modifier = Modifier
                    .background(CardBg, RoundedCornerShape(4.dp))
                    .padding(horizontal = 8.dp, vertical = 2.dp)
            )
        }
        Spacer(modifier = Modifier.height(12.dp))
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            colors = SliderDefaults.colors(
                thumbColor = Color.White,
                activeTrackColor = Color.White,
                inactiveTrackColor = Color.Gray.copy(alpha = 0.3f)
            )
        )
    }
}

@Composable
private fun CaptionInput(
    caption: String,
    onCaptionChange: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(CardBg.copy(alpha = 0.3f))
            .border(1.dp, BorderColor, RoundedCornerShape(20.dp))
            .padding(16.dp)
    ) {
        BasicTextField(
            value = caption,
            onValueChange = onCaptionChange,
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 80.dp),
            textStyle = androidx.compose.ui.text.TextStyle(
                color = Color.White,
                fontSize = 16.sp,
                lineHeight = 24.sp
            ),
            cursorBrush = SolidColor(PremiumPurple),
            decorationBox = { innerTextField ->
                Box {
                    if (caption.isEmpty()) {
                        Text(
                            "Write a caption...",
                            color = Color.Gray,
                            fontSize = 16.sp
                        )
                    }
                    innerTextField()
                }
            }
        )
        Spacer(modifier = Modifier.height(12.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                IconButton(onClick = { }, modifier = Modifier.size(36.dp)) {
                    Icon(Icons.Outlined.Tag, null, tint = Color.Gray, modifier = Modifier.size(20.dp))
                }
                IconButton(onClick = { }, modifier = Modifier.size(36.dp)) {
                    Icon(Icons.Outlined.Person, null, tint = Color.Gray, modifier = Modifier.size(20.dp))
                }
            }
            IconButton(onClick = { }, modifier = Modifier.size(36.dp)) {
                Icon(Icons.Outlined.EmojiEmotions, null, tint = Color.Gray, modifier = Modifier.size(20.dp))
            }
        }
    }
}

@Composable
private fun MetadataOptions(
    location: String,
    onLocationChange: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        MetadataButton(
            icon = Icons.Outlined.LocationOn,
            label = location.ifEmpty { "Add Location" },
            isActive = location.isNotEmpty(),
            onClick = { if (location.isEmpty()) onLocationChange("New York, USA") },
            onClear = { onLocationChange("") }
        )
        MetadataButton(
            icon = Icons.Outlined.MusicNote,
            label = "Add Music",
            isActive = false,
            onClick = { }
        )
        MetadataButton(
            icon = Icons.Outlined.MoreHoriz,
            label = "Advanced Settings",
            isActive = false,
            onClick = { }
        )
    }
}

@Composable
private fun MetadataButton(
    icon: ImageVector,
    label: String,
    isActive: Boolean,
    onClick: () -> Unit,
    onClear: (() -> Unit)? = null
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(CardBg.copy(alpha = 0.2f))
            .border(1.dp, Color.Transparent, RoundedCornerShape(16.dp))
            .clickable(onClick = onClick)
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(if (isActive) PremiumIndigo.copy(alpha = 0.2f) else CardBg.copy(alpha = 0.5f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = if (isActive) PremiumIndigo else Color.Gray,
                    modifier = Modifier.size(20.dp)
                )
            }
            Text(
                text = label,
                color = if (isActive) Color.White else Color.Gray,
                fontSize = 14.sp,
                fontWeight = if (isActive) FontWeight.Medium else FontWeight.Normal
            )
        }
        if (isActive && onClear != null) {
            IconButton(onClick = onClear, modifier = Modifier.size(32.dp)) {
                Icon(Icons.Filled.Close, null, tint = Color.Gray, modifier = Modifier.size(16.dp))
            }
        } else {
            Text(
                "Add",
                color = Color.Gray.copy(alpha = 0.6f),
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@Composable
private fun SuccessScreen(onDismiss: () -> Unit) {
    LaunchedEffect(Unit) {
        kotlinx.coroutines.delay(2500)
        onDismiss()
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBg),
        contentAlignment = Alignment.Center
    ) {
        // Background Glow
        Box(
            modifier = Modifier
                .size(256.dp)
                .blur(120.dp)
                .background(PremiumIndigo.copy(alpha = 0.3f), CircleShape)
        )

        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                modifier = Modifier
                    .size(96.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.linearGradient(
                            colors = listOf(PremiumIndigo, PremiumPurple)
                        )
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.Check,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(48.dp)
                )
            }
            Spacer(modifier = Modifier.height(32.dp))
            Text(
                text = "Uploaded",
                color = Color.White,
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "Your moment has been shared.",
                color = Color.Gray,
                fontSize = 16.sp
            )
        }
    }
}

// Helper function for color filters
private fun getColorFilter(filter: ImageFilter, adjustments: ImageAdjustments): ColorFilter {
    val colorMatrix = ColorMatrix()

    // Apply adjustments
    colorMatrix.setToScale(
        adjustments.brightness,
        adjustments.brightness,
        adjustments.brightness,
        1f
    )

    // Apply saturation from adjustments
    val satMatrix = ColorMatrix()
    satMatrix.setToSaturation(adjustments.saturation * filter.saturation)
    colorMatrix.timesAssign(satMatrix)

    // Apply grayscale if filter requires it
    if (filter.grayscale) {
        val grayMatrix = ColorMatrix()
        grayMatrix.setToSaturation(0f)
        colorMatrix.timesAssign(grayMatrix)
    }

    return ColorFilter.colorMatrix(colorMatrix)
}
