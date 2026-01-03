package com.orignal.buddylynk.ui.screens

import androidx.compose.animation.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.data.model.Group
import kotlinx.coroutines.launch
import java.util.UUID

// Premium Colors
private val DarkBg = Color(0xFF050505)
private val CardBg = Color(0xFF1A1A1A)
private val Zinc800 = Color(0xFF27272A)
private val Zinc600 = Color(0xFF52525B)
private val Zinc500 = Color(0xFF71717A)
private val IndigoAccent = Color(0xFF6366F1)
private val VioletAccent = Color(0xFF8B5CF6)
private val CyanAccent = Color(0xFF22D3EE)

/**
 * Create Group/Channel Screen - Premium UI
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreateGroupScreen(
    isChannel: Boolean = false,
    onNavigateBack: () -> Unit = {},
    onGroupCreated: (String) -> Unit = {}
) {
    val scope = rememberCoroutineScope()
    var groupName by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    
    val currentUser = AuthManager.currentUser.collectAsState().value
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBg)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
        ) {
            // Header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onNavigateBack) {
                    Icon(
                        imageVector = Icons.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = Color.White
                    )
                }
                
                Spacer(modifier = Modifier.width(8.dp))
                
                Text(
                    text = if (isChannel) "New Channel" else "New Group",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                
                Spacer(modifier = Modifier.weight(1f))
                
                // Create Button
                TextButton(
                    onClick = {
                        if (groupName.isBlank()) {
                            error = "Please enter a name"
                            return@TextButton
                        }
                        if (currentUser == null) {
                            error = "Not logged in"
                            return@TextButton
                        }
                        
                        isLoading = true
                        error = null
                        
                        scope.launch {
                            val group = Group(
                                groupId = UUID.randomUUID().toString(),
                                name = groupName.trim(),
                                description = description.ifBlank { null },
                                imageUrl = null,
                                creatorId = currentUser.userId,
                                memberIds = listOf(currentUser.userId),
                                memberCount = 1,
                                isPublic = isChannel,
                                createdAt = System.currentTimeMillis().toString()
                            )
                            
                            val success = BackendRepository.createGroup(group)
                            isLoading = false
                            
                            if (success) {
                                onGroupCreated(group.groupId)
                            } else {
                                error = "Failed to create ${if (isChannel) "channel" else "group"}"
                            }
                        }
                    },
                    enabled = groupName.isNotBlank() && !isLoading
                ) {
                    if (isLoading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            color = IndigoAccent,
                            strokeWidth = 2.dp
                        )
                    } else {
                        Text(
                            text = "Create",
                            color = if (groupName.isNotBlank()) IndigoAccent else Zinc500,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
            
            // Content
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
                    .verticalScroll(rememberScrollState())
            ) {
                // Icon Selection Area
                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .align(Alignment.CenterHorizontally)
                        .clip(CircleShape)
                        .background(
                            Brush.linearGradient(
                                colors = listOf(IndigoAccent.copy(alpha = 0.3f), VioletAccent.copy(alpha = 0.3f))
                            )
                        )
                        .border(2.dp, Color.White.copy(alpha = 0.1f), CircleShape)
                        .clickable { /* TODO: Image picker */ },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (isChannel) Icons.Filled.Campaign else Icons.Filled.Group,
                        contentDescription = null,
                        tint = IndigoAccent,
                        modifier = Modifier.size(40.dp)
                    )
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Text(
                    text = "Add Photo",
                    color = IndigoAccent,
                    fontSize = 14.sp,
                    modifier = Modifier.align(Alignment.CenterHorizontally)
                )
                
                Spacer(modifier = Modifier.height(32.dp))
                
                // Name Input
                Text(
                    text = if (isChannel) "Channel Name" else "Group Name",
                    color = Zinc500,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                OutlinedTextField(
                    value = groupName,
                    onValueChange = { groupName = it },
                    placeholder = { Text("Enter ${if (isChannel) "channel" else "group"} name") },
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = IndigoAccent,
                        unfocusedBorderColor = Zinc600,
                        cursorColor = IndigoAccent,
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedPlaceholderColor = Zinc500,
                        unfocusedPlaceholderColor = Zinc600
                    ),
                    shape = RoundedCornerShape(12.dp),
                    singleLine = true
                )
                
                Spacer(modifier = Modifier.height(24.dp))
                
                // Description Input
                Text(
                    text = "Description (Optional)",
                    color = Zinc500,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                OutlinedTextField(
                    value = description,
                    onValueChange = { description = it },
                    placeholder = { Text("What's this ${if (isChannel) "channel" else "group"} about?") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(120.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = IndigoAccent,
                        unfocusedBorderColor = Zinc600,
                        cursorColor = IndigoAccent,
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedPlaceholderColor = Zinc500,
                        unfocusedPlaceholderColor = Zinc600
                    ),
                    shape = RoundedCornerShape(12.dp),
                    maxLines = 4
                )
                
                // Error message
                error?.let {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = it,
                        color = Color(0xFFEF4444),
                        fontSize = 14.sp
                    )
                }
                
                Spacer(modifier = Modifier.height(32.dp))
                
                // Info Card
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(CardBg)
                        .border(1.dp, Color.White.copy(alpha = 0.05f), RoundedCornerShape(16.dp))
                        .padding(16.dp)
                ) {
                    Row(verticalAlignment = Alignment.Top) {
                        Icon(
                            imageVector = Icons.Filled.Info,
                            contentDescription = null,
                            tint = CyanAccent,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(
                            text = if (isChannel) {
                                "Channels are public and anyone can join. Use them to broadcast messages to a large audience."
                            } else {
                                "Groups are private. Only members you add can see messages and participate."
                            },
                            color = Zinc500,
                            fontSize = 13.sp,
                            lineHeight = 18.sp
                        )
                    }
                }
            }
        }
    }
}
