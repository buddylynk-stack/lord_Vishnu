package com.orignal.buddylynk.ui.components

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.orignal.buddylynk.ui.theme.*

/**
 * VideoPlayer - Video player for feed videos
 * 
 * To enable full video playback, add to build.gradle.kts:
 * implementation("androidx.media3:media3-exoplayer:1.2.0")
 * implementation("androidx.media3:media3-ui:1.2.0")
 */
@Composable
fun VideoPlayer(
    videoUrl: String,
    thumbnailUrl: String? = null,
    modifier: Modifier = Modifier,
    autoPlay: Boolean = false,
    showControls: Boolean = true,
    isFullScreen: Boolean = false,
    onFullScreenToggle: () -> Unit = {},
    onPlay: () -> Unit = {}
) {
    var isPlaying by remember { mutableStateOf(autoPlay) }
    var isMuted by remember { mutableStateOf(true) }
    var progress by remember { mutableFloatStateOf(0f) }
    var showControlsOverlay by remember { mutableStateOf(false) }
    
    // Simulate progress when playing
    LaunchedEffect(isPlaying) {
        if (isPlaying) {
            while (progress < 1f) {
                kotlinx.coroutines.delay(100)
                progress = (progress + 0.01f).coerceAtMost(1f)
            }
            isPlaying = false
            progress = 0f
        }
    }
    
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(if (isFullScreen) 0.dp else 12.dp))
            .background(Color.Black)
            .clickable { showControlsOverlay = !showControlsOverlay }
    ) {
        // Thumbnail/poster image
        thumbnailUrl?.let { url ->
            AsyncImage(
                model = url,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        }
        
        // Dark overlay when paused
        if (!isPlaying) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.3f))
            )
        }
        
        // Center play/pause button
        Box(
            modifier = Modifier
                .align(Alignment.Center)
                .size(64.dp)
                .clip(CircleShape)
                .background(Color.White.copy(alpha = 0.2f))
                .clickable {
                    isPlaying = !isPlaying
                    if (isPlaying) onPlay()
                },
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = if (isPlaying) Icons.Filled.Pause else Icons.Filled.PlayArrow,
                contentDescription = if (isPlaying) "Pause" else "Play",
                tint = Color.White,
                modifier = Modifier.size(32.dp)
            )
        }
        
        // Controls overlay
        AnimatedVisibility(
            visible = showControls && (showControlsOverlay || !isPlaying),
            enter = fadeIn(),
            exit = fadeOut()
        ) {
            // Bottom controls
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .background(Color.Black.copy(alpha = 0.5f))
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Progress bar
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier
                        .weight(1f)
                        .height(3.dp)
                        .clip(RoundedCornerShape(1.5.dp)),
                    color = Color.White,
                    trackColor = Color.White.copy(alpha = 0.3f)
                )
                
                Spacer(modifier = Modifier.width(8.dp))
                
                // Mute button
                IconButton(
                    onClick = { isMuted = !isMuted },
                    modifier = Modifier.size(32.dp)
                ) {
                    Icon(
                        imageVector = if (isMuted) Icons.Filled.VolumeOff else Icons.Filled.VolumeUp,
                        contentDescription = if (isMuted) "Unmute" else "Mute",
                        tint = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                }
                
                // Fullscreen button
                IconButton(
                    onClick = onFullScreenToggle,
                    modifier = Modifier.size(32.dp)
                ) {
                    Icon(
                        imageVector = if (isFullScreen) Icons.Filled.FullscreenExit else Icons.Filled.Fullscreen,
                        contentDescription = if (isFullScreen) "Exit fullscreen" else "Fullscreen",
                        tint = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }
        }
        
        // Video indicator badge
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(8.dp)
                .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(4.dp))
                .padding(horizontal = 6.dp, vertical = 2.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Filled.PlayCircle,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(12.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "Video",
                    color = Color.White,
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }
    }
}

/**
 * Mini video thumbnail for feed
 */
@Composable
fun VideoThumbnail(
    videoUrl: String,
    thumbnailUrl: String?,
    modifier: Modifier = Modifier,
    onClick: () -> Unit = {}
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(Color.Black)
            .clickable { onClick() }
    ) {
        // Thumbnail
        thumbnailUrl?.let {
            AsyncImage(
                model = it,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        }
        
        // Play button overlay
        Box(
            modifier = Modifier
                .align(Alignment.Center)
                .size(48.dp)
                .background(Color.White.copy(alpha = 0.2f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Filled.PlayArrow,
                contentDescription = "Play",
                tint = Color.White,
                modifier = Modifier.size(28.dp)
            )
        }
        
        // Video indicator
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(4.dp)
        ) {
            Icon(
                imageVector = Icons.Filled.PlayCircle,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(16.dp)
            )
        }
    }
}
