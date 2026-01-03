package com.orignal.buddylynk.ui.screens

import android.view.ViewGroup
import androidx.compose.animation.*
import androidx.compose.foundation.background
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import coil.compose.AsyncImage
import com.orignal.buddylynk.data.webrtc.CallManager
import kotlinx.coroutines.delay
import org.webrtc.SurfaceViewRenderer
import org.webrtc.VideoTrack

// Colors
private val CallGradientStart = Color(0xFF1A1A2E)
private val CallGradientEnd = Color(0xFF16213E)
private val AcceptGreen = Color(0xFF4CAF50)
private val RejectRed = Color(0xFFE63946)
private val ControlBg = Color(0xFF2A2A4A)

/**
 * Full-screen Call Screen for WebRTC video/voice calls
 * Industry-ready implementation
 */
@Composable
fun CallScreen(
    targetUserId: String = "",
    targetUsername: String = "User",
    targetAvatar: String? = null,
    isVideo: Boolean = true,
    isIncoming: Boolean = false,
    callId: String? = null,
    onNavigateBack: () -> Unit
) {
    val context = LocalContext.current
    val callState by CallManager.callState.collectAsState()
    val localVideoTrack by CallManager.localVideoTrack.collectAsState()
    val remoteVideoTrack by CallManager.remoteVideoTrack.collectAsState()
    val isMuted by CallManager.isMuted.collectAsState()
    val isCameraEnabled by CallManager.isCameraEnabled.collectAsState()
    
    // Call duration timer
    var callDuration by remember { mutableIntStateOf(0) }
    val isConnected = callState is CallManager.CallState.Connected
    
    LaunchedEffect(isConnected) {
        while (isConnected) {
            delay(1000)
            callDuration++
        }
    }
    
    // Handle call ended
    LaunchedEffect(callState) {
        when (callState) {
            is CallManager.CallState.Ended,
            is CallManager.CallState.Error -> {
                delay(2000)
                onNavigateBack()
            }
            else -> {}
        }
    }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(CallGradientStart, CallGradientEnd)
                )
            )
    ) {
        // Remote video (full screen background)
        if (isVideo && remoteVideoTrack != null) {
            WebRtcVideoView(
                videoTrack = remoteVideoTrack,
                modifier = Modifier.fillMaxSize(),
                isMirror = false
            )
        }
        
        // Local video (small PiP)
        if (isVideo && localVideoTrack != null && isConnected) {
            Box(
                modifier = Modifier
                    .padding(16.dp)
                    .size(120.dp, 160.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .align(Alignment.TopEnd)
            ) {
                WebRtcVideoView(
                    videoTrack = localVideoTrack,
                    modifier = Modifier.fillMaxSize(),
                    isMirror = true
                )
            }
        }
        
        // Call info overlay
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(60.dp))
            
            // Status text
            Text(
                text = when (callState) {
                    is CallManager.CallState.Idle -> "Initializing..."
                    is CallManager.CallState.Connecting -> "Connecting..."
                    is CallManager.CallState.Ringing -> if ((callState as CallManager.CallState.Ringing).isOutgoing) "Ringing..." else "Incoming call"
                    is CallManager.CallState.Connected -> formatDuration(callDuration)
                    is CallManager.CallState.Ended -> (callState as CallManager.CallState.Ended).reason
                    is CallManager.CallState.Error -> "Error: ${(callState as CallManager.CallState.Error).message}"
                },
                color = Color.White.copy(alpha = 0.8f),
                fontSize = 16.sp
            )
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Avatar (shown when no video or not connected)
            if (!isVideo || !isConnected || remoteVideoTrack == null) {
                AsyncImage(
                    model = targetAvatar ?: "",
                    contentDescription = "Avatar",
                    modifier = Modifier
                        .size(120.dp)
                        .clip(CircleShape)
                        .background(ControlBg)
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Text(
                    text = targetUsername,
                    color = Color.White,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold
                )
                
                Text(
                    text = if (isVideo) "Video Call" else "Voice Call",
                    color = Color.White.copy(alpha = 0.6f),
                    fontSize = 16.sp
                )
            }
            
            Spacer(modifier = Modifier.weight(1f))
            
            // Controls
            when (callState) {
                is CallManager.CallState.Ringing -> {
                    if (!(callState as CallManager.CallState.Ringing).isOutgoing) {
                        // Incoming call - show accept/reject
                        IncomingCallControls(
                            onAccept = {
                                callId?.let {
                                    CallManager.acceptCall(it, targetUserId, if (isVideo) "video" else "voice")
                                }
                            },
                            onReject = {
                                callId?.let { CallManager.rejectCall(it) }
                                onNavigateBack()
                            }
                        )
                    } else {
                        // Outgoing call - just end button
                        EndCallButton {
                            CallManager.endCall()
                        }
                    }
                }
                is CallManager.CallState.Connected,
                is CallManager.CallState.Connecting -> {
                    ActiveCallControls(
                        isVideo = isVideo,
                        isMuted = isMuted,
                        isCameraEnabled = isCameraEnabled,
                        onMuteToggle = { CallManager.toggleMute() },
                        onCameraToggle = { CallManager.toggleCamera() },
                        onSwitchCamera = { CallManager.switchCamera() },
                        onEndCall = { CallManager.endCall() }
                    )
                }
                else -> {
                    EndCallButton { onNavigateBack() }
                }
            }
            
            Spacer(modifier = Modifier.height(48.dp))
        }
    }
}

@Composable
private fun IncomingCallControls(
    onAccept: () -> Unit,
    onReject: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        // Reject
        IconButton(
            onClick = onReject,
            modifier = Modifier
                .size(72.dp)
                .background(RejectRed, CircleShape)
        ) {
            Icon(
                Icons.Default.CallEnd,
                contentDescription = "Reject",
                tint = Color.White,
                modifier = Modifier.size(36.dp)
            )
        }
        
        // Accept
        IconButton(
            onClick = onAccept,
            modifier = Modifier
                .size(72.dp)
                .background(AcceptGreen, CircleShape)
        ) {
            Icon(
                Icons.Default.Call,
                contentDescription = "Accept",
                tint = Color.White,
                modifier = Modifier.size(36.dp)
            )
        }
    }
}

@Composable
private fun ActiveCallControls(
    isVideo: Boolean,
    isMuted: Boolean,
    isCameraEnabled: Boolean,
    onMuteToggle: () -> Unit,
    onCameraToggle: () -> Unit,
    onSwitchCamera: () -> Unit,
    onEndCall: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        // Mute
        IconButton(
            onClick = onMuteToggle,
            modifier = Modifier
                .size(56.dp)
                .background(if (isMuted) RejectRed else ControlBg, CircleShape)
        ) {
            Icon(
                if (isMuted) Icons.Default.MicOff else Icons.Default.Mic,
                contentDescription = "Mute",
                tint = Color.White,
                modifier = Modifier.size(28.dp)
            )
        }
        
        // Camera toggle (video calls only)
        if (isVideo) {
            IconButton(
                onClick = onCameraToggle,
                modifier = Modifier
                    .size(56.dp)
                    .background(if (!isCameraEnabled) RejectRed else ControlBg, CircleShape)
            ) {
                Icon(
                    if (isCameraEnabled) Icons.Default.Videocam else Icons.Default.VideocamOff,
                    contentDescription = "Camera",
                    tint = Color.White,
                    modifier = Modifier.size(28.dp)
                )
            }
            
            // Switch camera
            IconButton(
                onClick = onSwitchCamera,
                modifier = Modifier
                    .size(56.dp)
                    .background(ControlBg, CircleShape)
            ) {
                Icon(
                    Icons.Default.Cameraswitch,
                    contentDescription = "Switch Camera",
                    tint = Color.White,
                    modifier = Modifier.size(28.dp)
                )
            }
        }
        
        // End call
        IconButton(
            onClick = onEndCall,
            modifier = Modifier
                .size(56.dp)
                .background(RejectRed, CircleShape)
        ) {
            Icon(
                Icons.Default.CallEnd,
                contentDescription = "End Call",
                tint = Color.White,
                modifier = Modifier.size(28.dp)
            )
        }
    }
}

@Composable
private fun EndCallButton(onClick: () -> Unit) {
    IconButton(
        onClick = onClick,
        modifier = Modifier
            .size(72.dp)
            .background(RejectRed, CircleShape)
    ) {
        Icon(
            Icons.Default.CallEnd,
            contentDescription = "End Call",
            tint = Color.White,
            modifier = Modifier.size(36.dp)
        )
    }
}

@Composable
private fun WebRtcVideoView(
    videoTrack: VideoTrack?,
    modifier: Modifier = Modifier,
    isMirror: Boolean = false
) {
    val eglContext = CallManager.getEglContext()
    
    AndroidView(
        factory = { context ->
            SurfaceViewRenderer(context).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
                eglContext?.let { init(it, null) }
                setMirror(isMirror)
                setEnableHardwareScaler(true)
            }
        },
        modifier = modifier,
        update = { renderer ->
            videoTrack?.addSink(renderer)
        }
    )
}

private fun formatDuration(seconds: Int): String {
    val mins = seconds / 60
    val secs = seconds % 60
    return "%02d:%02d".format(mins, secs)
}
