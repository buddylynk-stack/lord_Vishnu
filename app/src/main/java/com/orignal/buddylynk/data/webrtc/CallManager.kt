package com.orignal.buddylynk.data.webrtc

import android.content.Context
import android.util.Log
import com.orignal.buddylynk.data.api.ApiService
import com.orignal.buddylynk.data.auth.AuthManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.webrtc.*

/**
 * Call Manager - High-level orchestration for video/voice calls
 * Coordinates between WebRTC client and signaling service
 */
object CallManager : SignalingService.SignalingListener, WebRtcClient.WebRtcListener {
    private const val TAG = "CallManager"
    private val scope = CoroutineScope(Dispatchers.Main)
    
    private var webRtcClient: WebRtcClient? = null
    private var context: Context? = null
    
    // Current call state
    private val _callState = MutableStateFlow<CallState>(CallState.Idle)
    val callState: StateFlow<CallState> = _callState
    
    private val _localVideoTrack = MutableStateFlow<VideoTrack?>(null)
    val localVideoTrack: StateFlow<VideoTrack?> = _localVideoTrack
    
    private val _remoteVideoTrack = MutableStateFlow<VideoTrack?>(null)
    val remoteVideoTrack: StateFlow<VideoTrack?> = _remoteVideoTrack
    
    private val _isMuted = MutableStateFlow(false)
    val isMuted: StateFlow<Boolean> = _isMuted
    
    private val _isCameraEnabled = MutableStateFlow(true)
    val isCameraEnabled: StateFlow<Boolean> = _isCameraEnabled
    
    private val _isSpeakerOn = MutableStateFlow(true)
    val isSpeakerOn: StateFlow<Boolean> = _isSpeakerOn
    
    // Current call info
    private var currentCallId: String? = null
    private var currentTargetUserId: String? = null
    private var currentCallType: String = "voice"
    
    sealed class CallState {
        object Idle : CallState()
        object Connecting : CallState()
        data class Ringing(val isOutgoing: Boolean, val callerId: String, val callType: String) : CallState()
        object Connected : CallState()
        data class Ended(val reason: String) : CallState()
        data class Error(val message: String) : CallState()
    }
    
    /**
     * Initialize call manager
     */
    fun initialize(ctx: Context) {
        context = ctx.applicationContext
        
        // Connect to signaling server
        val token = ApiService.getAuthToken()
        if (token != null) {
            SignalingService.connect(token, this)
        } else {
            Log.w(TAG, "No auth token, cannot connect to signaling")
        }
    }
    
    /**
     * Start outgoing call
     */
    fun startCall(targetUserId: String, isVideo: Boolean) {
        if (_callState.value !is CallState.Idle) {
            Log.w(TAG, "Cannot start call - not idle")
            return
        }
        
        currentTargetUserId = targetUserId
        currentCallType = if (isVideo) "video" else "voice"
        _callState.value = CallState.Connecting
        
        // Initialize WebRTC
        context?.let { ctx ->
            webRtcClient = WebRtcClient(ctx, this)
            webRtcClient?.initialize()
            
            // Start local media
            scope.launch {
                kotlinx.coroutines.delay(500) // Wait for WebRTC init
                webRtcClient?.startLocalMedia(isVideo)
                
                // Start call via signaling
                SignalingService.startCall(targetUserId, currentCallType)
            }
        }
    }
    
    /**
     * Accept incoming call
     */
    fun acceptCall(callId: String, callerId: String, callType: String) {
        currentCallId = callId
        currentTargetUserId = callerId
        currentCallType = callType
        _callState.value = CallState.Connecting
        
        // Initialize WebRTC
        context?.let { ctx ->
            webRtcClient = WebRtcClient(ctx, this)
            webRtcClient?.initialize()
            
            scope.launch {
                kotlinx.coroutines.delay(500) // Wait for WebRTC init
                webRtcClient?.startLocalMedia(callType == "video")
                webRtcClient?.createPeerConnection()
                
                // Accept via signaling
                SignalingService.answerCall(callId, true)
            }
        }
    }
    
    /**
     * Reject incoming call
     */
    fun rejectCall(callId: String) {
        SignalingService.answerCall(callId, false)
        _callState.value = CallState.Idle
    }
    
    /**
     * End current call
     */
    fun endCall() {
        currentCallId?.let { callId ->
            currentTargetUserId?.let { targetUserId ->
                SignalingService.endCall(callId, targetUserId)
            }
        }
        
        cleanup()
        _callState.value = CallState.Ended("Call ended")
    }
    
    /**
     * Toggle microphone mute
     */
    fun toggleMute() {
        _isMuted.value = !_isMuted.value
        webRtcClient?.setMicEnabled(!_isMuted.value)
    }
    
    /**
     * Toggle camera
     */
    fun toggleCamera() {
        _isCameraEnabled.value = !_isCameraEnabled.value
        webRtcClient?.setCameraEnabled(_isCameraEnabled.value)
    }
    
    /**
     * Switch front/back camera
     */
    fun switchCamera() {
        webRtcClient?.switchCamera()
    }
    
    /**
     * Get EGL context for video rendering
     */
    fun getEglContext() = webRtcClient?.getEglContext()
    
    private fun cleanup() {
        webRtcClient?.endCall()
        webRtcClient = null
        currentCallId = null
        currentTargetUserId = null
        _localVideoTrack.value = null
        _remoteVideoTrack.value = null
        _isMuted.value = false
        _isCameraEnabled.value = true
    }
    
    // ============= SIGNALING CALLBACKS =============
    
    override fun onAuthenticated(success: Boolean) {
        Log.d(TAG, "Signaling authenticated: $success")
    }
    
    override fun onIncomingCall(callId: String, callerId: String, callType: String) {
        currentCallId = callId
        _callState.value = CallState.Ringing(isOutgoing = false, callerId = callerId, callType = callType)
    }
    
    override fun onCallStarted(callId: String) {
        currentCallId = callId
        _callState.value = CallState.Ringing(isOutgoing = true, callerId = currentTargetUserId ?: "", callType = currentCallType)
        
        // Create peer connection and offer
        webRtcClient?.createPeerConnection()
        scope.launch {
            kotlinx.coroutines.delay(300)
            webRtcClient?.createOffer { sdp ->
                currentTargetUserId?.let { targetId ->
                    SignalingService.sendOffer(callId, targetId, sdp)
                }
            }
        }
    }
    
    override fun onCallAnswered(callId: String, accepted: Boolean) {
        if (accepted) {
            _callState.value = CallState.Connected
        } else {
            cleanup()
            _callState.value = CallState.Ended("Call rejected")
        }
    }
    
    override fun onCallConnected(callId: String) {
        _callState.value = CallState.Connected
    }
    
    override fun onCallEnded(callId: String, endedBy: String) {
        cleanup()
        _callState.value = CallState.Ended("Call ended by other party")
    }
    
    override fun onCallError(error: String) {
        cleanup()
        _callState.value = CallState.Error(error)
    }
    
    override fun onWebRtcOffer(callId: String, sdp: String, fromUserId: String) {
        val sessionDescription = SessionDescription(SessionDescription.Type.OFFER, sdp)
        webRtcClient?.setRemoteSdp(sessionDescription)
        
        // Create answer
        webRtcClient?.createAnswer { answer ->
            SignalingService.sendAnswer(callId, fromUserId, answer)
        }
    }
    
    override fun onWebRtcAnswer(callId: String, sdp: String, fromUserId: String) {
        val sessionDescription = SessionDescription(SessionDescription.Type.ANSWER, sdp)
        webRtcClient?.setRemoteSdp(sessionDescription)
        _callState.value = CallState.Connected
    }
    
    override fun onWebRtcIceCandidate(callId: String, candidate: String, sdpMid: String?, sdpMLineIndex: Int, fromUserId: String) {
        val iceCandidate = IceCandidate(sdpMid, sdpMLineIndex, candidate)
        webRtcClient?.addIceCandidate(iceCandidate)
    }
    
    // ============= WEBRTC CALLBACKS =============
    
    override fun onLocalStream(videoTrack: VideoTrack?, audioTrack: AudioTrack?) {
        _localVideoTrack.value = videoTrack
    }
    
    override fun onRemoteStream(videoTrack: VideoTrack?, audioTrack: AudioTrack?) {
        _remoteVideoTrack.value = videoTrack
    }
    
    override fun onIceCandidate(candidate: IceCandidate) {
        currentCallId?.let { callId ->
            currentTargetUserId?.let { targetId ->
                SignalingService.sendIceCandidate(callId, targetId, candidate)
            }
        }
    }
    
    override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {
        Log.d(TAG, "ICE state: $state")
        when (state) {
            PeerConnection.IceConnectionState.CONNECTED -> {
                _callState.value = CallState.Connected
            }
            PeerConnection.IceConnectionState.DISCONNECTED,
            PeerConnection.IceConnectionState.FAILED -> {
                cleanup()
                _callState.value = CallState.Ended("Connection lost")
            }
            else -> {}
        }
    }
    
    override fun onError(error: String) {
        _callState.value = CallState.Error(error)
    }
}
