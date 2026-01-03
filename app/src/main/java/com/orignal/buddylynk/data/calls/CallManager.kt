package com.orignal.buddylynk.data.calls

import android.util.Log
import com.orignal.buddylynk.data.api.ApiService
import com.orignal.buddylynk.data.auth.AuthManager
import io.socket.client.IO
import io.socket.client.Socket
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.net.URI
import java.util.Timer
import java.util.TimerTask

enum class CallState {
    IDLE,
    DIALING,
    RINGING,
    CONNECTED,
    ENDED
}

data class CallSession(
    val userId: String,
    val username: String,
    val avatar: String?,
    val isVideo: Boolean,
    val callId: String? = null,
    val startTime: Long = 0L
)

/**
 * CallManager - Production Socket.IO Signaling for Video/Voice Calls
 * Connects to EC2 backend via CloudFront HTTPS
 */
object CallManager {
    private const val TAG = "CallManager"
    private val scope = CoroutineScope(Dispatchers.Main)
    
    // SECURITY: HTTPS via custom domain
    private const val SIGNALING_URL = "https://app.buddylynk.com"
    
    private var socket: Socket? = null
    private var isConnected = false
    
    private val _callState = MutableStateFlow(CallState.IDLE)
    val callState: StateFlow<CallState> = _callState.asStateFlow()

    private val _currentSession = MutableStateFlow<CallSession?>(null)
    val currentSession: StateFlow<CallSession?> = _currentSession.asStateFlow()

    private val _durationSeconds = MutableStateFlow(0L)
    val durationSeconds: StateFlow<Long> = _durationSeconds.asStateFlow()

    // End reason for UI feedback
    private val _endReason = MutableStateFlow<String?>(null)
    val endReason: StateFlow<String?> = _endReason.asStateFlow()
    
    // Incoming call info
    private val _incomingCall = MutableStateFlow<IncomingCallInfo?>(null)
    val incomingCall: StateFlow<IncomingCallInfo?> = _incomingCall.asStateFlow()

    private var durationTimer: Timer? = null

    data class IncomingCallInfo(
        val callId: String,
        val callerId: String,
        val callerName: String? = null,
        val callerAvatar: String? = null,
        val callType: String
    )

    /**
     * Initialize and connect to signaling server
     */
    fun connect() {
        if (socket != null && isConnected) {
            Log.d(TAG, "Already connected to signaling server")
            return
        }
        
        try {
            val options = IO.Options().apply {
                path = "/socket.io"
                reconnection = true
                reconnectionAttempts = 10
                reconnectionDelay = 1000
                timeout = 20000
            }
            
            socket = IO.socket(URI.create(SIGNALING_URL), options)
            setupSocketListeners()
            socket?.connect()
            
            Log.d(TAG, "Connecting to signaling server: $SIGNALING_URL")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create socket", e)
        }
    }
    
    private fun setupSocketListeners() {
        socket?.apply {
            on(Socket.EVENT_CONNECT) {
                Log.d(TAG, "Connected to signaling server")
                isConnected = true
                
                // Authenticate with JWT token
                val token = ApiService.getAuthToken()
                if (token != null) {
                    emit("authenticate", token)
                    Log.d(TAG, "Authenticating with token")
                } else {
                    Log.w(TAG, "No auth token available for signaling")
                }
            }
            
            on(Socket.EVENT_DISCONNECT) {
                Log.d(TAG, "Disconnected from signaling server")
                isConnected = false
            }
            
            on(Socket.EVENT_CONNECT_ERROR) { args ->
                Log.e(TAG, "Connection error: ${args.firstOrNull()}")
                isConnected = false
            }
            
            on("authenticated") { args ->
                val data = args.firstOrNull() as? JSONObject
                val success = data?.optBoolean("success", false) ?: false
                Log.d(TAG, "Authentication result: $success")
            }
            
            // Call started confirmation
            on("call:started") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                val callType = data.optString("callType")
                
                Log.d(TAG, "Call started: $callId, type: $callType")
                
                // Update session with callId
                _currentSession.value = _currentSession.value?.copy(callId = callId)
            }
            
            // Incoming call
            on("call:incoming") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                val callerId = data.optString("callerId")
                val callType = data.optString("callType")
                
                Log.d(TAG, "Incoming call from $callerId, callId: $callId")
                
                scope.launch {
                    _incomingCall.value = IncomingCallInfo(
                        callId = callId,
                        callerId = callerId,
                        callType = callType
                    )
                    _callState.value = CallState.RINGING
                }
            }
            
            // Call answered
            on("call:answered") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                val accepted = data.optBoolean("accepted", false)
                
                Log.d(TAG, "Call $callId answered: $accepted")
                
                scope.launch {
                    if (accepted) {
                        _callState.value = CallState.CONNECTED
                        startTimer()
                    } else {
                        _endReason.value = "Call Declined"
                        _callState.value = CallState.ENDED
                        delay(2000)
                        resetCallState()
                    }
                }
            }
            
            // Call connected
            on("call:connected") { args ->
                Log.d(TAG, "Call connected")
                scope.launch {
                    _callState.value = CallState.CONNECTED
                    startTimer()
                }
            }
            
            // Call ended
            on("call:ended") { args ->
                val data = args.firstOrNull() as? JSONObject
                val endedBy = data?.optString("endedBy") ?: "other"
                
                Log.d(TAG, "Call ended by $endedBy")
                scope.launch {
                    endCall(sendSignal = false)
                }
            }
            
            // Call error
            on("call:error") { args ->
                val data = args.firstOrNull() as? JSONObject
                val error = data?.optString("error") ?: "Unknown error"
                
                Log.e(TAG, "Call error: $error")
                scope.launch {
                    _endReason.value = error
                    _callState.value = CallState.ENDED
                    delay(2000)
                    resetCallState()
                }
            }
        }
    }

    /**
     * Start outgoing call
     */
    fun startCall(userId: String, username: String, avatar: String?, isVideo: Boolean) {
        if (_callState.value != CallState.IDLE) {
            Log.w(TAG, "Cannot start call - not in IDLE state")
            return
        }
        
        // Ensure connected
        if (!isConnected) {
            connect()
            scope.launch {
                delay(1000) // Wait for connection
                if (isConnected) {
                    startCall(userId, username, avatar, isVideo)
                } else {
                    _endReason.value = "Connection failed"
                    _callState.value = CallState.ENDED
                    delay(2000)
                    resetCallState()
                }
            }
            return
        }
        
        Log.d(TAG, "Starting ${if (isVideo) "video" else "voice"} call to $userId")
        
        _currentSession.value = CallSession(userId, username, avatar, isVideo)
        _callState.value = CallState.DIALING
        
        // Emit call:start event
        val callData = JSONObject().apply {
            put("targetUserId", userId)
            put("callType", if (isVideo) "video" else "voice")
        }
        socket?.emit("call:start", callData)
    }
    
    /**
     * Answer incoming call
     */
    fun answerCall(accept: Boolean) {
        val incoming = _incomingCall.value ?: return
        
        Log.d(TAG, "Answering call ${incoming.callId}: $accept")
        
        val answerData = JSONObject().apply {
            put("callId", incoming.callId)
            put("accept", accept)
        }
        socket?.emit("call:answer", answerData)
        
        if (accept) {
            _currentSession.value = CallSession(
                userId = incoming.callerId,
                username = incoming.callerName ?: incoming.callerId,
                avatar = incoming.callerAvatar,
                isVideo = incoming.callType == "video",
                callId = incoming.callId
            )
        } else {
            scope.launch {
                _incomingCall.value = null
                _callState.value = CallState.IDLE
            }
        }
    }

    /**
     * End current call
     */
    fun endCall(sendSignal: Boolean = true) {
        val session = _currentSession.value
        val incoming = _incomingCall.value
        
        if (sendSignal && (session != null || incoming != null)) {
            val callId = session?.callId ?: incoming?.callId ?: ""
            val targetId = session?.userId ?: incoming?.callerId ?: ""
            
            if (callId.isNotEmpty()) {
                val endData = JSONObject().apply {
                    put("callId", callId)
                    put("targetUserId", targetId)
                }
                socket?.emit("call:end", endData)
                Log.d(TAG, "Sent call:end signal")
            }
        }
        
        stopTimer()
        
        if (_endReason.value == null) {
            _endReason.value = null // Normal end
        }
        
        _callState.value = CallState.ENDED
        
        scope.launch {
            delay(1500)
            resetCallState()
        }
    }
    
    private fun resetCallState() {
        _callState.value = CallState.IDLE
        _currentSession.value = null
        _incomingCall.value = null
        _durationSeconds.value = 0
        _endReason.value = null
    }

    fun toggleMute() { /* Real Audio Logic - requires WebRTC audio track */ }
    fun toggleVideo() { /* Real Camera Logic - requires WebRTC video track */ }

    private fun startTimer() {
        durationTimer?.cancel()
        durationTimer = Timer()
        durationTimer?.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                _durationSeconds.value += 1
            }
        }, 1000, 1000)
    }

    private fun stopTimer() {
        durationTimer?.cancel()
        durationTimer = null
    }
    
    /**
     * Disconnect from signaling server
     */
    fun disconnect() {
        socket?.disconnect()
        socket = null
        isConnected = false
        Log.d(TAG, "Disconnected from signaling server")
    }
}
