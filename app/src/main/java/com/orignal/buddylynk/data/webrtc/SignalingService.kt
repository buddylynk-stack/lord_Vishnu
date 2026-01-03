package com.orignal.buddylynk.data.webrtc

import android.util.Log
import io.socket.client.IO
import io.socket.client.Socket
import io.socket.emitter.Emitter
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONObject
import org.webrtc.IceCandidate
import org.webrtc.SessionDescription

/**
 * Socket.io Signaling Service for WebRTC calls
 * Connects to EC2 server via CloudFront HTTPS for call signaling
 */
object SignalingService {
    private const val TAG = "SignalingService"
    // SECURITY: Using HTTPS via custom domain
    private const val SERVER_URL = "https://app.buddylynk.com"
    
    private var socket: Socket? = null
    private var authToken: String? = null
    private var listener: SignalingListener? = null
    
    private val _isConnected = MutableStateFlow(false)
    val isConnected: StateFlow<Boolean> = _isConnected
    
    interface SignalingListener {
        fun onAuthenticated(success: Boolean)
        fun onIncomingCall(callId: String, callerId: String, callType: String)
        fun onCallStarted(callId: String)
        fun onCallAnswered(callId: String, accepted: Boolean)
        fun onCallConnected(callId: String)
        fun onCallEnded(callId: String, endedBy: String)
        fun onCallError(error: String)
        fun onWebRtcOffer(callId: String, sdp: String, fromUserId: String)
        fun onWebRtcAnswer(callId: String, sdp: String, fromUserId: String)
        fun onWebRtcIceCandidate(callId: String, candidate: String, sdpMid: String?, sdpMLineIndex: Int, fromUserId: String)
    }
    
    /**
     * Initialize and connect to signaling server
     */
    fun connect(token: String, signalListener: SignalingListener) {
        authToken = token
        listener = signalListener
        
        try {
            val options = IO.Options().apply {
                forceNew = true
                reconnection = true
                reconnectionAttempts = 5
                reconnectionDelay = 1000
                timeout = 10000
            }
            
            socket = IO.socket(SERVER_URL, options)
            
            setupEventListeners()
            
            socket?.connect()
            Log.d(TAG, "Connecting to signaling server...")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect to signaling server", e)
            listener?.onCallError("Failed to connect: ${e.message}")
        }
    }
    
    private fun setupEventListeners() {
        socket?.apply {
            on(Socket.EVENT_CONNECT) {
                Log.d(TAG, "Socket connected, authenticating...")
                _isConnected.value = true
                emit("authenticate", authToken)
            }
            
            on(Socket.EVENT_DISCONNECT) {
                Log.d(TAG, "Socket disconnected")
                _isConnected.value = false
            }
            
            on(Socket.EVENT_CONNECT_ERROR) { args ->
                Log.e(TAG, "Connection error: ${args.firstOrNull()}")
                listener?.onCallError("Connection failed")
            }
            
            on("authenticated") { args ->
                val data = args.firstOrNull() as? JSONObject
                val success = data?.optBoolean("success", false) ?: false
                Log.d(TAG, "Authentication result: $success")
                listener?.onAuthenticated(success)
            }
            
            // Call events
            on("call:incoming") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                val callerId = data.optString("callerId")
                val callType = data.optString("callType")
                Log.d(TAG, "Incoming call: $callId from $callerId")
                listener?.onIncomingCall(callId, callerId, callType)
            }
            
            on("call:started") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                Log.d(TAG, "Call started: $callId")
                listener?.onCallStarted(callId)
            }
            
            on("call:answered") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                val accepted = data.optBoolean("accepted", false)
                Log.d(TAG, "Call answered: $callId, accepted: $accepted")
                listener?.onCallAnswered(callId, accepted)
            }
            
            on("call:connected") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                Log.d(TAG, "Call connected: $callId")
                listener?.onCallConnected(callId)
            }
            
            on("call:ended") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                val endedBy = data.optString("endedBy")
                Log.d(TAG, "Call ended: $callId by $endedBy")
                listener?.onCallEnded(callId, endedBy)
            }
            
            on("call:error") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val error = data.optString("error")
                Log.e(TAG, "Call error: $error")
                listener?.onCallError(error)
            }
            
            // WebRTC signaling events
            on("webrtc:offer") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                val sdp = data.optString("sdp")
                val fromUserId = data.optString("fromUserId")
                Log.d(TAG, "Received WebRTC offer from $fromUserId")
                listener?.onWebRtcOffer(callId, sdp, fromUserId)
            }
            
            on("webrtc:answer") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                val sdp = data.optString("sdp")
                val fromUserId = data.optString("fromUserId")
                Log.d(TAG, "Received WebRTC answer from $fromUserId")
                listener?.onWebRtcAnswer(callId, sdp, fromUserId)
            }
            
            on("webrtc:ice-candidate") { args ->
                val data = args.firstOrNull() as? JSONObject ?: return@on
                val callId = data.optString("callId")
                val candidateObj = data.optJSONObject("candidate") ?: return@on
                val candidate = candidateObj.optString("candidate")
                val sdpMid = candidateObj.optString("sdpMid")
                val sdpMLineIndex = candidateObj.optInt("sdpMLineIndex")
                val fromUserId = data.optString("fromUserId")
                listener?.onWebRtcIceCandidate(callId, candidate, sdpMid, sdpMLineIndex, fromUserId)
            }
        }
    }
    
    /**
     * Start a call to target user
     */
    fun startCall(targetUserId: String, callType: String) {
        val data = JSONObject().apply {
            put("targetUserId", targetUserId)
            put("callType", callType) // "voice" or "video"
        }
        socket?.emit("call:start", data)
        Log.d(TAG, "Starting $callType call to $targetUserId")
    }
    
    /**
     * Answer or reject an incoming call
     */
    fun answerCall(callId: String, accept: Boolean) {
        val data = JSONObject().apply {
            put("callId", callId)
            put("accept", accept)
        }
        socket?.emit("call:answer", data)
        Log.d(TAG, "Answering call $callId: $accept")
    }
    
    /**
     * End an active call
     */
    fun endCall(callId: String, targetUserId: String) {
        val data = JSONObject().apply {
            put("callId", callId)
            put("targetUserId", targetUserId)
        }
        socket?.emit("call:end", data)
        Log.d(TAG, "Ending call $callId")
    }
    
    /**
     * Send WebRTC SDP offer
     */
    fun sendOffer(callId: String, targetUserId: String, sdp: SessionDescription) {
        val data = JSONObject().apply {
            put("callId", callId)
            put("targetUserId", targetUserId)
            put("sdp", sdp.description)
        }
        socket?.emit("webrtc:offer", data)
    }
    
    /**
     * Send WebRTC SDP answer
     */
    fun sendAnswer(callId: String, targetUserId: String, sdp: SessionDescription) {
        val data = JSONObject().apply {
            put("callId", callId)
            put("targetUserId", targetUserId)
            put("sdp", sdp.description)
        }
        socket?.emit("webrtc:answer", data)
    }
    
    /**
     * Send ICE candidate
     */
    fun sendIceCandidate(callId: String, targetUserId: String, candidate: IceCandidate) {
        val candidateObj = JSONObject().apply {
            put("candidate", candidate.sdp)
            put("sdpMid", candidate.sdpMid)
            put("sdpMLineIndex", candidate.sdpMLineIndex)
        }
        val data = JSONObject().apply {
            put("callId", callId)
            put("targetUserId", targetUserId)
            put("candidate", candidateObj)
        }
        socket?.emit("webrtc:ice-candidate", data)
    }
    
    /**
     * Disconnect from signaling server
     */
    fun disconnect() {
        socket?.disconnect()
        socket = null
        _isConnected.value = false
        Log.d(TAG, "Disconnected from signaling server")
    }
}
