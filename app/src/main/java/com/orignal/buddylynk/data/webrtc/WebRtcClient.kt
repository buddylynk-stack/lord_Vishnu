package com.orignal.buddylynk.data.webrtc

import android.content.Context
import android.util.Log
import org.webrtc.*
import java.util.concurrent.Executors

/**
 * WebRTC Client - Manages peer connections for video/voice calls
 * Industry-ready implementation using Google's WebRTC library
 */
class WebRtcClient(
    private val context: Context,
    private val listener: WebRtcListener
) {
    companion object {
        private const val TAG = "WebRtcClient"
        
        // Google's free STUN servers for NAT traversal
        private val ICE_SERVERS = listOf(
            PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer(),
            PeerConnection.IceServer.builder("stun:stun1.l.google.com:19302").createIceServer(),
            PeerConnection.IceServer.builder("stun:stun2.l.google.com:19302").createIceServer(),
            PeerConnection.IceServer.builder("stun:stun3.l.google.com:19302").createIceServer()
        )
    }

    private val executor = Executors.newSingleThreadExecutor()
    private var peerConnectionFactory: PeerConnectionFactory? = null
    private var peerConnection: PeerConnection? = null
    private var localVideoTrack: VideoTrack? = null
    private var localAudioTrack: AudioTrack? = null
    private var videoCapturer: VideoCapturer? = null
    private var surfaceTextureHelper: SurfaceTextureHelper? = null
    private var localVideoSource: VideoSource? = null
    private var eglBase: EglBase? = null

    interface WebRtcListener {
        fun onLocalStream(videoTrack: VideoTrack?, audioTrack: AudioTrack?)
        fun onRemoteStream(videoTrack: VideoTrack?, audioTrack: AudioTrack?)
        fun onIceCandidate(candidate: IceCandidate)
        fun onIceConnectionChange(state: PeerConnection.IceConnectionState)
        fun onError(error: String)
    }

    /**
     * Initialize WebRTC - must be called before any other methods
     */
    fun initialize() {
        executor.execute {
            try {
                eglBase = EglBase.create()
                
                val options = PeerConnectionFactory.InitializationOptions.builder(context)
                    .setEnableInternalTracer(false)
                    .createInitializationOptions()
                PeerConnectionFactory.initialize(options)

                val encoderFactory = DefaultVideoEncoderFactory(
                    eglBase!!.eglBaseContext, true, true
                )
                val decoderFactory = DefaultVideoDecoderFactory(eglBase!!.eglBaseContext)

                peerConnectionFactory = PeerConnectionFactory.builder()
                    .setVideoEncoderFactory(encoderFactory)
                    .setVideoDecoderFactory(decoderFactory)
                    .createPeerConnectionFactory()

                Log.d(TAG, "WebRTC initialized successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize WebRTC", e)
                listener.onError("Failed to initialize WebRTC: ${e.message}")
            }
        }
    }

    /**
     * Start local video/audio capture
     */
    fun startLocalMedia(isVideoCall: Boolean) {
        executor.execute {
            try {
                val factory = peerConnectionFactory ?: throw Exception("Not initialized")

                // Audio
                val audioConstraints = MediaConstraints()
                val audioSource = factory.createAudioSource(audioConstraints)
                localAudioTrack = factory.createAudioTrack("audio0", audioSource)

                // Video (only for video calls)
                if (isVideoCall) {
                    videoCapturer = createCameraCapturer()
                    videoCapturer?.let { capturer ->
                        surfaceTextureHelper = SurfaceTextureHelper.create(
                            "CaptureThread", eglBase!!.eglBaseContext
                        )
                        localVideoSource = factory.createVideoSource(capturer.isScreencast)
                        capturer.initialize(surfaceTextureHelper, context, localVideoSource!!.capturerObserver)
                        capturer.startCapture(1280, 720, 30)
                        
                        localVideoTrack = factory.createVideoTrack("video0", localVideoSource)
                    }
                }

                listener.onLocalStream(localVideoTrack, localAudioTrack)
                Log.d(TAG, "Local media started - video: $isVideoCall")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start local media", e)
                listener.onError("Failed to start camera/microphone: ${e.message}")
            }
        }
    }

    /**
     * Create peer connection
     */
    fun createPeerConnection() {
        executor.execute {
            try {
                val factory = peerConnectionFactory ?: throw Exception("Not initialized")

                val rtcConfig = PeerConnection.RTCConfiguration(ICE_SERVERS).apply {
                    sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN
                    continualGatheringPolicy = PeerConnection.ContinualGatheringPolicy.GATHER_CONTINUALLY
                }

                peerConnection = factory.createPeerConnection(rtcConfig, object : PeerConnection.Observer {
                    override fun onSignalingChange(state: PeerConnection.SignalingState?) {
                        Log.d(TAG, "Signaling state: $state")
                    }

                    override fun onIceConnectionChange(state: PeerConnection.IceConnectionState?) {
                        Log.d(TAG, "ICE connection state: $state")
                        state?.let { listener.onIceConnectionChange(it) }
                    }

                    override fun onIceConnectionReceivingChange(receiving: Boolean) {}

                    override fun onIceGatheringChange(state: PeerConnection.IceGatheringState?) {
                        Log.d(TAG, "ICE gathering state: $state")
                    }

                    override fun onIceCandidate(candidate: IceCandidate?) {
                        candidate?.let { listener.onIceCandidate(it) }
                    }

                    override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>?) {}

                    override fun onAddStream(stream: MediaStream?) {
                        Log.d(TAG, "Remote stream added")
                        stream?.let {
                            val videoTrack = it.videoTracks.firstOrNull()
                            val audioTrack = it.audioTracks.firstOrNull()
                            listener.onRemoteStream(videoTrack, audioTrack)
                        }
                    }

                    override fun onRemoveStream(stream: MediaStream?) {
                        Log.d(TAG, "Remote stream removed")
                    }

                    override fun onDataChannel(channel: DataChannel?) {}
                    override fun onRenegotiationNeeded() {}
                    override fun onAddTrack(receiver: RtpReceiver?, streams: Array<out MediaStream>?) {
                        Log.d(TAG, "Track added")
                    }
                    override fun onTrack(transceiver: RtpTransceiver?) {}
                })

                // Add local tracks to peer connection
                localAudioTrack?.let { peerConnection?.addTrack(it) }
                localVideoTrack?.let { peerConnection?.addTrack(it) }

                Log.d(TAG, "Peer connection created")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create peer connection", e)
                listener.onError("Failed to create connection: ${e.message}")
            }
        }
    }

    /**
     * Create and set local SDP offer
     */
    fun createOffer(callback: (SessionDescription) -> Unit) {
        executor.execute {
            val constraints = MediaConstraints().apply {
                mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveAudio", "true"))
                mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "true"))
            }

            peerConnection?.createOffer(object : SdpObserver {
                override fun onCreateSuccess(sdp: SessionDescription?) {
                    sdp?.let {
                        peerConnection?.setLocalDescription(SimpleSdpObserver(), it)
                        callback(it)
                    }
                }
                override fun onSetSuccess() {}
                override fun onCreateFailure(error: String?) {
                    listener.onError("Failed to create offer: $error")
                }
                override fun onSetFailure(error: String?) {}
            }, constraints)
        }
    }

    /**
     * Create and set local SDP answer
     */
    fun createAnswer(callback: (SessionDescription) -> Unit) {
        executor.execute {
            val constraints = MediaConstraints().apply {
                mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveAudio", "true"))
                mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "true"))
            }

            peerConnection?.createAnswer(object : SdpObserver {
                override fun onCreateSuccess(sdp: SessionDescription?) {
                    sdp?.let {
                        peerConnection?.setLocalDescription(SimpleSdpObserver(), it)
                        callback(it)
                    }
                }
                override fun onSetSuccess() {}
                override fun onCreateFailure(error: String?) {
                    listener.onError("Failed to create answer: $error")
                }
                override fun onSetFailure(error: String?) {}
            }, constraints)
        }
    }

    /**
     * Set remote SDP (offer or answer from peer)
     */
    fun setRemoteSdp(sdp: SessionDescription) {
        executor.execute {
            peerConnection?.setRemoteDescription(SimpleSdpObserver(), sdp)
            Log.d(TAG, "Remote SDP set: ${sdp.type}")
        }
    }

    /**
     * Add ICE candidate from remote peer
     */
    fun addIceCandidate(candidate: IceCandidate) {
        executor.execute {
            peerConnection?.addIceCandidate(candidate)
        }
    }

    /**
     * Toggle microphone mute
     */
    fun setMicEnabled(enabled: Boolean) {
        localAudioTrack?.setEnabled(enabled)
    }

    /**
     * Toggle camera
     */
    fun setCameraEnabled(enabled: Boolean) {
        localVideoTrack?.setEnabled(enabled)
    }

    /**
     * Switch between front and back camera
     */
    fun switchCamera() {
        (videoCapturer as? CameraVideoCapturer)?.switchCamera(null)
    }

    /**
     * End call and release resources
     */
    fun endCall() {
        executor.execute {
            try {
                videoCapturer?.stopCapture()
                videoCapturer?.dispose()
                videoCapturer = null

                localVideoTrack?.dispose()
                localAudioTrack?.dispose()
                localVideoSource?.dispose()
                surfaceTextureHelper?.dispose()

                peerConnection?.close()
                peerConnection = null

                Log.d(TAG, "Call ended, resources released")
            } catch (e: Exception) {
                Log.e(TAG, "Error ending call", e)
            }
        }
    }

    /**
     * Fully dispose WebRTC
     */
    fun dispose() {
        endCall()
        executor.execute {
            peerConnectionFactory?.dispose()
            peerConnectionFactory = null
            eglBase?.release()
            eglBase = null
        }
    }

    /**
     * Get EGL context for video rendering
     */
    fun getEglContext() = eglBase?.eglBaseContext

    private fun createCameraCapturer(): VideoCapturer? {
        val cameraEnumerator = Camera2Enumerator(context)
        
        // Try front camera first
        for (deviceName in cameraEnumerator.deviceNames) {
            if (cameraEnumerator.isFrontFacing(deviceName)) {
                return cameraEnumerator.createCapturer(deviceName, null)
            }
        }
        
        // Fall back to any camera
        for (deviceName in cameraEnumerator.deviceNames) {
            return cameraEnumerator.createCapturer(deviceName, null)
        }
        
        return null
    }

    private class SimpleSdpObserver : SdpObserver {
        override fun onCreateSuccess(sdp: SessionDescription?) {}
        override fun onSetSuccess() {}
        override fun onCreateFailure(error: String?) {}
        override fun onSetFailure(error: String?) {}
    }
}
