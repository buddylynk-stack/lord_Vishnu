package com.orignal.buddylynk.data.live

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
/**
 * Service to manage live streaming lifecycle and interactions.
 * Currently simulates backend behavior.
 */
object LiveStreamService {

    private var isStreaming = false
    private var currentStreamId: String? = null

    suspend fun startStream(title: String, category: String): Result<StreamInfo> {
        delay(1500) // Simulate network latency
        isStreaming = true
        currentStreamId = "stream_${System.currentTimeMillis()}"
        
        return Result.success(
            StreamInfo(
                streamId = currentStreamId!!,
                wsUrl = "wss://mock-stream.buddylynk.com/chat",
                rtmpUrl = "rtmps://live.buddylynk.com/app",
                streamKey = "live_${System.currentTimeMillis()}_key"
            )
        )
    }

    suspend fun stopStream() {
        if (!isStreaming) return
        delay(1000)
        isStreaming = false
        currentStreamId = null
    }

    /**
     * Simulates receiving viewer count updates
     */
    fun observeViewerCount(): Flow<Int> = flow {
        var count = 0
        while (isStreaming) {
            emit(count)
            delay(3000)
            // Random fluctuation
            val delta = (1..5).random()
            if ((0..10).random() > 3) count += delta else count = maxOf(0, count - delta)
            if (count > 1000) count = 1000 // Cap for mock
        }
        emit(0)
    }

    data class StreamInfo(
        val streamId: String,
        val wsUrl: String,
        val rtmpUrl: String,
        val streamKey: String
    )
}
