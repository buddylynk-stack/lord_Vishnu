package com.orignal.buddylynk.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.live.LiveStreamService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class LiveViewModel : ViewModel() {

    private val _isStreaming = MutableStateFlow(false)
    val isStreaming: StateFlow<Boolean> = _isStreaming.asStateFlow()

    private val _viewerCount = MutableStateFlow(0)
    val viewerCount: StateFlow<Int> = _viewerCount.asStateFlow()

    private val _streamDuration = MutableStateFlow(0L)
    val streamDuration: StateFlow<Long> = _streamDuration.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    fun startStream(title: String, category: String) {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                val result = LiveStreamService.startStream(title, category)
                result.fold(
                    onSuccess = {
                        _isStreaming.value = true
                        startMonitoring()
                    },
                    onFailure = {
                        _error.value = "Failed to start stream: ${it.message}"
                    }
                )
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun stopStream() {
        viewModelScope.launch {
            _isLoading.value = true
            LiveStreamService.stopStream()
            _isStreaming.value = false
            _isLoading.value = false
        }
    }

    private fun startMonitoring() {
        viewModelScope.launch {
            LiveStreamService.observeViewerCount().collect { count ->
                _viewerCount.value = count
            }
        }
    }
}
