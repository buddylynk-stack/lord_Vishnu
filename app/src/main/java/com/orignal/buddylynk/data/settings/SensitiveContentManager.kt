package com.orignal.buddylynk.data.settings

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * SensitiveContentManager - Manages sensitive content display settings
 * 
 * Options:
 * - SHOW: Display all content without any blur
 * - BLUR: Show content with blur overlay, tap to reveal
 * - HIDE: Completely hide sensitive content
 */
object SensitiveContentManager {
    
    private const val PREFS_NAME = "sensitive_content_prefs"
    private const val KEY_MODE = "sensitive_content_mode"
    
    enum class ContentMode {
        SHOW,   // Show all content
        BLUR,   // Blur sensitive content
        HIDE    // Hide sensitive content
    }
    
    private var prefs: SharedPreferences? = null
    
    private val _contentMode = MutableStateFlow(ContentMode.BLUR)
    val contentMode: StateFlow<ContentMode> = _contentMode.asStateFlow()
    
    /**
     * Initialize with context (call from Application or MainActivity)
     */
    fun init(context: Context) {
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        loadMode()
    }
    
    /**
     * Load saved mode from preferences
     */
    private fun loadMode() {
        val savedMode = prefs?.getString(KEY_MODE, ContentMode.BLUR.name) ?: ContentMode.BLUR.name
        _contentMode.value = try {
            ContentMode.valueOf(savedMode)
        } catch (e: Exception) {
            ContentMode.BLUR
        }
    }
    
    /**
     * Set content mode and save to preferences
     */
    fun setMode(mode: ContentMode) {
        _contentMode.value = mode
        prefs?.edit()?.putString(KEY_MODE, mode.name)?.apply()
    }
    
    /**
     * Check if content should be blurred
     */
    fun shouldBlur(): Boolean = _contentMode.value == ContentMode.BLUR
    
    /**
     * Check if content should be hidden
     */
    fun shouldHide(): Boolean = _contentMode.value == ContentMode.HIDE
    
    /**
     * Check if content should be shown normally
     */
    fun shouldShow(): Boolean = _contentMode.value == ContentMode.SHOW
}
