package com.orignal.buddylynk.ui.viewmodel

import android.content.Context
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.data.aws.S3Service
import com.orignal.buddylynk.data.model.Post
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ImageFilter(
    val name: String,
    val saturation: Float = 1f,
    val contrast: Float = 1f,
    val brightness: Float = 1f,
    val grayscale: Boolean = false,
    val sepia: Float = 0f
)

data class ImageAdjustments(
    val brightness: Float = 1f,
    val contrast: Float = 1f,
    val saturation: Float = 1f
)

enum class UploadMode { FILES, CAMERA, EVENT }
enum class EditTab { FILTER, EDIT }
enum class PostState { IDLE, UPLOADING, SUCCESS, ERROR }

class CreatePostViewModel : ViewModel() {

    // Filters
    val filters = listOf(
        ImageFilter("Normal"),
        ImageFilter("Vivid", saturation = 1.5f, contrast = 1.1f),
        ImageFilter("Noir", grayscale = true, contrast = 1.25f, brightness = 0.9f),
        ImageFilter("Vintage", sepia = 0.5f, contrast = 0.9f, brightness = 1.1f),
        ImageFilter("Glacial", saturation = 0.8f, brightness = 1.05f),
        ImageFilter("Golden", sepia = 0.3f, saturation = 1.2f),
        ImageFilter("Drama", contrast = 1.25f, saturation = 1.1f, brightness = 0.9f)
    )

    // State
    private val _selectedImageUri = MutableStateFlow<Uri?>(null)
    val selectedImageUri: StateFlow<Uri?> = _selectedImageUri.asStateFlow()

    private val _caption = MutableStateFlow("")
    val caption: StateFlow<String> = _caption.asStateFlow()

    private val _location = MutableStateFlow("")
    val location: StateFlow<String> = _location.asStateFlow()

    private val _selectedFilter = MutableStateFlow(filters[0])
    val selectedFilter: StateFlow<ImageFilter> = _selectedFilter.asStateFlow()

    private val _adjustments = MutableStateFlow(ImageAdjustments())
    val adjustments: StateFlow<ImageAdjustments> = _adjustments.asStateFlow()

    private val _uploadMode = MutableStateFlow(UploadMode.FILES)
    val uploadMode: StateFlow<UploadMode> = _uploadMode.asStateFlow()

    private val _editTab = MutableStateFlow(EditTab.FILTER)
    val editTab: StateFlow<EditTab> = _editTab.asStateFlow()

    private val _postState = MutableStateFlow(PostState.IDLE)
    val postState: StateFlow<PostState> = _postState.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    // Actions
    fun setImageUri(uri: Uri?) {
        _selectedImageUri.value = uri
    }

    fun setCaption(text: String) {
        _caption.value = text
    }

    fun setLocation(loc: String) {
        _location.value = loc
    }

    fun setFilter(filter: ImageFilter) {
        _selectedFilter.value = filter
    }

    fun setAdjustments(adj: ImageAdjustments) {
        _adjustments.value = adj
    }

    fun setBrightness(value: Float) {
        _adjustments.value = _adjustments.value.copy(brightness = value)
    }

    fun setContrast(value: Float) {
        _adjustments.value = _adjustments.value.copy(contrast = value)
    }

    fun setSaturation(value: Float) {
        _adjustments.value = _adjustments.value.copy(saturation = value)
    }

    fun setUploadMode(mode: UploadMode) {
        _uploadMode.value = mode
    }

    fun setEditTab(tab: EditTab) {
        _editTab.value = tab
    }

    fun clearSelection() {
        _selectedImageUri.value = null
        _caption.value = ""
        _location.value = ""
        _selectedFilter.value = filters[0]
        _adjustments.value = ImageAdjustments()
        _postState.value = PostState.IDLE
        _errorMessage.value = null
    }

    /**
     * Upload and Create Post
     */
    fun createPost(context: Context) {
        val imageUri = _selectedImageUri.value ?: return
        val user = AuthManager.currentUser.value ?: return

        viewModelScope.launch {
            _postState.value = PostState.UPLOADING
            _errorMessage.value = null

            try {
                val postId = "post_${System.currentTimeMillis()}"

                // 1. Upload image to S3
                val mediaUrl = S3Service.uploadPostMedia(context, postId, imageUri)
                if (mediaUrl == null) {
                    _postState.value = PostState.ERROR
                    _errorMessage.value = "Failed to upload image"
                    return@launch
                }

                // 2. Create post via API
                val dateFormat = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", java.util.Locale.US)
                dateFormat.timeZone = java.util.TimeZone.getTimeZone("UTC")
                val createdAtISO = dateFormat.format(java.util.Date())
                
                val post = Post(
                    postId = postId,
                    userId = user.userId,
                    username = user.username,
                    userAvatar = user.avatar,
                    content = _caption.value.ifBlank { "📸" },
                    mediaUrl = mediaUrl,
                    mediaType = "image",
                    createdAt = createdAtISO,
                    likesCount = 0,
                    commentsCount = 0,
                    viewsCount = 0
                )
                
                val success = BackendRepository.createPost(post)
                if (success) {
                    _postState.value = PostState.SUCCESS
                } else {
                    _postState.value = PostState.ERROR
                    _errorMessage.value = "Failed to create post"
                }
            } catch (e: Exception) {
                _postState.value = PostState.ERROR
                _errorMessage.value = e.message ?: "Unknown error"
            }
        }
    }
}
