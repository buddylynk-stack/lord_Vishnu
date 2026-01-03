package com.orignal.buddylynk.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.comments.CommentsService
import com.orignal.buddylynk.data.model.Comment
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel for CommentsScreen
 */
class CommentsViewModel : ViewModel() {
    
    private val _comments = MutableStateFlow<List<Comment>>(emptyList())
    val comments: StateFlow<List<Comment>> = _comments.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _isSending = MutableStateFlow(false)
    val isSending: StateFlow<Boolean> = _isSending.asStateFlow()
    
    private val _replyingTo = MutableStateFlow<Comment?>(null)
    val replyingTo: StateFlow<Comment?> = _replyingTo.asStateFlow()
    
    private var currentPostId: String = ""
    
    /**
     * Load comments for a post
     */
    fun loadComments(postId: String) {
        currentPostId = postId
        viewModelScope.launch {
            _isLoading.value = true
            try {
                _comments.value = CommentsService.getComments(postId)
            } catch (e: Exception) {
                // Handle error
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    /**
     * Post a new comment or reply
     */
    fun postComment(postId: String, content: String) {
        viewModelScope.launch {
            _isSending.value = true
            try {
                val parentId = _replyingTo.value?.commentId
                val comment = CommentsService.createComment(postId, content, parentId)
                
                if (comment != null) {
                    _comments.value = _comments.value + comment
                    _replyingTo.value = null
                }
            } catch (e: Exception) {
                // Handle error
            } finally {
                _isSending.value = false
            }
        }
    }
    
    /**
     * Toggle like on a comment
     */
    fun toggleLike(commentId: String) {
        viewModelScope.launch {
            val comment = _comments.value.find { it.commentId == commentId } ?: return@launch
            
            // Optimistic update
            _comments.value = _comments.value.map {
                if (it.commentId == commentId) {
                    it.copy(
                        isLiked = !it.isLiked,
                        likesCount = if (it.isLiked) it.likesCount - 1 else it.likesCount + 1
                    )
                } else it
            }
            
            // Sync with server
            try {
                if (comment.isLiked) {
                    CommentsService.unlikeComment(commentId)
                } else {
                    CommentsService.likeComment(commentId)
                }
            } catch (e: Exception) {
                // Revert on failure
                _comments.value = _comments.value.map {
                    if (it.commentId == commentId) comment else it
                }
            }
        }
    }
    
    /**
     * Set replying to a comment
     */
    fun setReplyingTo(comment: Comment) {
        _replyingTo.value = comment
    }
    
    /**
     * Cancel reply
     */
    fun cancelReply() {
        _replyingTo.value = null
    }
    
    /**
     * Delete a comment
     */
    fun deleteComment(commentId: String) {
        viewModelScope.launch {
            try {
                if (CommentsService.deleteComment(commentId, currentPostId)) {
                    _comments.value = _comments.value.filter { it.commentId != commentId }
                }
            } catch (e: Exception) {
                // Handle error
            }
        }
    }
    
    /**
     * Toggle pin on a comment (post owner only)
     */
    fun togglePin(commentId: String) {
        viewModelScope.launch {
            _comments.value = _comments.value.map { comment ->
                if (comment.commentId == commentId) {
                    comment.copy(isPinned = !comment.isPinned)
                } else if (comment.isPinned && comment.commentId != commentId) {
                    // Only one pinned comment at a time
                    comment.copy(isPinned = false)
                } else {
                    comment
                }
            }
        }
    }
    
    /**
     * Edit a comment
     */
    fun editComment(commentId: String, newContent: String) {
        viewModelScope.launch {
            _isSending.value = true
            try {
                _comments.value = _comments.value.map { comment ->
                    if (comment.commentId == commentId) {
                        comment.copy(
                            content = newContent,
                            isEdited = true
                        )
                    } else comment
                }
            } catch (e: Exception) {
                // Handle error
            } finally {
                _isSending.value = false
            }
        }
    }
    
    /**
     * Refresh comments
     */
    fun refresh() {
        if (currentPostId.isNotBlank()) {
            loadComments(currentPostId)
        }
    }
}
