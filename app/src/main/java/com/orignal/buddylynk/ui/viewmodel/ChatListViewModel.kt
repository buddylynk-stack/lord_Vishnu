package com.orignal.buddylynk.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.repository.BackendRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ChatListViewModel - Fetches real conversations via Backend API
 */
class ChatListViewModel : ViewModel() {
    
    data class ConversationItem(
        val id: String,
        val name: String,
        val avatar: String?,
        val lastMessage: String,
        val time: String,
        val unread: Int,
        val isOnline: Boolean,
        val messageType: String = "text"
    )
    
    private val _conversations = MutableStateFlow<List<ConversationItem>>(emptyList())
    val conversations: StateFlow<List<ConversationItem>> = _conversations.asStateFlow()
    
    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()
    
    init {
        loadConversations()
    }
    
    fun loadConversations() {
        viewModelScope.launch {
            _isLoading.value = true
            _error.value = null
            
            try {
                val currentUser = AuthManager.currentUser.value
                if (currentUser == null) {
                    _conversations.value = emptyList()
                    _isLoading.value = false
                    return@launch
                }
                
                // Get all users the current user has conversed with via BackendRepository
                val conversationPartners = BackendRepository.getConversationPartners(currentUser.userId)
                
                // Build conversation items
                val items = mutableListOf<ConversationItem>()
                conversationPartners.forEach { user ->
                    try {
                        val conversationId = getConversationId(currentUser.userId, user.userId)
                        val messages = BackendRepository.getMessages(conversationId)
                        val lastMessage = messages.maxByOrNull { msg -> msg.createdAt.toLongOrNull() ?: 0L }
                        val unreadCount = messages.count { msg -> !msg.isRead && msg.senderId == user.userId }
                        
                        items.add(ConversationItem(
                            id = user.userId,
                            name = user.username,
                            avatar = user.avatar,
                            lastMessage = lastMessage?.content ?: "Start a conversation",
                            time = formatTime(lastMessage?.createdAt),
                            unread = unreadCount,
                            isOnline = user.isOnline,
                            messageType = "text"
                        ))
                    } catch (e: Exception) {
                        // Skip this conversation
                    }
                }
                
                // Sort by time (most recent first)
                _conversations.value = items.sortedByDescending { item -> item.time }
                
            } catch (e: Exception) {
                _error.value = "Failed to load conversations"
                _conversations.value = emptyList()
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    private fun getConversationId(userId1: String, userId2: String): String {
        val ids = listOf(userId1, userId2).sorted()
        return "conv_${ids[0]}_${ids[1]}"
    }
    
    private fun formatTime(timestamp: String?): String {
        if (timestamp == null) return ""
        val time = timestamp.toLongOrNull() ?: return ""
        val diff = System.currentTimeMillis() - time
        return when {
            diff < 60_000 -> "Now"
            diff < 3600_000 -> "${diff / 60_000}m"
            diff < 86400_000 -> "${diff / 3600_000}h"
            else -> "${diff / 86400_000}d"
        }
    }
    
    fun refreshConversations() {
        loadConversations()
    }
}
