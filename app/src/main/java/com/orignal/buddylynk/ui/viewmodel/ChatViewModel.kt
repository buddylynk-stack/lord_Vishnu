package com.orignal.buddylynk.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.data.model.Message
import com.orignal.buddylynk.data.model.User
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.UUID

/**
 * ChatViewModel - Real-time chat with Backend API integration
 * Professional messaging for BuddyLynk
 */
class ChatViewModel : ViewModel() {
    
    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()
    
    private val _partnerUser = MutableStateFlow<User?>(null)
    val partnerUser: StateFlow<User?> = _partnerUser.asStateFlow()
    
    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _isOnline = MutableStateFlow(false)
    val isOnline: StateFlow<Boolean> = _isOnline.asStateFlow()
    
    private val _isSending = MutableStateFlow(false)
    val isSending: StateFlow<Boolean> = _isSending.asStateFlow()
    
    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()
    
    private var conversationId: String = ""
    private var partnerId: String = ""
    
    /**
     * Load conversation with real data from Backend API
     */
    fun loadConversation(conversationIdOrUserId: String) {
        partnerId = conversationIdOrUserId
        conversationId = getOrCreateConversationId(conversationIdOrUserId)
        
        viewModelScope.launch {
            _isLoading.value = true
            _error.value = null
            
            try {
                // Get partner user from Backend API
                val user = BackendRepository.getUser(partnerId)
                _partnerUser.value = user ?: User(
                    userId = partnerId,
                    username = partnerId.replaceFirstChar { it.uppercase() },
                    email = "",
                    avatar = null
                )
                
                // Get user's online status
                _isOnline.value = user?.isOnline ?: false
                
                // Load messages via BackendRepository
                val loadedMessages = BackendRepository.getMessages(conversationId)
                _messages.value = loadedMessages.sortedBy { 
                    it.createdAt.toLongOrNull() ?: 0 
                }
                
            } catch (e: Exception) {
                _error.value = "Failed to load chat: ${e.message}"
                e.printStackTrace()
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    /**
     * Send a new message - saves via Backend API
     */
    fun sendMessage(content: String) {
        if (content.isBlank()) return
        
        val currentUser = AuthManager.currentUser.value ?: return
        
        val message = Message(
            messageId = UUID.randomUUID().toString(),
            conversationId = conversationId,
            senderId = currentUser.userId,
            receiverId = partnerId,
            senderName = currentUser.username,
            senderAvatar = currentUser.avatar,
            content = content.trim(),
            isRead = false,
            createdAt = System.currentTimeMillis().toString()
        )
        
        // Add to local list immediately for responsive UI
        _messages.value = _messages.value + message
        
        // Save via Backend API
        viewModelScope.launch {
            _isSending.value = true
            try {
                val success = BackendRepository.sendMessage(partnerId, content)
                if (!success) {
                    _error.value = "Failed to send message"
                    // Remove message from list if failed
                    _messages.value = _messages.value.filter { it.messageId != message.messageId }
                }
            } catch (e: Exception) {
                _error.value = "Error sending message: ${e.message}"
                _messages.value = _messages.value.filter { it.messageId != message.messageId }
            } finally {
                _isSending.value = false
            }
        }
    }
    
    /**
     * Check if message is from current user
     */
    fun isFromCurrentUser(message: Message): Boolean {
        val currentUserId = AuthManager.currentUser.value?.userId ?: return false
        return message.senderId == currentUserId
    }
    
    /**
     * Get or create conversation ID between two users
     * Creates a deterministic ID based on sorted user IDs
     */
    private fun getOrCreateConversationId(otherUserId: String): String {
        val currentUserId = AuthManager.currentUser.value?.userId ?: "unknown"
        val ids = listOf(currentUserId, otherUserId).sorted()
        return "conv_${ids[0]}_${ids[1]}"
    }
    
    /**
     * Mark messages from partner as read
     */
    fun markAsRead() {
        viewModelScope.launch {
            // Update local state
            _messages.value = _messages.value.map { message ->
                if (!message.isRead && !isFromCurrentUser(message)) {
                    message.copy(isRead = true)
                } else {
                    message
                }
            }
            // Mark read is handled by backend automatically
        }
    }
    
    /**
     * Refresh messages
     */
    fun refreshMessages() {
        viewModelScope.launch {
            try {
                val loadedMessages = BackendRepository.getMessages(conversationId)
                _messages.value = loadedMessages.sortedBy { 
                    it.createdAt.toLongOrNull() ?: 0 
                }
            } catch (e: Exception) {
                // Silent refresh failure
            }
        }
    }
}
