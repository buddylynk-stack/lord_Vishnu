package com.orignal.buddylynk.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.aws.DynamoDbService
import com.orignal.buddylynk.data.model.Activity
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel for ActivityScreen
 */
class ActivityViewModel : ViewModel() {
    
    private val _activities = MutableStateFlow<List<Activity>>(emptyList())
    val activities: StateFlow<List<Activity>> = _activities.asStateFlow()
    
    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _selectedFilter = MutableStateFlow("All")
    val selectedFilter: StateFlow<String> = _selectedFilter.asStateFlow()
    
    private var allActivities: List<Activity> = emptyList()
    
    /**
     * Load activities for current user
     */
    fun loadActivities() {
        viewModelScope.launch {
            _isLoading.value = true
            
            try {
                val userId = AuthManager.currentUser.value?.userId ?: return@launch
                allActivities = DynamoDbService.getActivities(userId)
                    .sortedByDescending { it.createdAt.toLongOrNull() ?: 0L }
                
                applyFilter()
            } catch (e: Exception) {
                // Handle error
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    /**
     * Set filter
     */
    fun setFilter(filter: String) {
        _selectedFilter.value = filter
        applyFilter()
    }
    
    /**
     * Apply current filter
     */
    private fun applyFilter() {
        val filter = _selectedFilter.value.lowercase()
        
        _activities.value = if (filter == "all") {
            allActivities
        } else {
            allActivities.filter { activity ->
                when (filter) {
                    "likes" -> activity.type == "like"
                    "comments" -> activity.type == "comment"
                    "follows" -> activity.type == "follow"
                    "mentions" -> activity.type == "mention"
                    else -> true
                }
            }
        }
    }
    
    /**
     * Mark all as read
     */
    fun markAllAsRead() {
        viewModelScope.launch {
            allActivities = allActivities.map { it.copy(isRead = true) }
            applyFilter()
            // TODO: Update in DynamoDB
        }
    }
}
