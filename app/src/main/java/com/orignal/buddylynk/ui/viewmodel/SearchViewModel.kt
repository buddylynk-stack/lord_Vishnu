package com.orignal.buddylynk.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.data.moderation.ModerationService
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.data.model.User
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch


/**
 * ViewModel for SearchScreen
 * Handles user and content search with debouncing
 */
@OptIn(FlowPreview::class)
class SearchViewModel : ViewModel() {
    
    // Search query
    private val _searchQuery = MutableStateFlow("")
    val searchQuery: StateFlow<String> = _searchQuery.asStateFlow()
    
    // Search results
    private val _users = MutableStateFlow<List<User>>(emptyList())
    val users: StateFlow<List<User>> = _users.asStateFlow()
    
    private val _posts = MutableStateFlow<List<Post>>(emptyList())
    val posts: StateFlow<List<Post>> = _posts.asStateFlow()
    
    // Suggested/Trending users (loaded initially)
    private val _suggestedUsers = MutableStateFlow<List<User>>(emptyList())
    val suggestedUsers: StateFlow<List<User>> = _suggestedUsers.asStateFlow()
    
    private val _trendingPosts = MutableStateFlow<List<Post>>(emptyList())
    val trendingPosts: StateFlow<List<Post>> = _trendingPosts.asStateFlow()
    
    // Loading states
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _isSearching = MutableStateFlow(false)
    val isSearching: StateFlow<Boolean> = _isSearching.asStateFlow()
    
    // Selected category
    private val _selectedCategory = MutableStateFlow("All")
    val selectedCategory: StateFlow<String> = _selectedCategory.asStateFlow()
    
    // Error
    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()
    
    // Debounce job
    private var searchJob: Job? = null
    
    init {
        loadInitialData()
    }
    
    /**
     * Load suggested users and trending posts on init
     */
    private fun loadInitialData() {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                android.util.Log.d("SearchViewModel", "=== Loading initial data ===")
                
                // Load blocked users from ModerationService (local state)
                val currentUserId = AuthManager.currentUser.value?.userId ?: ""
                val blockedSet = ModerationService.blockedUsers.value
                android.util.Log.d("SearchViewModel", "Loaded ${blockedSet.size} blocked users to filter")
                
                // Load suggested users (excluding current user and blocked users)
                android.util.Log.d("SearchViewModel", "Current user ID: $currentUserId")
                
                val allUsers = BackendRepository.getUsers(20)
                android.util.Log.d("SearchViewModel", "Loaded ${allUsers.size} users from API")
                
                val filteredUsers = allUsers.filter { 
                    it.userId != currentUserId && it.userId !in blockedSet 
                }.take(10)
                android.util.Log.d("SearchViewModel", "Filtered to ${filteredUsers.size} suggested users")
                _suggestedUsers.value = filteredUsers
                
                // Load trending posts - filter out blocked users
                android.util.Log.d("SearchViewModel", "Fetching posts from API...")
                val allPosts = BackendRepository.getFeedPosts()
                android.util.Log.d("SearchViewModel", "Loaded ${allPosts.size} posts from API")
                
                // Filter out blocked users' posts
                val filteredPosts = allPosts.filter { post -> 
                    post.userId !in blockedSet 
                }
                android.util.Log.d("SearchViewModel", "After blocking filter: ${filteredPosts.size} posts")
                
                if (filteredPosts.isEmpty()) {
                    android.util.Log.w("SearchViewModel", "WARNING: No posts after filtering!")
                } else {
                    val firstPost = filteredPosts.firstOrNull()
                    android.util.Log.d("SearchViewModel", "First post: id=${firstPost?.postId}, content=${firstPost?.content?.take(50)}, likes=${firstPost?.likesCount}")
                }
                
                // Sort by engagement (likes + comments + views)
                val trendingPosts = filteredPosts
                    .sortedByDescending { it.likesCount + it.commentsCount + (it.viewsCount / 10) }
                    .take(20)
                
                android.util.Log.d("SearchViewModel", "=== Trending posts: ${trendingPosts.size} ===")
                trendingPosts.forEachIndexed { index, post ->
                    android.util.Log.d("SearchViewModel", "Post $index: ${post.postId} - likes=${post.likesCount}, comments=${post.commentsCount}")
                }
                
                _trendingPosts.value = trendingPosts
                
            } catch (e: Exception) {
                android.util.Log.e("SearchViewModel", "Error loading data: ${e.message}", e)
                _error.value = e.message
            } finally {
                _isLoading.value = false
                android.util.Log.d("SearchViewModel", "=== Loading complete ===")
            }
        }
    }
    
    /**
     * Update search query with debouncing
     */
    fun updateSearchQuery(query: String) {
        _searchQuery.value = query
        
        // Cancel previous search
        searchJob?.cancel()
        
        if (query.isBlank()) {
            _users.value = emptyList()
            _posts.value = emptyList()
            _isSearching.value = false
            return
        }
        
        // Debounce search
        searchJob = viewModelScope.launch {
            delay(300) // 300ms debounce
            performSearch(query)
        }
    }
    
    /**
     * Perform search based on category
     */
    private suspend fun performSearch(query: String) {
        android.util.Log.d("SearchViewModel", "=== performSearch: Starting search for '$query' ===")
        _isSearching.value = true
        try {
            val category = _selectedCategory.value
            android.util.Log.d("SearchViewModel", "performSearch: Category = $category")
            
            when (category) {
                "All" -> {
                    android.util.Log.d("SearchViewModel", "performSearch: Searching users...")
                    val searchedUsers = BackendRepository.searchUsers(query)
                    android.util.Log.d("SearchViewModel", "performSearch: Found ${searchedUsers.size} users")
                    _users.value = searchedUsers
                    
                    android.util.Log.d("SearchViewModel", "performSearch: Searching posts...")
                    val allPosts = BackendRepository.getFeedPosts()
                    val searchedPosts = allPosts.filter { it.content.contains(query, ignoreCase = true) }.take(10)
                    android.util.Log.d("SearchViewModel", "performSearch: Found ${searchedPosts.size} posts")
                    _posts.value = searchedPosts
                }
                "People" -> {
                    android.util.Log.d("SearchViewModel", "performSearch: Searching users only...")
                    val searchedUsers = BackendRepository.searchUsers(query)
                    android.util.Log.d("SearchViewModel", "performSearch: Found ${searchedUsers.size} users")
                    _users.value = searchedUsers
                    _posts.value = emptyList()
                }
                else -> {
                    // For Teams, Events, Topics - search posts with tags
                    _users.value = emptyList()
                    val allPosts = BackendRepository.getFeedPosts()
                    val searchedPosts = allPosts.filter { it.content.contains(query, ignoreCase = true) }.take(20)
                    android.util.Log.d("SearchViewModel", "performSearch: Found ${searchedPosts.size} posts")
                    _posts.value = searchedPosts
                }
            }
            
            android.util.Log.d("SearchViewModel", "=== performSearch: Complete. Users=${_users.value.size}, Posts=${_posts.value.size} ===")
        } catch (e: Exception) {
            android.util.Log.e("SearchViewModel", "performSearch: Error - ${e.message}", e)
            _error.value = e.message
        } finally {
            _isSearching.value = false
        }
    }
    
    /**
     * Update selected category
     */
    fun selectCategory(category: String) {
        _selectedCategory.value = category
        
        // Re-search if there's a query
        if (_searchQuery.value.isNotBlank()) {
            searchJob?.cancel()
            searchJob = viewModelScope.launch {
                performSearch(_searchQuery.value)
            }
        }
    }
    
    /**
     * Follow a user
     */
    fun followUser(userId: String) {
        viewModelScope.launch {
            try {
                val currentUser = AuthManager.currentUser.value ?: return@launch
                val follow = com.orignal.buddylynk.data.model.Follow(
                    followId = "${currentUser.userId}_$userId",
                    followerId = currentUser.userId,
                    followingId = userId,
                    followerUsername = currentUser.username,
                    followerAvatar = currentUser.avatar,
                    createdAt = System.currentTimeMillis().toString()
                )
                BackendRepository.followUser(userId, currentUser.userId)
                
                // Update UI - mark as followed
                _suggestedUsers.value = _suggestedUsers.value.filter { it.userId != userId }
                _users.value = _users.value.filter { it.userId != userId }
            } catch (e: Exception) {
                _error.value = "Failed to follow user"
            }
        }
    }
    
    /**
     * Refresh data
     */
    fun refresh() {
        loadInitialData()
        if (_searchQuery.value.isNotBlank()) {
            viewModelScope.launch {
                performSearch(_searchQuery.value)
            }
        }
    }
    
    /**
     * Like a post
     */
    fun likePost(postId: String) {
        viewModelScope.launch {
            _trendingPosts.value = _trendingPosts.value.map { post ->
                if (post.postId == postId) {
                    val newIsLiked = !post.isLiked
                    val newCount = if (newIsLiked) post.likesCount + 1 else post.likesCount - 1
                    post.copy(isLiked = newIsLiked, likesCount = newCount)
                } else post
            }
        }
    }
    
    /**
     * Save/bookmark a post
     */
    fun savePost(postId: String) {
        viewModelScope.launch {
            _trendingPosts.value = _trendingPosts.value.map { post ->
                if (post.postId == postId) {
                    post.copy(isBookmarked = !post.isBookmarked)
                } else post
            }
        }
    }
    
    fun clearError() {
        _error.value = null
    }
}
