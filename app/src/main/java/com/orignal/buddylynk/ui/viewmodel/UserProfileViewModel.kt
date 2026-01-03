package com.orignal.buddylynk.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.data.model.User
import com.orignal.buddylynk.data.model.UserProfile
import com.orignal.buddylynk.data.social.FollowService
import com.orignal.buddylynk.data.api.ApiService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel for UserProfileScreen
 */
class UserProfileViewModel : ViewModel() {
    
    private val _userProfile = MutableStateFlow<UserProfile?>(null)
    val userProfile: StateFlow<UserProfile?> = _userProfile.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    // User's posts for profile grid
    private val _userPosts = MutableStateFlow<List<Post>>(emptyList())
    val userPosts: StateFlow<List<Post>> = _userPosts.asStateFlow()
    
    private val _isFollowing = MutableStateFlow(false)
    val isFollowing: StateFlow<Boolean> = _isFollowing.asStateFlow()
    
    private val _isFollowLoading = MutableStateFlow(false)
    val isFollowLoading: StateFlow<Boolean> = _isFollowLoading.asStateFlow()
    
    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()
    
    // Followers and Following lists
    private val _followers = MutableStateFlow<List<User>>(emptyList())
    val followers: StateFlow<List<User>> = _followers.asStateFlow()
    
    private val _following = MutableStateFlow<List<User>>(emptyList())
    val following: StateFlow<List<User>> = _following.asStateFlow()
    
    private val _isLoadingFollowers = MutableStateFlow(false)
    val isLoadingFollowers: StateFlow<Boolean> = _isLoadingFollowers.asStateFlow()
    
    private val _isLoadingFollowing = MutableStateFlow(false)
    val isLoadingFollowing: StateFlow<Boolean> = _isLoadingFollowing.asStateFlow()
    
    private var currentUserId: String = ""
    
    /**
     * Load user posts for profile grid
     */
    fun loadUserPosts(userId: String) {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                val posts = BackendRepository.getUserPosts(userId)
                _userPosts.value = posts
            } catch (e: Exception) {
                _error.value = e.message
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    /**
     * Load user profile by ID
     */
    fun loadUserProfile(userId: String) {
        android.util.Log.d("UserProfileVM", "=== loadUserProfile called for userId: $userId ===")
        currentUserId = userId
        viewModelScope.launch {
            _isLoading.value = true
            _error.value = null
            
            try {
                // Get user details
                android.util.Log.d("UserProfileVM", "Fetching user from Backend API...")
                val user = BackendRepository.getUser(userId)
                android.util.Log.d("UserProfileVM", "User result: ${user?.username ?: "NULL"}")
                
                if (user != null) {
                    android.util.Log.d("UserProfileVM", "User found! Username: ${user.username}, Avatar: ${user.avatar?.take(50)}")
                    
                    // Get user's posts
                    android.util.Log.d("UserProfileVM", "Fetching user posts...")
                    val posts = BackendRepository.getUserPosts(userId)
                    android.util.Log.d("UserProfileVM", "Found ${posts.size} posts for user")
                    
                    // Check follow status
                    val following = FollowService.isFollowing(userId)
                    val isFollower = FollowService.followersCache.value.contains(userId)
                    android.util.Log.d("UserProfileVM", "Follow status: following=$following, isFollower=$isFollower")
                    
                    _isFollowing.value = following
                    
                    _userProfile.value = UserProfile(
                        user = user,
                        isFollowing = following,
                        isFollower = isFollower,
                        isMutual = following && isFollower,
                        posts = posts
                    )
                    android.util.Log.d("UserProfileVM", "=== Profile loaded successfully! ===")
                } else {
                    android.util.Log.e("UserProfileVM", "User NOT FOUND for userId: $userId")
                    _error.value = "User not found"
                }
            } catch (e: Exception) {
                android.util.Log.e("UserProfileVM", "Error loading profile: ${e.message}", e)
                _error.value = e.message
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    /**
     * Toggle follow/unfollow
     */
    fun toggleFollow() {
        if (currentUserId.isBlank()) return
        
        viewModelScope.launch {
            _isFollowLoading.value = true
            
            try {
                val success = if (_isFollowing.value) {
                    FollowService.unfollowUser(currentUserId)
                } else {
                    FollowService.followUser(currentUserId)
                }
                
                if (success) {
                    _isFollowing.value = !_isFollowing.value
                    
                    // Update profile
                    _userProfile.value?.let { profile ->
                        val newFollowersCount = if (_isFollowing.value) {
                            profile.user.followersCount + 1
                        } else {
                            (profile.user.followersCount - 1).coerceAtLeast(0)
                        }
                        
                        _userProfile.value = profile.copy(
                            user = profile.user.copy(followersCount = newFollowersCount),
                            isFollowing = _isFollowing.value,
                            isMutual = _isFollowing.value && profile.isFollower
                        )
                    }
                }
            } catch (e: Exception) {
                _error.value = e.message
            } finally {
                _isFollowLoading.value = false
            }
        }
    }
    
    /**
     * Refresh profile
     */
    fun refresh() {
        if (currentUserId.isNotBlank()) {
            loadUserProfile(currentUserId)
        }
    }
    
    /**
     * Load followers list with full user data
     */
    fun loadFollowers(userId: String) {
        viewModelScope.launch {
            _isLoadingFollowers.value = true
            try {
                val followerIds = ApiService.getFollowers(userId).getOrNull() ?: emptyList()
                val users = followerIds.mapNotNull { id ->
                    BackendRepository.getUser(id)
                }
                _followers.value = users
            } catch (e: Exception) {
                android.util.Log.e("UserProfileVM", "Error loading followers: ${e.message}")
            } finally {
                _isLoadingFollowers.value = false
            }
        }
    }
    
    /**
     * Load following list with full user data
     */
    fun loadFollowing(userId: String) {
        viewModelScope.launch {
            _isLoadingFollowing.value = true
            try {
                val followingIds = ApiService.getFollowing(userId).getOrNull() ?: emptyList()
                val users = followingIds.mapNotNull { id ->
                    BackendRepository.getUser(id)
                }
                _following.value = users
            } catch (e: Exception) {
                android.util.Log.e("UserProfileVM", "Error loading following: ${e.message}")
            } finally {
                _isLoadingFollowing.value = false
            }
        }
    }
}
