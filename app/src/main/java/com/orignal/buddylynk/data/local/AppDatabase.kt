package com.orignal.buddylynk.data.local

import android.content.Context
import androidx.room.*
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.data.model.User
import kotlinx.coroutines.flow.Flow

/**
 * Room Database for offline caching
 */
@Database(
    entities = [CachedPost::class, CachedUser::class, PendingAction::class],
    version = 1,
    exportSchema = false
)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun postDao(): PostDao
    abstract fun userDao(): UserDao
    abstract fun pendingActionDao(): PendingActionDao
    
    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null
        
        fun getInstance(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "buddylynk_cache.db"
                )
                .fallbackToDestructiveMigration()
                .build()
                INSTANCE = instance
                instance
            }
        }
    }
}

/**
 * Type converters for Room
 */
class Converters {
    @TypeConverter
    fun fromStringList(value: List<String>?): String = value?.joinToString(",") ?: ""
    
    @TypeConverter
    fun toStringList(value: String): List<String> = 
        if (value.isEmpty()) emptyList() else value.split(",")
}

/**
 * Cached Post entity
 */
@Entity(tableName = "posts")
data class CachedPost(
    @PrimaryKey val postId: String,
    val userId: String,
    val username: String,
    val userAvatar: String?,
    val content: String?,
    val mediaUrl: String?,
    val mediaType: String,
    val likesCount: Int,
    val commentsCount: Int,
    val sharesCount: Int,
    val viewsCount: Int,
    val isLiked: Boolean,
    val isBookmarked: Boolean,
    val createdAt: String,
    val cachedAt: Long = System.currentTimeMillis()
) {
    fun toPost() = Post(
        postId = postId,
        userId = userId,
        username = username,
        userAvatar = userAvatar,
        content = content ?: "",
        mediaUrl = mediaUrl,
        mediaType = mediaType,
        likesCount = likesCount,
        commentsCount = commentsCount,
        sharesCount = sharesCount,
        viewsCount = viewsCount,
        isLiked = isLiked,
        isBookmarked = isBookmarked,
        createdAt = createdAt
    )
    
    companion object {
        fun fromPost(post: Post) = CachedPost(
            postId = post.postId,
            userId = post.userId,
            username = post.username ?: "",
            userAvatar = post.userAvatar,
            content = post.content,
            mediaUrl = post.mediaUrl,
            mediaType = post.mediaType ?: "text",
            likesCount = post.likesCount,
            commentsCount = post.commentsCount,
            sharesCount = post.sharesCount,
            viewsCount = post.viewsCount,
            isLiked = post.isLiked,
            isBookmarked = post.isBookmarked,
            createdAt = post.createdAt
        )
    }
}

/**
 * Cached User entity
 */
@Entity(tableName = "users")
data class CachedUser(
    @PrimaryKey val userId: String,
    val username: String,
    val email: String,
    val avatar: String?,
    val bio: String?,
    val isVerified: Boolean,
    val followersCount: Int,
    val followingCount: Int,
    val postsCount: Int,
    val cachedAt: Long = System.currentTimeMillis()
) {
    fun toUser() = User(
        userId = userId,
        username = username,
        email = email,
        avatar = avatar,
        bio = bio,
        isVerified = isVerified,
        followersCount = followersCount,
        followingCount = followingCount,
        postsCount = postsCount
    )
    
    companion object {
        fun fromUser(user: User) = CachedUser(
            userId = user.userId,
            username = user.username,
            email = user.email,
            avatar = user.avatar,
            bio = user.bio,
            isVerified = user.isVerified,
            followersCount = user.followersCount,
            followingCount = user.followingCount,
            postsCount = user.postsCount
        )
    }
}

/**
 * Pending actions for offline sync
 */
@Entity(tableName = "pending_actions")
data class PendingAction(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val actionType: String, // "like", "unlike", "follow", "unfollow", "post", "comment"
    val targetId: String,
    val payload: String?, // JSON payload for complex data
    val createdAt: Long = System.currentTimeMillis(),
    val retryCount: Int = 0
)

/**
 * Post DAO
 */
@Dao
interface PostDao {
    @Query("SELECT * FROM posts ORDER BY cachedAt DESC LIMIT :limit")
    fun getRecentPosts(limit: Int = 50): Flow<List<CachedPost>>
    
    @Query("SELECT * FROM posts WHERE postId = :postId")
    suspend fun getPost(postId: String): CachedPost?
    
    @Query("SELECT * FROM posts WHERE userId = :userId ORDER BY createdAt DESC")
    fun getUserPosts(userId: String): Flow<List<CachedPost>>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPosts(posts: List<CachedPost>)
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPost(post: CachedPost)
    
    @Query("UPDATE posts SET isLiked = :isLiked, likesCount = likesCount + :delta WHERE postId = :postId")
    suspend fun updateLike(postId: String, isLiked: Boolean, delta: Int)
    
    @Query("UPDATE posts SET isBookmarked = :isBookmarked WHERE postId = :postId")
    suspend fun updateBookmark(postId: String, isBookmarked: Boolean)
    
    @Query("DELETE FROM posts WHERE cachedAt < :timestamp")
    suspend fun deleteOldPosts(timestamp: Long)
    
    @Query("DELETE FROM posts")
    suspend fun clearAll()
}

/**
 * User DAO
 */
@Dao
interface UserDao {
    @Query("SELECT * FROM users WHERE userId = :userId")
    suspend fun getUser(userId: String): CachedUser?
    
    @Query("SELECT * FROM users WHERE userId IN (:userIds)")
    suspend fun getUsers(userIds: List<String>): List<CachedUser>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUser(user: CachedUser)
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUsers(users: List<CachedUser>)
    
    @Query("DELETE FROM users WHERE cachedAt < :timestamp")
    suspend fun deleteOldUsers(timestamp: Long)
    
    @Query("DELETE FROM users")
    suspend fun clearAll()
}

/**
 * Pending Action DAO
 */
@Dao
interface PendingActionDao {
    @Query("SELECT * FROM pending_actions ORDER BY createdAt ASC")
    suspend fun getAllPending(): List<PendingAction>
    
    @Insert
    suspend fun insert(action: PendingAction)
    
    @Delete
    suspend fun delete(action: PendingAction)
    
    @Query("DELETE FROM pending_actions WHERE id = :id")
    suspend fun deleteById(id: Long)
    
    @Query("UPDATE pending_actions SET retryCount = retryCount + 1 WHERE id = :id")
    suspend fun incrementRetryCount(id: Long)
    
    @Query("DELETE FROM pending_actions WHERE retryCount >= :maxRetries")
    suspend fun deleteFailedActions(maxRetries: Int = 5)
}
