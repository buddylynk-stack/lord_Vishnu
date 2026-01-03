package com.orignal.buddylynk.data.api

import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import kotlin.math.min
import kotlin.math.pow

/**
 * SmoothApiClient - Robust API client with automatic retries, caching, and smooth error handling
 * 
 * Features:
 * - Automatic retry with exponential backoff
 * - Response caching with TTL
 * - Connection pooling for performance
 * - Request deduplication (prevents duplicate concurrent requests)
 * - Graceful error handling with typed errors
 * - Request/response logging
 * - Offline detection
 */
object SmoothApiClient {
    
    private const val TAG = "SmoothApiClient"
    
    // Config
    private const val MAX_RETRIES = 3
    private const val INITIAL_RETRY_DELAY_MS = 500L
    private const val MAX_RETRY_DELAY_MS = 5000L
    private const val CACHE_DEFAULT_TTL_MS = 60_000L // 1 minute cache
    private const val CONNECTION_TIMEOUT_SECONDS = 15L
    private const val READ_TIMEOUT_SECONDS = 20L
    private const val WRITE_TIMEOUT_SECONDS = 20L
    
    // Connection pool for better performance
    private val connectionPool = ConnectionPool(
        maxIdleConnections = 10,
        keepAliveDuration = 5,
        timeUnit = TimeUnit.MINUTES
    )
    
    // OkHttp client with optimized settings
    private val client = OkHttpClient.Builder()
        .connectTimeout(CONNECTION_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .readTimeout(READ_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .writeTimeout(WRITE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .connectionPool(connectionPool)
        .retryOnConnectionFailure(true)
        .addInterceptor(LoggingInterceptor())
        .build()
    
    // Simple in-memory cache
    private val cache = ConcurrentHashMap<String, CacheEntry>()
    
    // Request deduplication - prevent duplicate concurrent requests
    private val inFlightRequests = ConcurrentHashMap<String, Deferred<ApiResponse>>()
    
    // Auth token
    private var authToken: String? = null
    
    // Server status
    private val _isServerOnline = MutableStateFlow(true)
    val isServerOnline: StateFlow<Boolean> = _isServerOnline.asStateFlow()
    
    // Last successful request timestamp
    private var lastSuccessTime = System.currentTimeMillis()
    
    fun setAuthToken(token: String?) {
        authToken = token
        Log.d(TAG, "Auth token ${if (token != null) "set" else "cleared"}")
    }
    
    // ==================== PUBLIC API ====================
    
    /**
     * GET request with automatic caching and retries
     */
    suspend fun get(
        url: String,
        useCache: Boolean = true,
        cacheTtlMs: Long = CACHE_DEFAULT_TTL_MS,
        forceRefresh: Boolean = false
    ): ApiResponse = withContext(Dispatchers.IO) {
        
        // Check cache first (if not forcing refresh)
        if (useCache && !forceRefresh) {
            cache[url]?.let { cached ->
                if (!cached.isExpired()) {
                    Log.d(TAG, "Cache HIT: $url")
                    return@withContext ApiResponse.Success(cached.data)
                }
            }
        }
        
        // Deduplicate concurrent requests to same URL
        inFlightRequests[url]?.let { existing ->
            Log.d(TAG, "Dedup: Waiting for in-flight request to $url")
            return@withContext existing.await()
        }
        
        // Create deferred for this request
        val deferred = async {
            executeWithRetry(url) {
                val request = Request.Builder()
                    .url(url)
                    .get()
                    .apply { authToken?.let { header("Authorization", "Bearer $it") } }
                    .build()
                
                val response = client.newCall(request).execute()
                handleResponse(response, url, useCache, cacheTtlMs)
            }
        }
        
        inFlightRequests[url] = deferred
        try {
            deferred.await()
        } finally {
            inFlightRequests.remove(url)
        }
    }
    
    /**
     * POST request with automatic retries
     */
    suspend fun post(
        url: String,
        body: JSONObject,
        retryOnError: Boolean = true
    ): ApiResponse = withContext(Dispatchers.IO) {
        
        val requestBody = body.toString().toRequestBody("application/json".toMediaType())
        
        if (retryOnError) {
            executeWithRetry(url) {
                val request = Request.Builder()
                    .url(url)
                    .post(requestBody)
                    .apply { authToken?.let { header("Authorization", "Bearer $it") } }
                    .build()
                
                val response = client.newCall(request).execute()
                handleResponse(response, url, false, 0)
            }
        } else {
            try {
                val request = Request.Builder()
                    .url(url)
                    .post(requestBody)
                    .apply { authToken?.let { header("Authorization", "Bearer $it") } }
                    .build()
                
                val response = client.newCall(request).execute()
                handleResponse(response, url, false, 0)
            } catch (e: Exception) {
                handleError(e)
            }
        }
    }
    
    /**
     * PUT request
     */
    suspend fun put(url: String, body: JSONObject): ApiResponse = withContext(Dispatchers.IO) {
        executeWithRetry(url) {
            val request = Request.Builder()
                .url(url)
                .put(body.toString().toRequestBody("application/json".toMediaType()))
                .apply { authToken?.let { header("Authorization", "Bearer $it") } }
                .build()
            
            val response = client.newCall(request).execute()
            handleResponse(response, url, false, 0)
        }
    }
    
    /**
     * DELETE request
     */
    suspend fun delete(url: String): ApiResponse = withContext(Dispatchers.IO) {
        executeWithRetry(url) {
            val request = Request.Builder()
                .url(url)
                .delete()
                .apply { authToken?.let { header("Authorization", "Bearer $it") } }
                .build()
            
            val response = client.newCall(request).execute()
            handleResponse(response, url, false, 0)
        }
    }
    
    // ==================== RETRY LOGIC ====================
    
    private suspend fun executeWithRetry(
        url: String,
        block: suspend () -> ApiResponse
    ): ApiResponse {
        var lastException: Exception? = null
        var currentDelay = INITIAL_RETRY_DELAY_MS
        
        repeat(MAX_RETRIES) { attempt ->
            try {
                val result = block()
                
                // Success - update server status
                if (result is ApiResponse.Success) {
                    _isServerOnline.value = true
                    lastSuccessTime = System.currentTimeMillis()
                }
                
                return result
                
            } catch (e: Exception) {
                lastException = e
                Log.w(TAG, "Attempt ${attempt + 1}/$MAX_RETRIES failed for $url: ${e.message}")
                
                // Don't retry on certain errors
                if (e is ApiException.Unauthorized || e is ApiException.NotFound) {
                    return handleError(e)
                }
                
                // Exponential backoff
                if (attempt < MAX_RETRIES - 1) {
                    delay(currentDelay)
                    currentDelay = min(currentDelay * 2, MAX_RETRY_DELAY_MS)
                }
            }
        }
        
        // All retries failed
        _isServerOnline.value = false
        return handleError(lastException ?: IOException("Request failed after $MAX_RETRIES attempts"))
    }
    
    // ==================== RESPONSE HANDLING ====================
    
    private fun handleResponse(
        response: Response,
        url: String,
        useCache: Boolean,
        cacheTtlMs: Long
    ): ApiResponse {
        val body = response.body?.string() ?: ""
        
        return when {
            response.isSuccessful -> {
                // Cache successful responses
                if (useCache && cacheTtlMs > 0) {
                    cache[url] = CacheEntry(body, System.currentTimeMillis() + cacheTtlMs)
                }
                
                ApiResponse.Success(body)
            }
            
            response.code == 401 -> {
                Log.w(TAG, "Unauthorized: $url")
                ApiResponse.Error(ApiError.Unauthorized, "Authentication required")
            }
            
            response.code == 404 -> {
                Log.w(TAG, "Not found: $url")
                ApiResponse.Error(ApiError.NotFound, "Resource not found")
            }
            
            response.code == 429 -> {
                Log.w(TAG, "Rate limited: $url")
                ApiResponse.Error(ApiError.RateLimited, "Too many requests")
            }
            
            response.code in 500..599 -> {
                Log.e(TAG, "Server error ${response.code}: $url")
                _isServerOnline.value = false
                ApiResponse.Error(ApiError.ServerError, "Server error: ${response.code}")
            }
            
            else -> {
                val errorMsg = try {
                    JSONObject(body).optString("error", "Unknown error")
                } catch (e: Exception) {
                    "Request failed: ${response.code}"
                }
                ApiResponse.Error(ApiError.Unknown, errorMsg)
            }
        }
    }
    
    private fun handleError(e: Exception): ApiResponse {
        return when (e) {
            is java.net.SocketTimeoutException -> {
                Log.e(TAG, "Timeout: ${e.message}")
                _isServerOnline.value = false
                ApiResponse.Error(ApiError.Timeout, "Connection timed out")
            }
            
            is java.net.UnknownHostException -> {
                Log.e(TAG, "No internet: ${e.message}")
                _isServerOnline.value = false
                ApiResponse.Error(ApiError.NoInternet, "No internet connection")
            }
            
            is java.net.ConnectException -> {
                Log.e(TAG, "Connection failed: ${e.message}")
                _isServerOnline.value = false
                ApiResponse.Error(ApiError.ConnectionFailed, "Could not connect to server")
            }
            
            is IOException -> {
                Log.e(TAG, "Network error: ${e.message}")
                _isServerOnline.value = false
                ApiResponse.Error(ApiError.NetworkError, e.message ?: "Network error")
            }
            
            else -> {
                Log.e(TAG, "Unknown error: ${e.message}", e)
                ApiResponse.Error(ApiError.Unknown, e.message ?: "Unknown error")
            }
        }
    }
    
    // ==================== CACHE MANAGEMENT ====================
    
    /**
     * Clear all cached responses
     */
    fun clearCache() {
        cache.clear()
        Log.d(TAG, "Cache cleared")
    }
    
    /**
     * Clear specific cached URL
     */
    fun invalidateCache(url: String) {
        cache.remove(url)
        Log.d(TAG, "Cache invalidated: $url")
    }
    
    /**
     * Clear expired cache entries
     */
    fun cleanExpiredCache() {
        val now = System.currentTimeMillis()
        cache.entries.removeIf { it.value.isExpired() }
        Log.d(TAG, "Expired cache entries cleaned")
    }
    
    // ==================== HEALTH CHECK ====================
    
    /**
     * Quick health check - pings server
     */
    suspend fun healthCheck(): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${ApiConfig.API_URL}/posts/feed?limit=1")
                .head()
                .build()
            
            val response = client.newCall(request).execute()
            val isHealthy = response.isSuccessful
            _isServerOnline.value = isHealthy
            isHealthy
        } catch (e: Exception) {
            Log.w(TAG, "Health check failed: ${e.message}")
            _isServerOnline.value = false
            false
        }
    }
}

// ==================== DATA CLASSES ====================

/**
 * API Response sealed class for type-safe responses
 */
sealed class ApiResponse {
    data class Success(val data: String) : ApiResponse() {
        fun asJson(): JSONObject = JSONObject(data)
        fun asJsonArray(): JSONArray = JSONArray(data)
    }
    
    data class Error(val error: ApiError, val message: String) : ApiResponse()
    
    fun isSuccess(): Boolean = this is Success
    fun isError(): Boolean = this is Error
    
    fun getOrNull(): String? = (this as? Success)?.data
    fun getOrThrow(): String = (this as? Success)?.data 
        ?: throw ApiException.fromError((this as Error).error, (this as Error).message)
}

/**
 * API Error types
 */
enum class ApiError {
    NoInternet,
    Timeout,
    ConnectionFailed,
    NetworkError,
    Unauthorized,
    NotFound,
    RateLimited,
    ServerError,
    Unknown
}

/**
 * API Exception for throwing errors
 */
sealed class ApiException(message: String) : Exception(message) {
    class NoInternet : ApiException("No internet connection")
    class Timeout : ApiException("Request timed out")
    class ConnectionFailed : ApiException("Connection failed")
    class Unauthorized : ApiException("Unauthorized")
    class NotFound : ApiException("Not found")
    class RateLimited : ApiException("Rate limited")
    class ServerError(code: Int) : ApiException("Server error: $code")
    class Unknown(msg: String) : ApiException(msg)
    
    companion object {
        fun fromError(error: ApiError, message: String): ApiException = when (error) {
            ApiError.NoInternet -> NoInternet()
            ApiError.Timeout -> Timeout()
            ApiError.ConnectionFailed -> ConnectionFailed()
            ApiError.Unauthorized -> Unauthorized()
            ApiError.NotFound -> NotFound()
            ApiError.RateLimited -> RateLimited()
            ApiError.ServerError -> ServerError(500)
            else -> Unknown(message)
        }
    }
}

/**
 * Cache entry with TTL
 */
private data class CacheEntry(
    val data: String,
    val expiresAt: Long
) {
    fun isExpired(): Boolean = System.currentTimeMillis() > expiresAt
}

/**
 * Logging interceptor for debugging
 */
private class LoggingInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        val startTime = System.currentTimeMillis()
        
        Log.d("SmoothApiClient", "➡️ ${request.method} ${request.url}")
        
        val response = chain.proceed(request)
        val duration = System.currentTimeMillis() - startTime
        
        val emoji = if (response.isSuccessful) "✅" else "❌"
        Log.d("SmoothApiClient", "$emoji ${response.code} ${request.url} (${duration}ms)")
        
        return response
    }
}
