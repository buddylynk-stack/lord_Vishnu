package com.orignal.buddylynk.data.util

import kotlinx.coroutines.delay
import kotlin.math.pow

/**
 * Result wrapper for handling success/error states
 */
sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Error(val exception: Throwable, val message: String? = null) : Result<Nothing>()
    data object Loading : Result<Nothing>()
    
    val isSuccess get() = this is Success
    val isError get() = this is Error
    val isLoading get() = this is Loading
    
    fun getOrNull(): T? = when (this) {
        is Success -> data
        else -> null
    }
    
    fun getOrDefault(default: @UnsafeVariance T): T = when (this) {
        is Success -> data
        else -> default
    }
    
    fun <R> map(transform: (T) -> R): Result<R> = when (this) {
        is Success -> Success(transform(data))
        is Error -> this
        is Loading -> Loading
    }
    
    fun onSuccess(action: (T) -> Unit): Result<T> {
        if (this is Success) action(data)
        return this
    }
    
    fun onError(action: (Throwable, String?) -> Unit): Result<T> {
        if (this is Error) action(exception, message)
        return this
    }
}

/**
 * Error types for categorized error handling
 */
sealed class AppError : Throwable() {
    // Network errors
    data class NetworkError(override val message: String = "No internet connection") : AppError()
    data class TimeoutError(override val message: String = "Request timed out") : AppError()
    data class ServerError(val code: Int, override val message: String = "Server error") : AppError()
    
    // Auth errors
    data class AuthError(override val message: String = "Authentication failed") : AppError()
    data class SessionExpired(override val message: String = "Session expired, please login again") : AppError()
    
    // Data errors
    data class NotFoundError(override val message: String = "Resource not found") : AppError()
    data class ValidationError(override val message: String = "Invalid data") : AppError()
    data class ConflictError(override val message: String = "Resource already exists") : AppError()
    
    // Storage errors
    data class StorageError(override val message: String = "Storage operation failed") : AppError()
    data class UploadError(override val message: String = "Upload failed") : AppError()
    
    // Generic
    data class UnknownError(override val message: String = "Something went wrong") : AppError()
}

/**
 * Retry configuration
 */
data class RetryConfig(
    val maxRetries: Int = 3,
    val initialDelayMs: Long = 1000,
    val maxDelayMs: Long = 10000,
    val backoffMultiplier: Double = 2.0,
    val retryOn: Set<Class<out Throwable>> = setOf(
        AppError.NetworkError::class.java,
        AppError.TimeoutError::class.java,
        AppError.ServerError::class.java
    )
)

/**
 * Execute with retry logic
 */
suspend fun <T> withRetry(
    config: RetryConfig = RetryConfig(),
    block: suspend () -> T
): Result<T> {
    var currentDelay = config.initialDelayMs
    var lastException: Throwable? = null
    
    repeat(config.maxRetries) { attempt ->
        try {
            return Result.Success(block())
        } catch (e: Throwable) {
            lastException = e
            
            // Check if should retry
            val shouldRetry = config.retryOn.any { it.isInstance(e) }
            
            if (!shouldRetry || attempt == config.maxRetries - 1) {
                return Result.Error(e, e.message)
            }
            
            // Exponential backoff
            delay(currentDelay)
            currentDelay = (currentDelay * config.backoffMultiplier)
                .toLong()
                .coerceAtMost(config.maxDelayMs)
        }
    }
    
    return Result.Error(
        lastException ?: AppError.UnknownError(),
        lastException?.message
    )
}

/**
 * Execute safely without retry
 */
suspend fun <T> safeCall(block: suspend () -> T): Result<T> {
    return try {
        Result.Success(block())
    } catch (e: Throwable) {
        Result.Error(e, e.message)
    }
}

/**
 * Map exceptions to AppError types
 */
fun Throwable.toAppError(): AppError {
    return when (this) {
        is AppError -> this
        is java.net.UnknownHostException -> AppError.NetworkError()
        is java.net.SocketTimeoutException -> AppError.TimeoutError()
        is java.io.IOException -> AppError.NetworkError("Connection error")
        else -> AppError.UnknownError(message ?: "Unknown error")
    }
}

/**
 * Extension to run and handle errors inline
 */
suspend inline fun <T> runCatching(
    onError: (AppError) -> Unit = {},
    crossinline block: suspend () -> T
): T? {
    return try {
        block()
    } catch (e: Throwable) {
        onError(e.toAppError())
        null
    }
}
