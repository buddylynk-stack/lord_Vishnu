package com.orignal.buddylynk.data.aws

import android.content.Context
import android.net.Uri
import android.util.Log
import com.orignal.buddylynk.data.api.ApiService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.InputStream
import java.util.UUID

/**
 * S3Service - Uploads media via backend API pre-signed URLs
 * 
 * This is the API-first implementation that:
 * 1. Gets a pre-signed URL from the backend
 * 2. Uploads the file directly to S3 using that URL
 * 3. Returns the final public file URL
 */
object S3Service {
    
    private const val TAG = "S3Service"
    
    /**
     * Upload profile image using pre-signed URL
     */
    suspend fun uploadProfileImage(
        context: Context,
        userId: String,
        imageUri: Uri
    ): String? = uploadMedia(context, imageUri, "avatars", "image/jpeg")
    
    /**
     * Variant without userId
     */
    suspend fun uploadProfileImage(
        context: Context,
        imageUri: Uri
    ): String? = uploadMedia(context, imageUri, "avatars", "image/jpeg")
    
    /**
     * Upload post media
     */
    suspend fun uploadPostMedia(
        context: Context,
        postId: String,
        mediaUri: Uri,
        isVideo: Boolean = false
    ): String? {
        val contentType = if (isVideo) "video/mp4" else "image/jpeg"
        val folder = "posts"
        return uploadMedia(context, mediaUri, folder, contentType)
    }
    
    /**
     * Variant with mediaType string
     */
    suspend fun uploadPostMedia(
        context: Context,
        mediaUri: Uri,
        mediaType: String = "image"
    ): String? {
        val contentType = if (mediaType == "video") "video/mp4" else "image/jpeg"
        return uploadMedia(context, mediaUri, "posts", contentType)
    }
    
    /**
     * Upload story media
     */
    suspend fun uploadStoryMedia(
        context: Context,
        storyId: String,
        mediaUri: Uri,
        isVideo: Boolean = false
    ): String? {
        val contentType = if (isVideo) "video/mp4" else "image/jpeg"
        return uploadMedia(context, mediaUri, "stories", contentType)
    }
    
    /**
     * Variant without storyId
     */
    suspend fun uploadStoryMedia(
        context: Context,
        mediaUri: Uri,
        mediaType: String = "image"
    ): String? {
        val contentType = if (mediaType == "video") "video/mp4" else "image/jpeg"
        return uploadMedia(context, mediaUri, "stories", contentType)
    }
    
    /**
     * Upload message media
     */
    suspend fun uploadMessageMedia(
        context: Context,
        conversationId: String,
        mediaUri: Uri
    ): String? = uploadMedia(context, mediaUri, "messages", "image/jpeg")
    
    /**
     * Variant without conversationId
     */
    suspend fun uploadMessageMedia(
        context: Context,
        mediaUri: Uri,
        mediaType: String = "image"
    ): String? {
        val contentType = if (mediaType == "video") "video/mp4" else "image/jpeg"
        return uploadMedia(context, mediaUri, "messages", contentType)
    }
    
    /**
     * Upload image from Uri (legacy API)
     */
    suspend fun uploadImage(
        context: Context,
        imageUri: Uri,
        folder: String = "posts"
    ): String? = uploadMedia(context, imageUri, folder, "image/jpeg")
    
    /**
     * Upload video from Uri
     */
    suspend fun uploadVideo(
        context: Context,
        videoUri: Uri,
        folder: String = "posts"
    ): String? = uploadMedia(context, videoUri, folder, "video/mp4")
    
    /**
     * Upload from InputStream
     */
    suspend fun uploadFile(
        inputStream: InputStream,
        filename: String,
        contentType: String,
        folder: String = "posts"
    ): String? = withContext(Dispatchers.IO) {
        try {
            val data = inputStream.readBytes()
            uploadBytes(data, filename, contentType, folder)
        } catch (e: Exception) {
            Log.e(TAG, "uploadFile failed: ${e.message}")
            null
        }
    }
    
    /**
     * Upload from bytes
     */
    suspend fun uploadBytes(
        data: ByteArray,
        filename: String,
        contentType: String,
        folder: String = "posts"
    ): String? = withContext(Dispatchers.IO) {
        try {
            // Get pre-signed URL from backend
            val presignResult = ApiService.getPresignedUrl(filename, contentType, folder)
            if (presignResult.isFailure) {
                Log.e(TAG, "Failed to get pre-signed URL")
                return@withContext null
            }
            
            val presignJson = presignResult.getOrNull() ?: return@withContext null
            val uploadUrl = presignJson.optString("uploadUrl", "")
            val fileUrl = presignJson.optString("fileUrl", "")
            
            if (uploadUrl.isEmpty() || fileUrl.isEmpty()) {
                Log.e(TAG, "Invalid pre-signed URL response")
                return@withContext null
            }
            
            // Upload to S3 using pre-signed URL
            val uploadResult = ApiService.uploadToPresignedUrl(uploadUrl, data, contentType)
            if (uploadResult.isSuccess && uploadResult.getOrNull() == true) {
                Log.d(TAG, "Upload successful: $fileUrl")
                fileUrl
            } else {
                Log.e(TAG, "Upload to S3 failed")
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "uploadBytes failed: ${e.message}")
            null
        }
    }
    
    /**
     * Core upload function - uploads a Uri using pre-signed URL
     */
    private suspend fun uploadMedia(
        context: Context,
        mediaUri: Uri,
        folder: String,
        contentType: String
    ): String? = withContext(Dispatchers.IO) {
        try {
            // Generate unique filename
            val extension = when {
                contentType.contains("video") -> "mp4"
                contentType.contains("png") -> "png"
                else -> "jpg"
            }
            val filename = "${UUID.randomUUID()}.$extension"
            
            // Read file bytes from Uri
            val inputStream = context.contentResolver.openInputStream(mediaUri)
            if (inputStream == null) {
                Log.e(TAG, "Failed to open input stream for Uri: $mediaUri")
                return@withContext null
            }
            
            val data = inputStream.use { it.readBytes() }
            Log.d(TAG, "Read ${data.size} bytes from Uri")
            
            // Get pre-signed URL from backend
            val presignResult = ApiService.getPresignedUrl(filename, contentType, folder)
            if (presignResult.isFailure) {
                Log.e(TAG, "Failed to get pre-signed URL: ${presignResult.exceptionOrNull()?.message}")
                return@withContext null
            }
            
            val presignJson = presignResult.getOrNull() ?: return@withContext null
            val uploadUrl = presignJson.optString("uploadUrl", "")
            val fileUrl = presignJson.optString("fileUrl", "")
            
            if (uploadUrl.isEmpty()) {
                Log.e(TAG, "Empty upload URL from backend")
                return@withContext null
            }
            
            Log.d(TAG, "Got pre-signed URL, uploading ${data.size} bytes...")
            
            // Upload to S3 using pre-signed URL
            val uploadResult = ApiService.uploadToPresignedUrl(uploadUrl, data, contentType)
            if (uploadResult.isSuccess && uploadResult.getOrNull() == true) {
                Log.d(TAG, "Upload successful! File URL: $fileUrl")
                fileUrl
            } else {
                Log.e(TAG, "Upload to S3 failed: ${uploadResult.exceptionOrNull()?.message}")
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "uploadMedia error: ${e.message}", e)
            null
        }
    }
    
    /**
     * Delete a file (not implemented via API)
     */
    suspend fun deleteFile(fileUrl: String): Boolean {
        // File deletion not implemented - would need backend API endpoint
        Log.w(TAG, "deleteFile not implemented")
        return false
    }
}
