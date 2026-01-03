package com.orignal.buddylynk.data.share

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URL

/**
 * Share Service - Share content to other apps
 */
object ShareService {
    
    /**
     * Share text content
     */
    fun shareText(context: Context, text: String, title: String = "Share") {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, text)
        }
        context.startActivity(Intent.createChooser(intent, title))
    }
    
    /**
     * Share post link
     */
    fun sharePostLink(context: Context, postId: String, postContent: String?) {
        val deepLink = "https://app.buddylynk.com/post/$postId"
        val shareText = buildString {
            postContent?.take(100)?.let { 
                append(it)
                if (it.length >= 100) append("...")
                append("\n\n")
            }
            append("Check out this post on BuddyLynk!\n")
            append(deepLink)
        }
        
        shareText(context, shareText, "Share Post")
    }
    
    /**
     * Share user profile link
     */
    fun shareProfileLink(context: Context, userId: String, username: String) {
        val deepLink = "https://app.buddylynk.com/user/$userId"
        val shareText = "Check out @$username on BuddyLynk!\n$deepLink"
        
        shareText(context, shareText, "Share Profile")
    }
    
    /**
     * Share image
     */
    suspend fun shareImage(
        context: Context,
        imageUrl: String,
        caption: String? = null
    ) = withContext(Dispatchers.IO) {
        try {
            // Download image
            val url = URL(imageUrl)
            val connection = url.openConnection()
            connection.connect()
            
            val inputStream = connection.getInputStream()
            val fileName = "share_${System.currentTimeMillis()}.jpg"
            val file = File(context.cacheDir, fileName)
            
            file.outputStream().use { output ->
                inputStream.copyTo(output)
            }
            
            val uri = FileProvider.getUriForFile(
                context,
                "${context.packageName}.provider",
                file
            )
            
            withContext(Dispatchers.Main) {
                val intent = Intent(Intent.ACTION_SEND).apply {
                    type = "image/*"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    caption?.let { putExtra(Intent.EXTRA_TEXT, it) }
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                context.startActivity(Intent.createChooser(intent, "Share Image"))
            }
            
            true
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Copy link to clipboard
     */
    fun copyLink(context: Context, link: String): Boolean {
        return try {
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
            val clip = android.content.ClipData.newPlainText("BuddyLynk Link", link)
            clipboard.setPrimaryClip(clip)
            true
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Generate post deep link
     */
    fun getPostDeepLink(postId: String): String {
        return "buddylynk://post/$postId"
    }
    
    /**
     * Generate profile deep link
     */
    fun getProfileDeepLink(userId: String): String {
        return "buddylynk://user/$userId"
    }
    
    /**
     * Generate web link
     */
    fun getWebLink(path: String): String {
        return "https://app.buddylynk.com/$path"
    }
}

/**
 * Deep Link Handler
 */
object DeepLinkHandler {
    
    /**
     * Parse deep link to navigation destination
     */
    fun parseDeepLink(uri: Uri): DeepLinkDestination? {
        val scheme = uri.scheme
        val host = uri.host
        val path = uri.pathSegments
        
        // Handle buddylynk:// scheme
        if (scheme == "buddylynk") {
            return when (host) {
                "post" -> path.firstOrNull()?.let { DeepLinkDestination.Post(it) }
                "user" -> path.firstOrNull()?.let { DeepLinkDestination.Profile(it) }
                "chat" -> path.firstOrNull()?.let { DeepLinkDestination.Chat(it) }
                "activity" -> DeepLinkDestination.Activity
                else -> null
            }
        }
        
        // Handle https://buddylynk.app scheme
        if (scheme in listOf("http", "https") && host == "app.buddylynk.com") {
            return when (path.firstOrNull()) {
                "post" -> path.getOrNull(1)?.let { DeepLinkDestination.Post(it) }
                "user" -> path.getOrNull(1)?.let { DeepLinkDestination.Profile(it) }
                "chat" -> path.getOrNull(1)?.let { DeepLinkDestination.Chat(it) }
                else -> null
            }
        }
        
        return null
    }
}

/**
 * Deep link destinations
 */
sealed class DeepLinkDestination {
    data class Post(val postId: String) : DeepLinkDestination()
    data class Profile(val userId: String) : DeepLinkDestination()
    data class Chat(val conversationId: String) : DeepLinkDestination()
    data object Activity : DeepLinkDestination()
}
