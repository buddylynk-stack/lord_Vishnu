package com.orignal.buddylynk.data.firebase

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.RingtoneManager
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage
import com.orignal.buddylynk.MainActivity
import com.orignal.buddylynk.R
import com.orignal.buddylynk.data.api.ApiService
import com.orignal.buddylynk.data.auth.AuthManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Firebase Cloud Messaging Service
 * Handles incoming push notifications for calls and messages
 */
class MyFirebaseMessagingService : FirebaseMessagingService() {

    companion object {
        private const val TAG = "FCMService"
        private const val CHANNEL_ID_CALLS = "buddylynk_calls"
        private const val CHANNEL_ID_MESSAGES = "buddylynk_messages"
        private const val NOTIFICATION_ID_CALL = 1001
        private const val NOTIFICATION_ID_MESSAGE = 1002
    }

    override fun onNewToken(token: String) {
        super.onNewToken(token)
        Log.d(TAG, "New FCM token: $token")
        
        // Save token locally and send to backend
        CoroutineScope(Dispatchers.IO).launch {
            try {
                // Store token locally
                val prefs = applicationContext.getSharedPreferences("fcm_prefs", Context.MODE_PRIVATE)
                prefs.edit().putString("fcm_token", token).apply()
                
                // Send to backend if user is logged in
                val userId = AuthManager.getCurrentUserId()
                if (userId != null) {
                    sendTokenToBackend(token)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save FCM token", e)
            }
        }
    }

    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        super.onMessageReceived(remoteMessage)
        Log.d(TAG, "Message received from: ${remoteMessage.from}")
        Log.d(TAG, "Message data: ${remoteMessage.data}")

        val data = remoteMessage.data
        val type = data["type"]

        when (type) {
            "incoming_call" -> handleIncomingCall(data)
            "message" -> handleNewMessage(data)
            else -> {
                // Handle notification payload
                remoteMessage.notification?.let { notification ->
                    showNotification(notification.title ?: "BuddyLynk", notification.body ?: "")
                }
            }
        }
    }

    private fun handleIncomingCall(data: Map<String, String>) {
        val callId = data["callId"] ?: ""
        val callerId = data["callerId"] ?: ""
        val callerName = data["callerName"] ?: "Unknown"
        val callerAvatar = data["callerAvatar"]
        val callType = data["callType"] ?: "voice"

        Log.d(TAG, "Incoming $callType call from $callerName ($callerId)")

        // Create notification channel for calls
        createCallNotificationChannel()

        // Intent to open the app to the call screen
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra("type", "incoming_call")
            putExtra("callId", callId)
            putExtra("callerId", callerId)
            putExtra("callerName", callerName)
            putExtra("callerAvatar", callerAvatar)
            putExtra("callType", callType)
        }

        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        // Use ringtone for incoming calls
        val ringtoneUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)

        // Build full-screen notification for incoming call
        val notification = NotificationCompat.Builder(this, CHANNEL_ID_CALLS)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("Incoming ${if (callType == "video") "Video" else "Voice"} Call")
            .setContentText("$callerName is calling...")
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setFullScreenIntent(pendingIntent, true)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .setSound(ringtoneUri)
            .setVibrate(longArrayOf(0, 1000, 500, 1000, 500, 1000))
            .setOngoing(true)
            .addAction(
                android.R.drawable.ic_menu_call,
                "Answer",
                pendingIntent
            )
            .addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                "Decline",
                getDeclineIntent(callId, callerId)
            )
            .build()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID_CALL, notification)
    }

    private fun handleNewMessage(data: Map<String, String>) {
        val senderName = data["senderName"] ?: "New message"
        val content = data["content"] ?: ""
        val senderId = data["senderId"] ?: ""

        createMessageNotificationChannel()

        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra("type", "message")
            putExtra("senderId", senderId)
        }

        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(this, CHANNEL_ID_MESSAGES)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(senderName)
            .setContentText(content)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .build()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID_MESSAGE, notification)
    }

    private fun showNotification(title: String, body: String) {
        createMessageNotificationChannel()

        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }

        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(this, CHANNEL_ID_MESSAGES)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(title)
            .setContentText(body)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .build()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(System.currentTimeMillis().toInt(), notification)
    }

    private fun createCallNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID_CALLS,
                "Incoming Calls",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Notifications for incoming video and voice calls"
                setSound(
                    RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE),
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_NOTIFICATION_RINGTONE)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                enableVibration(true)
                vibrationPattern = longArrayOf(0, 1000, 500, 1000, 500, 1000)
                lockscreenVisibility = NotificationCompat.VISIBILITY_PUBLIC
            }

            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun createMessageNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID_MESSAGES,
                "Messages",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "New message notifications"
            }

            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun getDeclineIntent(callId: String, callerId: String): PendingIntent {
        val intent = Intent(this, CallDeclineReceiver::class.java).apply {
            action = "DECLINE_CALL"
            putExtra("callId", callId)
            putExtra("callerId", callerId)
        }
        return PendingIntent.getBroadcast(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    private suspend fun sendTokenToBackend(token: String) {
        try {
            // This endpoint will be created on the backend
            ApiService.updateFcmToken(token)
            Log.d(TAG, "FCM token sent to backend successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send FCM token to backend", e)
        }
    }
}
