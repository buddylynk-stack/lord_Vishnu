package com.orignal.buddylynk.data.firebase

import android.app.NotificationManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.orignal.buddylynk.data.calls.CallManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Broadcast receiver to handle declining calls from notification
 */
class CallDeclineReceiver : BroadcastReceiver() {
    
    companion object {
        private const val TAG = "CallDeclineReceiver"
    }
    
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == "DECLINE_CALL") {
            val callId = intent.getStringExtra("callId") ?: ""
            val callerId = intent.getStringExtra("callerId") ?: ""
            
            Log.d(TAG, "Declining call: $callId from $callerId")
            
            // Cancel the notification
            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.cancel(1001)
            
            // End the call via CallManager
            CoroutineScope(Dispatchers.Main).launch {
                CallManager.endCall(sendSignal = true)
            }
        }
    }
}
