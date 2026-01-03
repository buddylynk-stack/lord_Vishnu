/**
 * Firebase Cloud Messaging Service
 * Sends push notifications for incoming calls and messages
 */
const { GetCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const path = require('path');

// Firebase Admin SDK - will be initialized if available
let admin = null;
try {
    admin = require('firebase-admin');
    
    // Initialize Firebase Admin with service account
    if (!admin.apps.length) {
        const serviceAccountPath = path.join(__dirname, '..', 'firebase-service-account.json');
        const serviceAccount = require(serviceAccountPath);
        
        admin.initializeApp({
            credential: admin.credential.cert(serviceAccount)
        });
        console.log('Firebase Admin SDK initialized with service account');
    }
} catch (e) {
    console.log('firebase-admin not available:', e.message);
    console.log('Push notifications will be disabled. Run: npm install firebase-admin');
    admin = null;
}

/**
 * Get user's FCM token from database
 */
async function getUserFcmToken(userId) {
    try {
        const result = await docClient.send(new GetCommand({
            TableName: Tables.USERS,
            Key: { userId },
            ProjectionExpression: 'fcmToken, username, avatar'
        }));
        return result.Item || null;
    } catch (e) {
        console.error('Error getting user FCM token:', e);
        return null;
    }
}

/**
 * Send push notification for incoming call
 */
async function sendCallNotification(targetUserId, callerId, callerName, callerAvatar, callId, callType) {
    if (!admin) {
        console.log('Firebase Admin not available - cannot send push notification');
        return false;
    }

    const targetUser = await getUserFcmToken(targetUserId);
    if (!targetUser || !targetUser.fcmToken) {
        console.log(`No FCM token for user ${targetUserId}`);
        return false;
    }

    try {
        const message = {
            token: targetUser.fcmToken,
            data: {
                type: 'incoming_call',
                callId: callId,
                callerId: callerId,
                callerName: callerName || 'Unknown',
                callerAvatar: callerAvatar || '',
                callType: callType || 'voice'
            },
            android: {
                priority: 'high',
                ttl: 30000, // 30 seconds
                notification: {
                    channelId: 'buddylynk_calls',
                    priority: 'max',
                    defaultSound: true,
                    defaultVibrateTimings: true,
                    visibility: 'public'
                }
            }
        };

        const response = await admin.messaging().send(message);
        console.log(`Call notification sent to ${targetUserId}:`, response);
        return true;
    } catch (e) {
        console.error('Error sending call notification:', e);
        return false;
    }
}

/**
 * Send push notification for new message
 */
async function sendMessageNotification(targetUserId, senderId, senderName, content) {
    if (!admin) {
        return false;
    }

    const targetUser = await getUserFcmToken(targetUserId);
    if (!targetUser || !targetUser.fcmToken) {
        return false;
    }

    try {
        const message = {
            token: targetUser.fcmToken,
            notification: {
                title: senderName || 'New Message',
                body: content.substring(0, 100) // Limit content length
            },
            data: {
                type: 'message',
                senderId: senderId,
                senderName: senderName || ''
            },
            android: {
                priority: 'high',
                notification: {
                    channelId: 'buddylynk_messages'
                }
            }
        };

        await admin.messaging().send(message);
        return true;
    } catch (e) {
        console.error('Error sending message notification:', e);
        return false;
    }
}

module.exports = {
    sendCallNotification,
    sendMessageNotification,
    getUserFcmToken
};
