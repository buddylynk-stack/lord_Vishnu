/**
 * WebRTC Call Signaling Service
 * Uses Socket.io for real-time signaling
 * Uses in-memory storage for call state (Redis optional)
 */
const jwt = require('jsonwebtoken');

// In-memory storage for call state (works without Redis)
const inMemoryCallState = new Map();
const userSockets = new Map(); // userId -> socket.id

// Optional: Try to use Redis if REDIS_URL is set
let redis = null;
if (process.env.REDIS_URL) {
    try {
        const Redis = require('ioredis');
        redis = new Redis(process.env.REDIS_URL);
        redis.on('error', () => { redis = null; }); // Silently disable on error
        console.log('Redis connected for call signaling');
    } catch (e) {
        redis = null;
    }
}

/**
 * Initialize Socket.io signaling
 */
function initializeSignaling(io) {
    console.log('Initializing WebRTC signaling service...');

    io.on('connection', async (socket) => {
        console.log(`[Signaling] New connection: ${socket.id}`);

        let userId = null;

        // Authenticate user
        socket.on('authenticate', async (token) => {
            try {
                const decoded = jwt.verify(token, process.env.JWT_SECRET || 'buddylynk-secret-2024');
                userId = decoded.userId;

                // Store socket mapping
                userSockets.set(userId, socket.id);
                socket.userId = userId;

                // Store in Redis for presence
                if (redis) {
                    await redis.set(`call:user:${userId}`, socket.id, 'EX', 3600);
                    await redis.set(`presence:${userId}`, 'online', 'EX', 3600);
                }

                console.log(`[Signaling] User authenticated: ${userId}`);
                socket.emit('authenticated', { success: true });

            } catch (err) {
                console.log('[Signaling] Auth failed:', err.message);
                socket.emit('authenticated', { success: false, error: 'Invalid token' });
            }
        });

        // ======== CALL INITIATION ========

        socket.on('call:start', async (data) => {
            const { targetUserId, callType } = data; // callType: 'voice' or 'video'

            if (!userId) {
                socket.emit('call:error', { error: 'Not authenticated' });
                return;
            }

            console.log(`[Signaling] ${userId} starting ${callType} call to ${targetUserId}`);

            // Generate call ID
            const callId = `call_${Date.now()}_${userId}_${targetUserId}`;

            // Store call state
            const callState = {
                callId,
                callerId: userId,
                calleeId: targetUserId,
                callType,
                status: 'ringing',
                startedAt: new Date().toISOString()
            };

            if (redis) {
                await redis.set(`call:${callId}`, JSON.stringify(callState), 'EX', 600);
            } else {
                inMemoryCallState.set(callId, callState);
            }

            // Get target user's socket
            let targetSocketId = userSockets.get(targetUserId);
            if (!targetSocketId && redis) {
                targetSocketId = await redis.get(`call:user:${targetUserId}`);
            }

            if (targetSocketId) {
                // User is online - send via Socket.IO
                io.to(targetSocketId).emit('call:incoming', {
                    callId,
                    callerId: userId,
                    callType
                });
            } else {
                // User is offline - send FCM push notification
                console.log(`[Signaling] Target user ${targetUserId} is offline - sending FCM push`);
                try {
                    const { sendCallNotification, getUserFcmToken } = require('./fcm');
                    const { GetCommand } = require('@aws-sdk/lib-dynamodb');
                    const { docClient, Tables } = require('../config/aws');

                    // Get caller info for notification
                    const callerResult = await docClient.send(new GetCommand({
                        TableName: Tables.USERS,
                        Key: { userId },
                        ProjectionExpression: 'username, avatar'
                    }));

                    const callerName = callerResult.Item?.username || 'Unknown';
                    const callerAvatar = callerResult.Item?.avatar || '';

                    const success = await sendCallNotification(
                        targetUserId,
                        userId,
                        callerName,
                        callerAvatar,
                        callId,
                        callType
                    );

                    if (!success) {
                        socket.emit('call:error', { error: 'User is offline and has no push notifications' });
                        return;
                    }
                } catch (fcmError) {
                    console.error('[Signaling] FCM error:', fcmError);
                    socket.emit('call:error', { error: 'User is offline' });
                    return;
                }
            }

            // Confirm to caller
            socket.emit('call:started', { callId, callType });
        });

        // ======== CALL ANSWER/REJECT ========

        socket.on('call:answer', async (data) => {
            const { callId, accept } = data;

            // Get call state
            let callState;
            if (redis) {
                const stateJson = await redis.get(`call:${callId}`);
                callState = stateJson ? JSON.parse(stateJson) : null;
            } else {
                callState = inMemoryCallState.get(callId);
            }

            if (!callState) {
                socket.emit('call:error', { error: 'Call not found' });
                return;
            }

            // Get caller's socket
            const callerSocketId = userSockets.get(callState.callerId);

            if (accept) {
                console.log(`[Signaling] Call ${callId} accepted`);
                callState.status = 'connected';
                callState.connectedAt = new Date().toISOString();

                if (redis) {
                    await redis.set(`call:${callId}`, JSON.stringify(callState), 'EX', 3600);
                }

                // Notify caller
                if (callerSocketId) {
                    io.to(callerSocketId).emit('call:answered', { callId, accepted: true });
                }
                socket.emit('call:connected', { callId });

            } else {
                console.log(`[Signaling] Call ${callId} rejected`);

                if (redis) {
                    await redis.del(`call:${callId}`);
                } else {
                    inMemoryCallState.delete(callId);
                }

                if (callerSocketId) {
                    io.to(callerSocketId).emit('call:answered', { callId, accepted: false });
                }
            }
        });

        // ======== WEBRTC SIGNALING ========

        socket.on('webrtc:offer', async (data) => {
            const { callId, targetUserId, sdp } = data;
            console.log(`[Signaling] SDP offer from ${userId} to ${targetUserId}`);

            const targetSocketId = userSockets.get(targetUserId);
            if (targetSocketId) {
                io.to(targetSocketId).emit('webrtc:offer', { callId, sdp, fromUserId: userId });
            }
        });

        socket.on('webrtc:answer', async (data) => {
            const { callId, targetUserId, sdp } = data;
            console.log(`[Signaling] SDP answer from ${userId} to ${targetUserId}`);

            const targetSocketId = userSockets.get(targetUserId);
            if (targetSocketId) {
                io.to(targetSocketId).emit('webrtc:answer', { callId, sdp, fromUserId: userId });
            }
        });

        socket.on('webrtc:ice-candidate', async (data) => {
            const { callId, targetUserId, candidate } = data;

            const targetSocketId = userSockets.get(targetUserId);
            if (targetSocketId) {
                io.to(targetSocketId).emit('webrtc:ice-candidate', {
                    callId,
                    candidate,
                    fromUserId: userId
                });
            }
        });

        // ======== CALL END ========

        socket.on('call:end', async (data) => {
            const { callId, targetUserId } = data;
            console.log(`[Signaling] Call ${callId} ended by ${userId}`);

            // Clean up call state
            if (redis) {
                await redis.del(`call:${callId}`);
            } else {
                inMemoryCallState.delete(callId);
            }

            // Notify other party
            const targetSocketId = userSockets.get(targetUserId);
            if (targetSocketId) {
                io.to(targetSocketId).emit('call:ended', { callId, endedBy: userId });
            }
        });

        // ======== DISCONNECT ========

        socket.on('disconnect', async () => {
            console.log(`[Signaling] Disconnected: ${socket.id}, userId: ${userId}`);

            if (userId) {
                userSockets.delete(userId);

                if (redis) {
                    await redis.del(`call:user:${userId}`);
                    await redis.set(`presence:${userId}`, 'offline');
                }
            }
        });
    });

    console.log('WebRTC signaling service initialized');
}

/**
 * Get user online status
 */
async function isUserOnline(userId) {
    if (userSockets.has(userId)) return true;
    if (redis) {
        const socketId = await redis.get(`call:user:${userId}`);
        return !!socketId;
    }
    return false;
}

module.exports = { initializeSignaling, isUserOnline };
