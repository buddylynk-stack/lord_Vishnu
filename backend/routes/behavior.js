/**
 * Behavior Tracking Routes for MindFlow Algorithm
 * Tracks user interactions: view, like, share, comment, save, skip
 * Data stored in Buddylynk_UserBehavior DynamoDB table
 */

const express = require('express');
const router = express.Router();
const { DynamoDBClient, PutItemCommand, QueryCommand, BatchWriteItemCommand } = require('@aws-sdk/client-dynamodb');

// Action types for the algorithm
const ACTION_TYPES = {
    VIEW: 0,
    LIKE: 1,
    SHARE: 2,
    COMMENT: 3,
    SAVE: 4,
    SKIP: 5,
    UNLIKE: 6,
    UNSAVE: 7
};

// Action weights (higher = stronger signal)
const ACTION_WEIGHTS = {
    0: 1.0,   // VIEW - base engagement
    1: 3.0,   // LIKE - strong positive
    2: 5.0,   // SHARE - very strong positive (willing to associate publicly)
    3: 4.0,   // COMMENT - high engagement
    4: 4.0,   // SAVE - strong intent to revisit
    5: -0.5,  // SKIP - mild negative
    6: -2.0,  // UNLIKE - negative signal
    7: -2.0   // UNSAVE - negative signal
};

// DynamoDB client
const dynamodb = new DynamoDBClient({ region: 'us-east-1' });

const USER_BEHAVIOR_TABLE = 'Buddylynk_UserBehavior';
const USER_EMBEDDINGS_TABLE = 'Buddylynk_UserEmbeddings';
const CONTENT_FEATURES_TABLE = 'Buddylynk_ContentFeatures';

/**
 * Track user behavior - called when user interacts with content
 * POST /api/behavior/track
 */
router.post('/track', async (req, res) => {
    try {
        const {
            userId,
            contentId,
            contentOwnerId,  // Who created the content (to filter out own posts)
            actionType,  // 'view', 'like', 'share', 'comment', 'save', 'skip'
            watchTime,   // For video content (seconds)
            metadata     // Additional info (comment text length, share platform, etc.)
        } = req.body;

        if (!userId || !contentId || actionType === undefined) {
            return res.status(400).json({
                success: false,
                error: 'userId, contentId, and actionType are required'
            });
        }

        // Map action string to number
        const actionNum = typeof actionType === 'string'
            ? ACTION_TYPES[actionType.toUpperCase()] ?? 0
            : actionType;

        const now = Date.now();
        const date = new Date(now);

        const item = {
            userId: { S: userId },
            timestamp: { N: now.toString() },
            contentId: { S: contentId },
            contentOwnerId: { S: contentOwnerId || '' },  // Store content creator ID
            isNSFW: { BOOL: metadata?.isNSFW || false },  // 18+ content flag
            actionType: { N: actionNum.toString() },
            actionWeight: { N: (ACTION_WEIGHTS[actionNum] || 1.0).toString() },
            hour: { N: date.getHours().toString() },
            dayOfWeek: { N: date.getDay().toString() },
            watchTime: { N: (watchTime || 0).toString() },
            TTL: { N: Math.floor(now / 1000 + 90 * 24 * 60 * 60).toString() } // 90 days
        };

        // Add metadata if provided
        if (metadata) {
            item.metadata = { S: JSON.stringify(metadata) };
        }

        await dynamodb.send(new PutItemCommand({
            TableName: USER_BEHAVIOR_TABLE,
            Item: item
        }));

        console.log(`[Behavior] ${userId} ${Object.keys(ACTION_TYPES).find(k => ACTION_TYPES[k] === actionNum)} ${contentId}`);

        res.json({
            success: true,
            message: 'Behavior tracked',
            action: Object.keys(ACTION_TYPES).find(k => ACTION_TYPES[k] === actionNum),
            weight: ACTION_WEIGHTS[actionNum]
        });

    } catch (error) {
        console.error('[Behavior] Track error:', error);
        res.status(500).json({ success: false, error: 'Failed to track behavior' });
    }
});

/**
 * Batch track multiple behaviors at once
 * POST /api/behavior/batch
 */
router.post('/batch', async (req, res) => {
    try {
        const { behaviors } = req.body;

        if (!Array.isArray(behaviors) || behaviors.length === 0) {
            return res.status(400).json({ success: false, error: 'behaviors array required' });
        }

        // Process in batches of 25 (DynamoDB limit)
        const batches = [];
        for (let i = 0; i < behaviors.length; i += 25) {
            batches.push(behaviors.slice(i, i + 25));
        }

        let totalWritten = 0;

        for (const batch of batches) {
            const writeRequests = batch.map(b => {
                const now = Date.now();
                const date = new Date(now);
                const actionNum = typeof b.actionType === 'string'
                    ? ACTION_TYPES[b.actionType.toUpperCase()] ?? 0
                    : b.actionType;

                return {
                    PutRequest: {
                        Item: {
                            userId: { S: b.userId },
                            timestamp: { N: (now + Math.random() * 1000).toString() }, // Add jitter for uniqueness
                            contentId: { S: b.contentId },
                            actionType: { N: actionNum.toString() },
                            actionWeight: { N: (ACTION_WEIGHTS[actionNum] || 1.0).toString() },
                            hour: { N: date.getHours().toString() },
                            dayOfWeek: { N: date.getDay().toString() },
                            watchTime: { N: (b.watchTime || 0).toString() },
                            TTL: { N: Math.floor(now / 1000 + 90 * 24 * 60 * 60).toString() }
                        }
                    }
                };
            });

            await dynamodb.send(new BatchWriteItemCommand({
                RequestItems: {
                    [USER_BEHAVIOR_TABLE]: writeRequests
                }
            }));

            totalWritten += writeRequests.length;
        }

        res.json({ success: true, tracked: totalWritten });

    } catch (error) {
        console.error('[Behavior] Batch error:', error);
        res.status(500).json({ success: false, error: 'Failed to batch track' });
    }
});

/**
 * Get user behavior history
 * GET /api/behavior/history/:userId
 */
router.get('/history/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        const { limit = 100, days = 30 } = req.query;

        const startTime = Date.now() - (days * 24 * 60 * 60 * 1000);

        const result = await dynamodb.send(new QueryCommand({
            TableName: USER_BEHAVIOR_TABLE,
            KeyConditionExpression: 'userId = :uid AND #ts >= :start',
            ExpressionAttributeNames: { '#ts': 'timestamp' },
            ExpressionAttributeValues: {
                ':uid': { S: userId },
                ':start': { N: startTime.toString() }
            },
            Limit: parseInt(limit),
            ScanIndexForward: false // Most recent first
        }));

        const history = (result.Items || []).map(item => ({
            contentId: item.contentId.S,
            actionType: parseInt(item.actionType.N),
            actionName: Object.keys(ACTION_TYPES).find(k => ACTION_TYPES[k] === parseInt(item.actionType.N)),
            timestamp: parseInt(item.timestamp.N),
            watchTime: parseInt(item.watchTime?.N || 0),
            hour: parseInt(item.hour.N),
            dayOfWeek: parseInt(item.dayOfWeek.N)
        }));

        // Calculate user preferences from history
        const actionCounts = {};
        let totalWeight = 0;

        history.forEach(h => {
            actionCounts[h.actionName] = (actionCounts[h.actionName] || 0) + 1;
            totalWeight += ACTION_WEIGHTS[h.actionType] || 0;
        });

        res.json({
            success: true,
            userId,
            count: history.length,
            stats: {
                actionCounts,
                totalWeight,
                averageWeight: history.length > 0 ? totalWeight / history.length : 0
            },
            history
        });

    } catch (error) {
        console.error('[Behavior] History error:', error);
        res.status(500).json({ success: false, error: 'Failed to get history' });
    }
});

/**
 * Get content engagement stats
 * GET /api/behavior/content/:contentId
 */
router.get('/content/:contentId', async (req, res) => {
    try {
        const { contentId } = req.params;

        const result = await dynamodb.send(new QueryCommand({
            TableName: USER_BEHAVIOR_TABLE,
            IndexName: 'contentId-timestamp-index',
            KeyConditionExpression: 'contentId = :cid',
            ExpressionAttributeValues: {
                ':cid': { S: contentId }
            },
            Limit: 1000
        }));

        // Aggregate stats
        const stats = {
            views: 0,
            likes: 0,
            shares: 0,
            comments: 0,
            saves: 0,
            skips: 0,
            uniqueUsers: new Set(),
            totalWatchTime: 0
        };

        (result.Items || []).forEach(item => {
            const action = parseInt(item.actionType.N);
            stats.uniqueUsers.add(item.userId.S);
            stats.totalWatchTime += parseInt(item.watchTime?.N || 0);

            switch (action) {
                case 0: stats.views++; break;
                case 1: stats.likes++; break;
                case 2: stats.shares++; break;
                case 3: stats.comments++; break;
                case 4: stats.saves++; break;
                case 5: stats.skips++; break;
            }
        });

        // Calculate engagement score (0-100)
        const engagementScore = stats.views > 0
            ? Math.min(100, Math.round(
                ((stats.likes * 3 + stats.shares * 5 + stats.comments * 4 + stats.saves * 4) / stats.views) * 20
            ))
            : 0;

        res.json({
            success: true,
            contentId,
            stats: {
                ...stats,
                uniqueUsers: stats.uniqueUsers.size,
                engagementScore,
                avgWatchTime: stats.views > 0 ? Math.round(stats.totalWatchTime / stats.views) : 0
            }
        });

    } catch (error) {
        console.error('[Behavior] Content stats error:', error);
        res.status(500).json({ success: false, error: 'Failed to get content stats' });
    }
});

/**
 * Get action type constants
 * GET /api/behavior/types
 */
router.get('/types', (req, res) => {
    res.json({
        success: true,
        actionTypes: ACTION_TYPES,
        actionWeights: ACTION_WEIGHTS
    });
});

module.exports = router;
