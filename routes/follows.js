const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { PutCommand, DeleteCommand, ScanCommand, UpdateCommand, GetCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');

const router = express.Router();

// Follow a user
router.post('/:userId', verifyToken, async (req, res) => {
    try {
        const targetUserId = req.params.userId;
        const followerId = req.userId;

        console.log(`Follow request: ${followerId} following ${targetUserId}`);

        if (targetUserId === followerId) {
            return res.status(400).json({ error: 'Cannot follow yourself' });
        }

        // Use composite key: followerId (HASH) + followingId (RANGE)
        await docClient.send(new PutCommand({
            TableName: Tables.FOLLOWS,
            Item: {
                followerId: followerId,
                followingId: targetUserId,
                createdAt: new Date().toISOString()
            }
        }));

        // Update follower/following counts
        try {
            await docClient.send(new UpdateCommand({
                TableName: Tables.USERS,
                Key: { userId: targetUserId },
                UpdateExpression: 'SET followerCount = if_not_exists(followerCount, :zero) + :inc',
                ExpressionAttributeValues: { ':inc': 1, ':zero': 0 }
            }));
        } catch (e) { console.log('Could not update target followerCount:', e.message); }

        try {
            await docClient.send(new UpdateCommand({
                TableName: Tables.USERS,
                Key: { userId: followerId },
                UpdateExpression: 'SET followingCount = if_not_exists(followingCount, :zero) + :inc',
                ExpressionAttributeValues: { ':inc': 1, ':zero': 0 }
            }));
        } catch (e) { console.log('Could not update follower followingCount:', e.message); }

        console.log('Follow successful');
        res.json({ success: true });
    } catch (err) {
        console.error('Follow error:', err);
        res.status(500).json({ error: 'Failed to follow user' });
    }
});

// Unfollow a user
router.delete('/:userId', verifyToken, async (req, res) => {
    try {
        const targetUserId = req.params.userId;
        const followerId = req.userId;

        console.log(`Unfollow: ${followerId} unfollowing ${targetUserId}`);

        // Use composite key: followerId (HASH) + followingId (RANGE)
        await docClient.send(new DeleteCommand({
            TableName: Tables.FOLLOWS,
            Key: {
                followerId: followerId,
                followingId: targetUserId
            }
        }));

        // Update counts - wrap in try-catch to not fail if counts don't exist
        try {
            await docClient.send(new UpdateCommand({
                TableName: Tables.USERS,
                Key: { userId: targetUserId },
                UpdateExpression: 'SET followerCount = if_not_exists(followerCount, :one) - :dec',
                ExpressionAttributeValues: { ':dec': 1, ':one': 1 }
            }));
        } catch (e) { console.log('Could not update target followerCount:', e.message); }

        try {
            await docClient.send(new UpdateCommand({
                TableName: Tables.USERS,
                Key: { userId: followerId },
                UpdateExpression: 'SET followingCount = if_not_exists(followingCount, :one) - :dec',
                ExpressionAttributeValues: { ':dec': 1, ':one': 1 }
            }));
        } catch (e) { console.log('Could not update follower followingCount:', e.message); }

        console.log('Unfollow successful');
        res.json({ success: true });
    } catch (err) {
        console.error('Unfollow error:', err);
        res.status(500).json({ error: 'Failed to unfollow user' });
    }
});

// Get followers of a user
router.get('/:userId/followers', async (req, res) => {
    try {
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.FOLLOWS,
            FilterExpression: 'followingId = :uid',
            ExpressionAttributeValues: { ':uid': req.params.userId }
        }));

        res.json(result.Items || []);
    } catch (err) {
        console.error('Get followers error:', err);
        res.status(500).json({ error: 'Failed to get followers' });
    }
});

// Get following of a user
router.get('/:userId/following', async (req, res) => {
    try {
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.FOLLOWS,
            FilterExpression: 'followerId = :uid',
            ExpressionAttributeValues: { ':uid': req.params.userId }
        }));

        res.json(result.Items || []);
    } catch (err) {
        console.error('Get following error:', err);
        res.status(500).json({ error: 'Failed to get following' });
    }
});

// Check if following
router.get('/check/:userId', verifyToken, async (req, res) => {
    try {
        // Use composite key: followerId (HASH) + followingId (RANGE)
        const result = await docClient.send(new GetCommand({
            TableName: Tables.FOLLOWS,
            Key: {
                followerId: req.userId,
                followingId: req.params.userId
            }
        }));

        res.json({ isFollowing: !!result.Item });
    } catch (err) {
        console.error('Check follow error:', err);
        res.status(500).json({ error: 'Failed to check follow status' });
    }
});

module.exports = router;
