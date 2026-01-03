const express = require('express');
const { GetCommand, UpdateCommand, ScanCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');
const { convertUserMediaUrls } = require('../utils/cloudfront');

const router = express.Router();

// Get current user profile
router.get('/me', verifyToken, async (req, res) => {
    try {
        const result = await docClient.send(new GetCommand({
            TableName: Tables.USERS,
            Key: { userId: req.userId }
        }));

        if (!result.Item) {
            return res.status(404).json({ error: 'User not found' });
        }

        delete result.Item.password;
        res.json(convertUserMediaUrls(result.Item));
    } catch (err) {
        console.error('Get user error:', err);
        res.status(500).json({ error: 'Failed to get user' });
    }
});

// Get user by ID
router.get('/:userId', async (req, res) => {
    try {
        const result = await docClient.send(new GetCommand({
            TableName: Tables.USERS,
            Key: { userId: req.params.userId }
        }));

        if (!result.Item) {
            return res.status(404).json({ error: 'User not found' });
        }

        delete result.Item.password;
        res.json(convertUserMediaUrls(result.Item));
    } catch (err) {
        console.error('Get user error:', err);
        res.status(500).json({ error: 'Failed to get user' });
    }
});

// Update user profile
router.put('/me', verifyToken, async (req, res) => {
    try {
        const { fullName, bio, avatar, username } = req.body;
        const updates = {};
        const expressions = [];
        const values = {};

        if (fullName) { updates.fullName = fullName; expressions.push('fullName = :fn'); values[':fn'] = fullName; }
        if (bio !== undefined) { updates.bio = bio; expressions.push('bio = :bio'); values[':bio'] = bio; }
        if (avatar) { updates.avatar = avatar; expressions.push('avatar = :av'); values[':av'] = avatar; }
        if (username) { updates.username = username; expressions.push('username = :un'); values[':un'] = username.toLowerCase(); }

        expressions.push('updatedAt = :ua');
        values[':ua'] = new Date().toISOString();

        await docClient.send(new UpdateCommand({
            TableName: Tables.USERS,
            Key: { userId: req.userId },
            UpdateExpression: 'SET ' + expressions.join(', '),
            ExpressionAttributeValues: values
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Update user error:', err);
        res.status(500).json({ error: 'Failed to update user' });
    }
});

// Search users
router.get('/search/:query', async (req, res) => {
    try {
        const query = req.params.query.toLowerCase();

        const result = await docClient.send(new ScanCommand({
            TableName: Tables.USERS,
            FilterExpression: 'contains(#username, :q) OR contains(#fullName, :q)',
            ExpressionAttributeNames: { '#username': 'username', '#fullName': 'fullName' },
            ExpressionAttributeValues: { ':q': query },
            Limit: 20
        }));

        const users = (result.Items || []).map(u => {
            delete u.password;
            return u;
        });

        res.json(users);
    } catch (err) {
        console.error('Search error:', err);
        res.status(500).json({ error: 'Search failed' });
    }
});

// Get recommended users (shuffled from all users)
router.get('/recommended/list', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 20;

        // Scan all users
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.USERS,
            Limit: 100 // Get up to 100 users to shuffle from
        }));

        const users = (result.Items || []).map(u => {
            delete u.password;
            return u;
        });

        // Shuffle users randomly
        for (let i = users.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [users[i], users[j]] = [users[j], users[i]];
        }

        // Return limited shuffled users
        res.json(users.slice(0, limit));
    } catch (err) {
        console.error('Get recommended users error:', err);
        res.status(500).json({ error: 'Failed to get recommended users' });
    }
});

// Update FCM token for push notifications
router.put('/fcm-token', verifyToken, async (req, res) => {
    try {
        const { fcmToken } = req.body;

        if (!fcmToken) {
            return res.status(400).json({ error: 'FCM token required' });
        }

        await docClient.send(new UpdateCommand({
            TableName: Tables.USERS,
            Key: { userId: req.userId },
            UpdateExpression: 'SET fcmToken = :token',
            ExpressionAttributeValues: { ':token': fcmToken }
        }));

        console.log(`FCM token updated for user ${req.userId}`);
        res.json({ success: true });
    } catch (err) {
        console.error('Update FCM token error:', err);
        res.status(500).json({ error: 'Failed to update FCM token' });
    }
});

module.exports = router;