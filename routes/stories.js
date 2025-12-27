const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { PutCommand, ScanCommand, DeleteCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');

const router = express.Router();

// Get stories (last 24 hours)
router.get('/', verifyToken, async (req, res) => {
    try {
        const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();

        const result = await docClient.send(new ScanCommand({
            TableName: Tables.STORIES,
            FilterExpression: 'createdAt > :time',
            ExpressionAttributeValues: { ':time': oneDayAgo }
        }));

        const stories = (result.Items || []).sort((a, b) =>
            new Date(b.createdAt) - new Date(a.createdAt)
        );

        res.json(stories);
    } catch (err) {
        console.error('Get stories error:', err);
        res.status(500).json({ error: 'Failed to get stories' });
    }
});

// Create story
router.post('/', verifyToken, async (req, res) => {
    try {
        const { mediaUrl, mediaType, caption } = req.body;

        if (!mediaUrl) {
            return res.status(400).json({ error: 'Media URL required' });
        }

        const storyId = uuidv4();
        const now = new Date().toISOString();
        const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();

        const story = {
            storyId,
            userId: req.userId,
            mediaUrl,
            mediaType: mediaType || 'image',
            caption: caption || '',
            viewCount: 0,
            createdAt: now,
            expiresAt
        };

        await docClient.send(new PutCommand({
            TableName: Tables.STORIES,
            Item: story
        }));

        res.status(201).json(story);
    } catch (err) {
        console.error('Create story error:', err);
        res.status(500).json({ error: 'Failed to create story' });
    }
});

// Delete story
router.delete('/:storyId', verifyToken, async (req, res) => {
    try {
        await docClient.send(new DeleteCommand({
            TableName: Tables.STORIES,
            Key: { storyId: req.params.storyId }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Delete story error:', err);
        res.status(500).json({ error: 'Failed to delete story' });
    }
});

module.exports = router;
