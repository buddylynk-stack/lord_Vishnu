const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { PutCommand, ScanCommand, UpdateCommand, GetCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');
const { convertMessageMediaUrls } = require('../utils/cloudfront');

const router = express.Router();

// Get conversations for current user
router.get('/conversations', verifyToken, async (req, res) => {
    try {
        // Get all messages where user is sender or receiver
        const sentResult = await docClient.send(new ScanCommand({
            TableName: Tables.MESSAGES,
            FilterExpression: 'senderId = :uid',
            ExpressionAttributeValues: { ':uid': req.userId },
            ProjectionExpression: 'receiverId'
        }));

        const receivedResult = await docClient.send(new ScanCommand({
            TableName: Tables.MESSAGES,
            FilterExpression: 'receiverId = :uid',
            ExpressionAttributeValues: { ':uid': req.userId },
            ProjectionExpression: 'senderId'
        }));

        // Get unique conversation partners
        const sentTo = (sentResult.Items || []).map(i => i.receiverId);
        const receivedFrom = (receivedResult.Items || []).map(i => i.senderId);
        const partners = [...new Set([...sentTo, ...receivedFrom])].filter(id => id !== req.userId);

        res.json(partners);
    } catch (err) {
        console.error('Get conversations error:', err);
        res.status(500).json({ error: 'Failed to get conversations' });
    }
});

// Get messages with a user
router.get('/:userId', verifyToken, async (req, res) => {
    try {
        const partnerId = req.params.userId;
        const currentUserId = req.userId;

        // Get messages where current user sent to partner OR partner sent to current user
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.MESSAGES,
            FilterExpression: '(senderId = :me AND receiverId = :them) OR (senderId = :them AND receiverId = :me)',
            ExpressionAttributeValues: {
                ':me': currentUserId,
                ':them': partnerId
            }
        }));

        let messages = (result.Items || []).sort((a, b) =>
            new Date(a.createdAt) - new Date(b.createdAt)
        );

        // Convert media URLs to CloudFront
        messages = messages.map(convertMessageMediaUrls);

        res.json(messages);
    } catch (err) {
        console.error('Get messages error:', err);
        res.status(500).json({ error: 'Failed to get messages' });
    }
});

// Send message
router.post('/:userId', verifyToken, async (req, res) => {
    try {
        const { content, mediaUrl, mediaType } = req.body;
        const receiverId = req.params.userId;

        const messageId = uuidv4();
        const now = new Date().toISOString();

        // Create conversation ID (sorted user IDs for consistency)
        const ids = [req.userId, receiverId].sort();
        const conversationId = `${ids[0]}_${ids[1]}`;

        const message = {
            messageId,
            conversationId,
            senderId: req.userId,
            receiverId,
            content: content || '',
            mediaUrl: mediaUrl || null,
            mediaType: mediaType || null,
            isRead: false,
            status: 'sent',
            createdAt: now
        };

        await docClient.send(new PutCommand({
            TableName: Tables.MESSAGES,
            Item: message
        }));

        res.status(201).json(message);
    } catch (err) {
        console.error('Send message error:', err);
        res.status(500).json({ error: 'Failed to send message' });
    }
});

// Mark message as read
router.put('/:messageId/read', verifyToken, async (req, res) => {
    try {
        await docClient.send(new UpdateCommand({
            TableName: Tables.MESSAGES,
            Key: { messageId: req.params.messageId },
            UpdateExpression: 'SET isRead = :true, #status = :read',
            ExpressionAttributeNames: { '#status': 'status' },
            ExpressionAttributeValues: { ':true': true, ':read': 'read' }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Mark read error:', err);
        res.status(500).json({ error: 'Failed to mark as read' });
    }
});

module.exports = router;
