const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { PutCommand, GetCommand, ScanCommand, UpdateCommand, DeleteCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');
const { convertGroupMediaUrls, convertMessageMediaUrls } = require('../utils/cloudfront');

const router = express.Router();

// Get user's groups
router.get('/', verifyToken, async (req, res) => {
    try {
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.GROUPS,
            FilterExpression: 'creatorId = :uid OR contains(memberIds, :uid)',
            ExpressionAttributeValues: { ':uid': req.userId }
        }));

        const groups = (result.Items || []).map(convertGroupMediaUrls);
        res.json(groups);
    } catch (err) {
        console.error('Get groups error:', err);
        res.status(500).json({ error: 'Failed to get groups' });
    }
});

// Create group
router.post('/', verifyToken, async (req, res) => {
    try {
        const { name, description, imageUrl, isPublic } = req.body;

        if (!name) {
            return res.status(400).json({ error: 'Group name required' });
        }

        const groupId = uuidv4();
        const now = new Date().toISOString();

        const group = {
            groupId,
            name,
            description: description || null,
            imageUrl: imageUrl || null,
            creatorId: req.userId,
            memberIds: [req.userId],
            memberCount: 1,
            isPublic: isPublic || false,
            createdAt: now,
            updatedAt: now
        };

        await docClient.send(new PutCommand({
            TableName: Tables.GROUPS,
            Item: group
        }));

        res.status(201).json(group);
    } catch (err) {
        console.error('Create group error:', err);
        res.status(500).json({ error: 'Failed to create group' });
    }
});

// Get single group
router.get('/:groupId', async (req, res) => {
    try {
        const result = await docClient.send(new GetCommand({
            TableName: Tables.GROUPS,
            Key: { groupId: req.params.groupId }
        }));

        if (!result.Item) {
            return res.status(404).json({ error: 'Group not found' });
        }

        res.json(result.Item);
    } catch (err) {
        console.error('Get group error:', err);
        res.status(500).json({ error: 'Failed to get group' });
    }
});

// Add member to group
router.post('/:groupId/members', verifyToken, async (req, res) => {
    try {
        const { userId } = req.body;
        const { groupId } = req.params;

        await docClient.send(new UpdateCommand({
            TableName: Tables.GROUPS,
            Key: { groupId },
            UpdateExpression: 'SET memberIds = list_append(if_not_exists(memberIds, :empty), :member), memberCount = memberCount + :one',
            ExpressionAttributeValues: {
                ':member': [userId],
                ':empty': [],
                ':one': 1
            }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Add member error:', err);
        res.status(500).json({ error: 'Failed to add member' });
    }
});

// Get group messages with sender info
router.get('/:groupId/messages', verifyToken, async (req, res) => {
    try {
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.MESSAGES,
            FilterExpression: 'conversationId = :cid',
            ExpressionAttributeValues: { ':cid': `group_${req.params.groupId}` }
        }));

        const messages = (result.Items || []).sort((a, b) =>
            new Date(a.createdAt) - new Date(b.createdAt)
        );

        // Fetch sender info for each message
        const messagesWithSender = await Promise.all(messages.map(async (msg) => {
            try {
                const userResult = await docClient.send(new GetCommand({
                    TableName: Tables.USERS,
                    Key: { userId: msg.senderId }
                }));
                const sender = userResult.Item || {};
                return {
                    ...msg,
                    senderName: sender.username || 'User',
                    senderAvatar: sender.avatar || null
                };
            } catch (e) {
                return {
                    ...msg,
                    senderName: 'User',
                    senderAvatar: null
                };
            }
        }));

        res.json(messagesWithSender);
    } catch (err) {
        console.error('Get messages error:', err);
        res.status(500).json({ error: 'Failed to get messages' });
    }
});

// Send group message
router.post('/:groupId/messages', verifyToken, async (req, res) => {
    try {
        const { content, mediaUrl, mediaType } = req.body;
        const { groupId } = req.params;

        const messageId = uuidv4();
        const now = new Date().toISOString();

        const message = {
            messageId,
            conversationId: `group_${groupId}`,
            senderId: req.userId,
            receiverId: groupId,
            content: content || '',
            mediaUrl: mediaUrl || null,
            mediaType: mediaType || null,
            isRead: false,
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

module.exports = router;
