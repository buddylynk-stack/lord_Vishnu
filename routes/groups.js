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

// ============ INVITE LINKS ============

// Get invite links for a group
router.get('/:groupId/invite-links', verifyToken, async (req, res) => {
    try {
        const { groupId } = req.params;

        const result = await docClient.send(new ScanCommand({
            TableName: Tables.INVITE_LINKS,
            FilterExpression: 'groupId = :gid AND isActive = :active',
            ExpressionAttributeValues: { ':gid': groupId, ':active': true }
        }));

        const links = (result.Items || []).map(link => ({
            linkId: link.linkId,
            code: link.code,
            url: `https://app.buddylynk.com/invite/${link.code}`,
            createdAt: link.createdAt,
            createdBy: link.createdBy,
            isActive: link.isActive
        }));

        res.json(links);
    } catch (err) {
        console.error('Get invite links error:', err);
        res.status(500).json({ error: 'Failed to get invite links' });
    }
});

// Create invite link
router.post('/:groupId/invite-links', verifyToken, async (req, res) => {
    try {
        const { groupId } = req.params;
        const linkId = uuidv4();
        const code = Math.random().toString(36).substring(2, 10);
        const now = new Date().toISOString();

        const inviteLink = {
            linkId,
            groupId,
            code,
            url: `https://app.buddylynk.com/invite/${code}`,
            createdAt: now,
            createdBy: req.userId,
            isActive: true
        };

        await docClient.send(new PutCommand({
            TableName: Tables.INVITE_LINKS,
            Item: inviteLink
        }));

        res.status(201).json(inviteLink);
    } catch (err) {
        console.error('Create invite link error:', err);
        res.status(500).json({ error: 'Failed to create invite link' });
    }
});

// Delete/revoke invite link
router.delete('/:groupId/invite-links/:linkId', verifyToken, async (req, res) => {
    try {
        const { groupId, linkId } = req.params;
        const createNew = req.query.createNew === 'true';

        // Mark old link as inactive
        await docClient.send(new UpdateCommand({
            TableName: Tables.INVITE_LINKS,
            Key: { linkId },
            UpdateExpression: 'SET isActive = :inactive',
            ExpressionAttributeValues: { ':inactive': false }
        }));

        let newLink = null;

        // Create new link if requested
        if (createNew) {
            const newLinkId = uuidv4();
            const code = Math.random().toString(36).substring(2, 10);
            const now = new Date().toISOString();

            newLink = {
                linkId: newLinkId,
                groupId,
                code,
                url: `https://app.buddylynk.com/invite/${code}`,
                createdAt: now,
                createdBy: req.userId,
                isActive: true
            };

            await docClient.send(new PutCommand({
                TableName: Tables.INVITE_LINKS,
                Item: newLink
            }));
        }

        res.json({ success: true, newLink });
    } catch (err) {
        console.error('Delete invite link error:', err);
        res.status(500).json({ error: 'Failed to delete invite link' });
    }
});

// Join group via invite link
router.post('/join/:code', verifyToken, async (req, res) => {
    try {
        const { code } = req.params;

        // Find the invite link
        const linkResult = await docClient.send(new ScanCommand({
            TableName: Tables.INVITE_LINKS,
            FilterExpression: 'code = :code AND isActive = :active',
            ExpressionAttributeValues: { ':code': code, ':active': true }
        }));

        if (!linkResult.Items || linkResult.Items.length === 0) {
            return res.status(404).json({ error: 'Invalid or expired invite link' });
        }

        const inviteLink = linkResult.Items[0];
        const groupId = inviteLink.groupId;

        // Get the group
        const groupResult = await docClient.send(new GetCommand({
            TableName: Tables.GROUPS,
            Key: { groupId }
        }));

        if (!groupResult.Item) {
            return res.status(404).json({ error: 'Group not found' });
        }

        const group = groupResult.Item;
        const memberIds = group.memberIds || [];

        // Check if already a member
        if (memberIds.includes(req.userId) || group.creatorId === req.userId) {
            return res.json({ success: true, message: 'Already a member', group });
        }

        // Add user to group
        await docClient.send(new UpdateCommand({
            TableName: Tables.GROUPS,
            Key: { groupId },
            UpdateExpression: 'SET memberIds = list_append(if_not_exists(memberIds, :empty), :uid), memberCount = if_not_exists(memberCount, :zero) + :one',
            ExpressionAttributeValues: {
                ':uid': [req.userId],
                ':empty': [],
                ':zero': 0,
                ':one': 1
            }
        }));

        res.json({ success: true, message: 'Joined successfully', groupId, groupName: group.name });
    } catch (err) {
        console.error('Join via invite error:', err);
        res.status(500).json({ error: 'Failed to join group' });
    }
});

module.exports = router;
