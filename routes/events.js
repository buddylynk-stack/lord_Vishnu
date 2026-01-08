const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { PutCommand, GetCommand, ScanCommand, UpdateCommand, DeleteCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');

const router = express.Router();

// ============ EVENTS CRUD ============

// Get user's events (events they created or joined)
router.get('/', verifyToken, async (req, res) => {
    try {
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.EVENTS,
            FilterExpression: 'organizerId = :uid OR contains(memberIds, :uid)',
            ExpressionAttributeValues: { ':uid': req.userId }
        }));

        res.json(result.Items || []);
    } catch (err) {
        console.error('Get events error:', err);
        res.status(500).json({ error: 'Failed to get events' });
    }
});

// Create a new event
router.post('/', verifyToken, async (req, res) => {
    try {
        const { title, description, date, time, location, category, isOnline, imageUrl } = req.body;

        if (!title || !date || !time) {
            return res.status(400).json({ error: 'Title, date, and time are required' });
        }

        const eventId = 'evt_' + uuidv4();
        const now = new Date().toISOString();

        const event = {
            eventId,
            title,
            description: description || '',
            date,
            time,
            location: location || '',
            category: category || 'General',
            isOnline: isOnline || location?.toLowerCase() === 'online',
            imageUrl: imageUrl || null,
            organizerId: req.userId,
            memberIds: [req.userId], // Creator is automatically a member
            adminIds: [], // Creator is owner, not just admin
            attendeesCount: 1,
            chatEnabled: true,
            createdAt: now,
            updatedAt: now
        };

        await docClient.send(new PutCommand({
            TableName: Tables.EVENTS,
            Item: event
        }));

        console.log(`Event created: ${eventId} by user ${req.userId}`);
        res.status(201).json(event);
    } catch (err) {
        console.error('Create event error:', err);
        res.status(500).json({ error: 'Failed to create event' });
    }
});

// Get single event
router.get('/:eventId', verifyToken, async (req, res) => {
    try {
        const result = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId: req.params.eventId }
        }));

        if (!result.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        res.json(result.Item);
    } catch (err) {
        console.error('Get event error:', err);
        res.status(500).json({ error: 'Failed to get event' });
    }
});

// Update event details
router.put('/:eventId', verifyToken, async (req, res) => {
    try {
        const { eventId } = req.params;
        const { title, description, date, time, location, category, chatEnabled } = req.body;

        // Get existing event to check ownership
        const existing = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!existing.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        // Only owner or admin can update
        const isOwner = existing.Item.organizerId === req.userId;
        const isAdmin = (existing.Item.adminIds || []).includes(req.userId);

        if (!isOwner && !isAdmin) {
            return res.status(403).json({ error: 'Not authorized to update this event' });
        }

        // Build update expression dynamically
        let updateParts = [];
        let expressionValues = {};
        let expressionNames = {};

        if (title !== undefined) {
            updateParts.push('#title = :title');
            expressionValues[':title'] = title;
            expressionNames['#title'] = 'title';
        }
        if (description !== undefined) {
            updateParts.push('#desc = :desc');
            expressionValues[':desc'] = description;
            expressionNames['#desc'] = 'description';
        }
        if (date !== undefined) {
            updateParts.push('#date = :date');
            expressionValues[':date'] = date;
            expressionNames['#date'] = 'date';
        }
        if (time !== undefined) {
            updateParts.push('#time = :time');
            expressionValues[':time'] = time;
            expressionNames['#time'] = 'time';
        }
        if (location !== undefined) {
            updateParts.push('#loc = :loc');
            expressionValues[':loc'] = location;
            expressionNames['#loc'] = 'location';
        }
        if (category !== undefined) {
            updateParts.push('category = :cat');
            expressionValues[':cat'] = category;
        }
        if (chatEnabled !== undefined) {
            updateParts.push('chatEnabled = :chatEnabled');
            expressionValues[':chatEnabled'] = chatEnabled;
        }

        if (updateParts.length === 0) {
            return res.status(400).json({ error: 'No fields to update' });
        }

        updateParts.push('updatedAt = :updatedAt');
        expressionValues[':updatedAt'] = new Date().toISOString();

        await docClient.send(new UpdateCommand({
            TableName: Tables.EVENTS,
            Key: { eventId },
            UpdateExpression: 'SET ' + updateParts.join(', '),
            ExpressionAttributeValues: expressionValues,
            ExpressionAttributeNames: Object.keys(expressionNames).length > 0 ? expressionNames : undefined
        }));

        // Fetch updated event
        const updated = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        res.json(updated.Item);
    } catch (err) {
        console.error('Update event error:', err);
        res.status(500).json({ error: 'Failed to update event' });
    }
});

// Delete event
router.delete('/:eventId', verifyToken, async (req, res) => {
    try {
        const { eventId } = req.params;

        // Get existing event to check ownership
        const existing = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!existing.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        // Only owner can delete
        if (existing.Item.organizerId !== req.userId) {
            return res.status(403).json({ error: 'Only the event owner can delete this event' });
        }

        await docClient.send(new DeleteCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Delete event error:', err);
        res.status(500).json({ error: 'Failed to delete event' });
    }
});

// ============ MEMBER MANAGEMENT ============

// Get event members with user details
router.get('/:eventId/members', verifyToken, async (req, res) => {
    try {
        const { eventId } = req.params;

        const eventResult = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!eventResult.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        const event = eventResult.Item;
        const memberIds = event.memberIds || [];
        const adminIds = event.adminIds || [];

        // Fetch user details for each member
        const members = await Promise.all(memberIds.map(async (userId) => {
            try {
                const userResult = await docClient.send(new GetCommand({
                    TableName: Tables.USERS,
                    Key: { userId }
                }));
                const user = userResult.Item || {};
                return {
                    userId,
                    username: user.username || 'User',
                    avatar: user.avatar || null,
                    fullName: user.fullName || '',
                    role: userId === event.organizerId ? 'owner' :
                        adminIds.includes(userId) ? 'admin' : 'member'
                };
            } catch (e) {
                return {
                    userId,
                    username: 'User',
                    avatar: null,
                    fullName: '',
                    role: 'member'
                };
            }
        }));

        // Also include owner if not in memberIds
        if (!memberIds.includes(event.organizerId)) {
            try {
                const ownerResult = await docClient.send(new GetCommand({
                    TableName: Tables.USERS,
                    Key: { userId: event.organizerId }
                }));
                const owner = ownerResult.Item || {};
                members.unshift({
                    userId: event.organizerId,
                    username: owner.username || 'Owner',
                    avatar: owner.avatar || null,
                    fullName: owner.fullName || '',
                    role: 'owner'
                });
            } catch (e) {
                members.unshift({
                    userId: event.organizerId,
                    username: 'Owner',
                    avatar: null,
                    fullName: '',
                    role: 'owner'
                });
            }
        }

        res.json(members);
    } catch (err) {
        console.error('Get event members error:', err);
        res.status(500).json({ error: 'Failed to get event members' });
    }
});

// Add member to event
router.post('/:eventId/members', verifyToken, async (req, res) => {
    try {
        const { userId } = req.body;
        const { eventId } = req.params;

        await docClient.send(new UpdateCommand({
            TableName: Tables.EVENTS,
            Key: { eventId },
            UpdateExpression: 'SET memberIds = list_append(if_not_exists(memberIds, :empty), :member), attendeesCount = if_not_exists(attendeesCount, :zero) + :one',
            ExpressionAttributeValues: {
                ':member': [userId],
                ':empty': [],
                ':zero': 0,
                ':one': 1
            }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Add event member error:', err);
        res.status(500).json({ error: 'Failed to add member' });
    }
});

// Remove member from event
router.delete('/:eventId/members/:userId', verifyToken, async (req, res) => {
    try {
        const { eventId, userId } = req.params;

        // Get event to check permissions
        const eventResult = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!eventResult.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        const event = eventResult.Item;
        const isOwner = event.organizerId === req.userId;
        const isAdmin = (event.adminIds || []).includes(req.userId);
        const isSelf = userId === req.userId;

        if (!isOwner && !isAdmin && !isSelf) {
            return res.status(403).json({ error: 'Not authorized to remove this member' });
        }

        // Cannot remove owner
        if (userId === event.organizerId) {
            return res.status(400).json({ error: 'Cannot remove event owner' });
        }

        // Remove from memberIds and adminIds
        const newMemberIds = (event.memberIds || []).filter(id => id !== userId);
        const newAdminIds = (event.adminIds || []).filter(id => id !== userId);

        await docClient.send(new UpdateCommand({
            TableName: Tables.EVENTS,
            Key: { eventId },
            UpdateExpression: 'SET memberIds = :members, adminIds = :admins, attendeesCount = :count',
            ExpressionAttributeValues: {
                ':members': newMemberIds,
                ':admins': newAdminIds,
                ':count': Math.max(0, newMemberIds.length)
            }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Remove event member error:', err);
        res.status(500).json({ error: 'Failed to remove member' });
    }
});

// Promote member to admin
router.post('/:eventId/members/:userId/promote', verifyToken, async (req, res) => {
    try {
        const { eventId, userId } = req.params;

        // Get event to check permissions
        const eventResult = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!eventResult.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        const event = eventResult.Item;
        const isOwner = event.organizerId === req.userId;
        const isAdmin = (event.adminIds || []).includes(req.userId);

        if (!isOwner && !isAdmin) {
            return res.status(403).json({ error: 'Only owner or admin can promote members' });
        }

        // Check if already admin
        if ((event.adminIds || []).includes(userId)) {
            return res.json({ success: true, message: 'Already an admin' });
        }

        await docClient.send(new UpdateCommand({
            TableName: Tables.EVENTS,
            Key: { eventId },
            UpdateExpression: 'SET adminIds = list_append(if_not_exists(adminIds, :empty), :admin)',
            ExpressionAttributeValues: {
                ':admin': [userId],
                ':empty': []
            }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Promote member error:', err);
        res.status(500).json({ error: 'Failed to promote member' });
    }
});

// Demote admin to member
router.post('/:eventId/members/:userId/demote', verifyToken, async (req, res) => {
    try {
        const { eventId, userId } = req.params;

        // Get event to check permissions
        const eventResult = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!eventResult.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        const event = eventResult.Item;

        // Only owner can demote
        if (event.organizerId !== req.userId) {
            return res.status(403).json({ error: 'Only owner can demote admins' });
        }

        // Remove from adminIds
        const newAdminIds = (event.adminIds || []).filter(id => id !== userId);

        await docClient.send(new UpdateCommand({
            TableName: Tables.EVENTS,
            Key: { eventId },
            UpdateExpression: 'SET adminIds = :admins',
            ExpressionAttributeValues: {
                ':admins': newAdminIds
            }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Demote member error:', err);
        res.status(500).json({ error: 'Failed to demote member' });
    }
});

// ============ INVITE LINKS ============

// Get invite links for an event
router.get('/:eventId/invite-links', verifyToken, async (req, res) => {
    try {
        const { eventId } = req.params;

        const result = await docClient.send(new ScanCommand({
            TableName: Tables.INVITE_LINKS,
            FilterExpression: 'eventId = :eid AND isActive = :active',
            ExpressionAttributeValues: { ':eid': eventId, ':active': true }
        }));

        const links = (result.Items || []).map(link => ({
            linkId: link.linkId,
            code: link.code,
            url: `https://app.buddylynk.com/event-invite/${link.code}`,
            createdAt: link.createdAt,
            createdBy: link.createdBy,
            isActive: link.isActive
        }));

        res.json(links);
    } catch (err) {
        console.error('Get event invite links error:', err);
        res.status(500).json({ error: 'Failed to get invite links' });
    }
});

// Create invite link for event
router.post('/:eventId/invite-links', verifyToken, async (req, res) => {
    try {
        const { eventId } = req.params;

        // Get event to check permissions
        const eventResult = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!eventResult.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        const event = eventResult.Item;
        const isOwner = event.organizerId === req.userId;
        const isAdmin = (event.adminIds || []).includes(req.userId);

        if (!isOwner && !isAdmin) {
            return res.status(403).json({ error: 'Only owner or admin can create invite links' });
        }

        const linkId = uuidv4();
        const code = 'evt_' + Math.random().toString(36).substring(2, 10);
        const now = new Date().toISOString();

        const inviteLink = {
            linkId,
            eventId,
            code,
            url: `https://app.buddylynk.com/event-invite/${code}`,
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
        console.error('Create event invite link error:', err);
        res.status(500).json({ error: 'Failed to create invite link' });
    }
});

// Delete/revoke invite link
router.delete('/:eventId/invite-links/:linkId', verifyToken, async (req, res) => {
    try {
        const { eventId, linkId } = req.params;

        // Mark link as inactive
        await docClient.send(new UpdateCommand({
            TableName: Tables.INVITE_LINKS,
            Key: { linkId },
            UpdateExpression: 'SET isActive = :inactive',
            ExpressionAttributeValues: { ':inactive': false }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Delete event invite link error:', err);
        res.status(500).json({ error: 'Failed to delete invite link' });
    }
});

// Join event via invite link
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
        const eventId = inviteLink.eventId;

        if (!eventId) {
            return res.status(400).json({ error: 'This is not an event invite link' });
        }

        // Get the event
        const eventResult = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!eventResult.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        const event = eventResult.Item;
        const memberIds = event.memberIds || [];

        // Check if already a member
        if (memberIds.includes(req.userId) || event.organizerId === req.userId) {
            return res.json({ success: true, message: 'Already a member', event });
        }

        // Add user to event
        await docClient.send(new UpdateCommand({
            TableName: Tables.EVENTS,
            Key: { eventId },
            UpdateExpression: 'SET memberIds = list_append(if_not_exists(memberIds, :empty), :uid), attendeesCount = if_not_exists(attendeesCount, :zero) + :one',
            ExpressionAttributeValues: {
                ':uid': [req.userId],
                ':empty': [],
                ':zero': 0,
                ':one': 1
            }
        }));

        res.json({ success: true, message: 'Joined successfully', eventId, eventTitle: event.title });
    } catch (err) {
        console.error('Join event via invite error:', err);
        res.status(500).json({ error: 'Failed to join event' });
    }
});

// ============ CHAT MESSAGES ============

// Get event messages
router.get('/:eventId/messages', verifyToken, async (req, res) => {
    try {
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.MESSAGES,
            FilterExpression: 'conversationId = :cid',
            ExpressionAttributeValues: { ':cid': `event_${req.params.eventId}` }
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
        console.error('Get event messages error:', err);
        res.status(500).json({ error: 'Failed to get messages' });
    }
});

// Send event message (with permission check)
router.post('/:eventId/messages', verifyToken, async (req, res) => {
    try {
        const { content, mediaUrl, mediaType } = req.body;
        const { eventId } = req.params;

        // Get event to check chat permissions
        const eventResult = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!eventResult.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        const event = eventResult.Item;
        const chatEnabled = event.chatEnabled !== false; // Default to true
        const isOwner = event.organizerId === req.userId;
        const isAdmin = (event.adminIds || []).includes(req.userId);
        const isMember = (event.memberIds || []).includes(req.userId);

        // Check if user can send messages
        if (!chatEnabled && !isOwner && !isAdmin) {
            return res.status(403).json({ error: 'Chat is disabled. Only admins can send messages.' });
        }

        // Check if user is part of the event
        if (!isOwner && !isMember) {
            return res.status(403).json({ error: 'You must be a member of this event to send messages' });
        }

        const messageId = uuidv4();
        const now = new Date().toISOString();

        const message = {
            messageId,
            conversationId: `event_${eventId}`,
            senderId: req.userId,
            receiverId: eventId,
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

        // Fetch sender info
        const userResult = await docClient.send(new GetCommand({
            TableName: Tables.USERS,
            Key: { userId: req.userId }
        }));
        const sender = userResult.Item || {};

        res.status(201).json({
            ...message,
            senderName: sender.username || 'User',
            senderAvatar: sender.avatar || null
        });
    } catch (err) {
        console.error('Send event message error:', err);
        res.status(500).json({ error: 'Failed to send message' });
    }
});

// ============ JOIN/LEAVE EVENT (RSVP) ============

// Join an event directly (no invite needed)
router.post('/:eventId/join', verifyToken, async (req, res) => {
    try {
        const { eventId } = req.params;

        // Get event
        const eventResult = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!eventResult.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        const event = eventResult.Item;
        const memberIds = event.memberIds || [];

        // Check if already a member
        if (memberIds.includes(req.userId) || event.organizerId === req.userId) {
            return res.json({ success: true, message: 'Already a member', event });
        }

        // Add user to event
        await docClient.send(new UpdateCommand({
            TableName: Tables.EVENTS,
            Key: { eventId },
            UpdateExpression: 'SET memberIds = list_append(if_not_exists(memberIds, :empty), :uid), attendeesCount = if_not_exists(attendeesCount, :zero) + :one',
            ExpressionAttributeValues: {
                ':uid': [req.userId],
                ':empty': [],
                ':zero': 0,
                ':one': 1
            }
        }));

        // Fetch updated event
        const updated = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        console.log(`User ${req.userId} joined event ${eventId}`);
        res.json({ success: true, message: 'Joined successfully', event: updated.Item });
    } catch (err) {
        console.error('Join event error:', err);
        res.status(500).json({ error: 'Failed to join event' });
    }
});

// Leave an event
router.post('/:eventId/leave', verifyToken, async (req, res) => {
    try {
        const { eventId } = req.params;

        // Get event
        const eventResult = await docClient.send(new GetCommand({
            TableName: Tables.EVENTS,
            Key: { eventId }
        }));

        if (!eventResult.Item) {
            return res.status(404).json({ error: 'Event not found' });
        }

        const event = eventResult.Item;

        // Cannot leave if you're the owner
        if (event.organizerId === req.userId) {
            return res.status(400).json({ error: 'Event owner cannot leave. Transfer ownership or delete the event.' });
        }

        // Remove from memberIds and adminIds
        const newMemberIds = (event.memberIds || []).filter(id => id !== req.userId);
        const newAdminIds = (event.adminIds || []).filter(id => id !== req.userId);

        await docClient.send(new UpdateCommand({
            TableName: Tables.EVENTS,
            Key: { eventId },
            UpdateExpression: 'SET memberIds = :members, adminIds = :admins, attendeesCount = :count',
            ExpressionAttributeValues: {
                ':members': newMemberIds,
                ':admins': newAdminIds,
                ':count': Math.max(0, newMemberIds.length)
            }
        }));

        console.log(`User ${req.userId} left event ${eventId}`);
        res.json({ success: true });
    } catch (err) {
        console.error('Leave event error:', err);
        res.status(500).json({ error: 'Failed to leave event' });
    }
});

// ============ PUBLIC EVENTS DISCOVERY ============

// Get all public events (for discovery)
router.get('/public', verifyToken, async (req, res) => {
    try {
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.EVENTS,
            FilterExpression: 'isPublic = :pub OR attribute_not_exists(isPublic)',
            ExpressionAttributeValues: { ':pub': true }
        }));

        // Sort by date (newest first)
        const events = (result.Items || []).sort((a, b) =>
            new Date(b.createdAt) - new Date(a.createdAt)
        );

        // Add user membership status
        const eventsWithStatus = events.map(event => ({
            ...event,
            isJoined: (event.memberIds || []).includes(req.userId) || event.organizerId === req.userId
        }));

        res.json(eventsWithStatus);
    } catch (err) {
        console.error('Get public events error:', err);
        res.status(500).json({ error: 'Failed to get public events' });
    }
});

// Get all events (admin/browse)
router.get('/all', verifyToken, async (req, res) => {
    try {
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.EVENTS
        }));

        // Sort by date (newest first)
        const events = (result.Items || []).sort((a, b) =>
            new Date(b.createdAt) - new Date(a.createdAt)
        );

        // Add user membership status
        const eventsWithStatus = events.map(event => ({
            ...event,
            isJoined: (event.memberIds || []).includes(req.userId) || event.organizerId === req.userId,
            isOwner: event.organizerId === req.userId
        }));

        res.json(eventsWithStatus);
    } catch (err) {
        console.error('Get all events error:', err);
        res.status(500).json({ error: 'Failed to get all events' });
    }
});

module.exports = router;
