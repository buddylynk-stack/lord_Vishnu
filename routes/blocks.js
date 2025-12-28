const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { PutCommand, DeleteCommand, ScanCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');

const router = express.Router();

// Get blocked users list - uses Scan since blockId is the partition key
router.get('/', verifyToken, async (req, res) => {
    try {
        console.log(`[Blocks API] Getting blocks for userId: ${req.userId}`);

        // Scan for all blocks where blockerId matches current user
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.BLOCKS || 'Buddylynk_Blocks',
            FilterExpression: 'blockerId = :uid',
            ExpressionAttributeValues: { ':uid': req.userId }
        }));

        console.log(`[Blocks API] Scan returned ${result.Items?.length || 0} items`);
        console.log(`[Blocks API] Raw items: ${JSON.stringify(result.Items?.slice(0, 3) || [])}`);

        const blockedIds = (result.Items || []).map(item => item.blockedId);
        console.log(`[Blocks API] Get blocks for ${req.userId}: found ${blockedIds.length} blocked users: ${blockedIds.slice(0, 5).join(', ')}...`);
        res.json(blockedIds);
    } catch (err) {
        console.error('[Blocks API] Get blocked users error:', err);
        res.status(500).json({ error: 'Failed to get blocked users' });
    }
});

// Block a user - uses blockId as partition key
router.post('/:userId', verifyToken, async (req, res) => {
    try {
        const targetUserId = req.params.userId;

        if (targetUserId === req.userId) {
            return res.status(400).json({ error: 'Cannot block yourself' });
        }

        // Create unique blockId for partition key
        const blockId = uuidv4();

        await docClient.send(new PutCommand({
            TableName: Tables.BLOCKS || 'Buddylynk_Blocks',
            Item: {
                blockId: blockId,  // Partition key
                blockerId: req.userId,
                blockedId: targetUserId,
                createdAt: new Date().toISOString()
            }
        }));

        console.log(`User ${req.userId} blocked ${targetUserId}`);
        res.json({ success: true, message: 'User blocked' });
    } catch (err) {
        console.error('Block user error:', err);
        res.status(500).json({ error: 'Failed to block user' });
    }
});

// Unblock a user - need to find the blockId first, then delete
router.delete('/:userId', verifyToken, async (req, res) => {
    try {
        const targetUserId = req.params.userId;

        // First find the block record
        const scanResult = await docClient.send(new ScanCommand({
            TableName: Tables.BLOCKS || 'Buddylynk_Blocks',
            FilterExpression: 'blockerId = :uid AND blockedId = :tid',
            ExpressionAttributeValues: {
                ':uid': req.userId,
                ':tid': targetUserId
            }
        }));

        if (scanResult.Items && scanResult.Items.length > 0) {
            const blockId = scanResult.Items[0].blockId;

            await docClient.send(new DeleteCommand({
                TableName: Tables.BLOCKS || 'Buddylynk_Blocks',
                Key: { blockId: blockId }
            }));

            console.log(`User ${req.userId} unblocked ${targetUserId}`);
        }

        res.json({ success: true, message: 'User unblocked' });
    } catch (err) {
        console.error('Unblock user error:', err);
        res.status(500).json({ error: 'Failed to unblock user' });
    }
});

module.exports = router;
