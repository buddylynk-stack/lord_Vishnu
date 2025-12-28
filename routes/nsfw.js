const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { PutCommand, GetCommand, QueryCommand, ScanCommand, DeleteCommand, UpdateCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');

const router = express.Router();

// Dedicated NSFW Table
const NSFW_TABLE = 'Buddylynk_NSFW';
const POSTS_TABLE = 'Buddylynk_Posts';

/*
 * TABLE STRUCTURE: Buddylynk_NSFW
 * ================================
 * postId      (String, PK)  - Post ID
 * userId      (String)      - User who created the post
 * username    (String)      - Username of post creator
 * isAdult     (Boolean)     - true = Adult/18+, false = Safe
 * reason      (String)      - Why it was flagged
 * flaggedBy   (String)      - Admin who flagged it
 * flaggedAt   (String)      - When it was flagged (ISO date)
 * reviewedAt  (String)      - When it was last reviewed
 */

/**
 * GET /nsfw/posts
 * Get all posts where isAdult = true
 */
router.get('/posts', async (req, res) => {
    try {
        const command = new ScanCommand({
            TableName: NSFW_TABLE,
            FilterExpression: 'isAdult = :adult',
            ExpressionAttributeValues: {
                ':adult': true
            }
        });

        const result = await docClient.send(command);

        res.json({
            success: true,
            nsfwPosts: result.Items || [],
            count: result.Items?.length || 0
        });
    } catch (error) {
        console.error('Error fetching NSFW posts:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to fetch NSFW posts',
            nsfwPosts: []
        });
    }
});

/**
 * GET /nsfw/all
 * Get ALL reviewed posts (both adult and safe)
 */
router.get('/all', async (req, res) => {
    try {
        const command = new ScanCommand({
            TableName: NSFW_TABLE
        });

        const result = await docClient.send(command);

        res.json({
            success: true,
            posts: result.Items || [],
            count: result.Items?.length || 0
        });
    } catch (error) {
        console.error('Error fetching all reviewed posts:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to fetch posts'
        });
    }
});

/**
 * GET /nsfw/check/:postId
 * Check if a specific post is flagged
 */
router.get('/check/:postId', async (req, res) => {
    try {
        const { postId } = req.params;

        const command = new GetCommand({
            TableName: NSFW_TABLE,
            Key: { postId }
        });

        const result = await docClient.send(command);

        res.json({
            success: true,
            postId,
            exists: !!result.Item,
            isAdult: result.Item?.isAdult || false,
            data: result.Item || null
        });
    } catch (error) {
        console.error('Error checking post:', error);
        res.status(500).json({
            success: false,
            isAdult: false
        });
    }
});

/**
 * POST /nsfw/flag (Admin only)
 * Flag/review a post - adds all details to NSFW table
 * 
 * Body: {
 *   postId: "xxx",
 *   userId: "xxx",
 *   username: "xxx",
 *   isAdult: true/false,  (true = Adult, false = Safe)
 *   reason: "xxx"
 * }
 */
router.post('/flag', verifyToken, async (req, res) => {
    try {
        const { postId, userId, username, isAdult, reason } = req.body;
        const adminUserId = req.user.userId;

        if (!postId) {
            return res.status(400).json({
                success: false,
                error: 'postId is required'
            });
        }

        // isAdult: true = 18+ adult content, false = safe content
        const adultFlag = isAdult === true || isAdult === 'true';
        const now = new Date().toISOString();

        // Add to NSFW table with ALL details
        const command = new PutCommand({
            TableName: NSFW_TABLE,
            Item: {
                postId: postId,                              // Post ID (Primary Key)
                userId: userId || 'unknown',                 // User who created post
                username: username || 'unknown',             // Username
                isAdult: adultFlag,                          // true = Adult/18+, false = Safe
                reason: reason || (adultFlag ? '18+ adult content' : 'Reviewed as safe'),
                flaggedBy: adminUserId,                      // Admin who flagged
                flaggedAt: now,                              // When flagged
                reviewedAt: now                              // Last review time
            }
        });

        await docClient.send(command);

        // Note: No sync to Posts table needed - feed endpoint queries NSFW table directly

        console.log(`Post ${postId} by ${username} flagged as ${adultFlag ? 'ADULT' : 'SAFE'} by admin ${adminUserId}`);

        res.json({
            success: true,
            message: adultFlag ? 'Post flagged as ADULT (18+)' : 'Post marked as SAFE',
            data: {
                postId,
                userId,
                username,
                isAdult: adultFlag,
                reason,
                flaggedBy: adminUserId,
                flaggedAt: now
            }
        });
    } catch (error) {
        console.error('Error flagging post:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to flag post'
        });
    }
});

/**
 * DELETE /nsfw/remove/:postId (Admin only)
 * Remove a post from NSFW table
 */
router.delete('/remove/:postId', verifyToken, async (req, res) => {
    try {
        const { postId } = req.params;

        const command = new DeleteCommand({
            TableName: NSFW_TABLE,
            Key: { postId }
        });

        await docClient.send(command);

        res.json({
            success: true,
            message: 'Post removed from NSFW table',
            postId
        });
    } catch (error) {
        console.error('Error removing post:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to remove post'
        });
    }
});

/**
 * GET /nsfw/stats
 * Get statistics
 */
router.get('/stats', async (req, res) => {
    try {
        const adultCommand = new ScanCommand({
            TableName: NSFW_TABLE,
            FilterExpression: 'isAdult = :val',
            ExpressionAttributeValues: { ':val': true },
            Select: 'COUNT'
        });

        const safeCommand = new ScanCommand({
            TableName: NSFW_TABLE,
            FilterExpression: 'isAdult = :val',
            ExpressionAttributeValues: { ':val': false },
            Select: 'COUNT'
        });

        const [adultResult, safeResult] = await Promise.all([
            docClient.send(adultCommand),
            docClient.send(safeCommand)
        ]);

        res.json({
            success: true,
            adultPosts: adultResult.Count || 0,
            safePosts: safeResult.Count || 0,
            totalReviewed: (adultResult.Count || 0) + (safeResult.Count || 0)
        });
    } catch (error) {
        console.error('Error fetching stats:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to fetch stats'
        });
    }
});

module.exports = router;
