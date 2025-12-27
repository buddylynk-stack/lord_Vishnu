const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { PutCommand, GetCommand, QueryCommand, UpdateCommand, ScanCommand, DeleteCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');

const router = express.Router();

// CloudFront CDN domain for faster media delivery (S3-backed distribution)
const CLOUDFRONT_DOMAIN = 'd2cwas7x7omdpp.cloudfront.net';
const S3_PATTERNS = [
    'buddylynk-media-bucket-2024.s3.amazonaws.com',
    'buddylynk-media-bucket-2024.s3.us-east-1.amazonaws.com',
    'buddylynk-mobile-server.s3.amazonaws.com',
    'buddylynk-mobile-server.s3.us-east-1.amazonaws.com'
];

// Convert S3 URL to CloudFront URL
function toCloudFrontUrl(url) {
    if (!url || typeof url !== 'string') return url;
    for (const pattern of S3_PATTERNS) {
        if (url.includes(pattern)) {
            return url.replace(pattern, CLOUDFRONT_DOMAIN);
        }
    }
    return url;
}

// Convert all media URLs in a post to CloudFront
function convertPostMediaUrls(post) {
    if (!post) return post;

    // Convert single URL fields
    if (post.mediaUrl) post.mediaUrl = toCloudFrontUrl(post.mediaUrl);
    if (post.userAvatar) post.userAvatar = toCloudFrontUrl(post.userAvatar);

    // Convert media array
    if (post.media && Array.isArray(post.media)) {
        post.media = post.media.map(m => ({
            ...m,
            url: toCloudFrontUrl(m.url)
        }));
    }

    // Convert mediaUrls array
    if (post.mediaUrls && Array.isArray(post.mediaUrls)) {
        post.mediaUrls = post.mediaUrls.map(url => toCloudFrontUrl(url));
    }

    return post;
}

// Simple in-memory cache for feed
let feedCache = null;
let feedCacheTime = 0;
const CACHE_TTL = 30000; // 30 seconds

// Get feed posts (PUBLIC - no auth required for browsing)
router.get('/feed', async (req, res) => {
    try {
        const now = Date.now();

        // Return cached response if valid
        if (feedCache && (now - feedCacheTime) < CACHE_TTL) {
            console.log('Serving cached feed');
            return res.json(feedCache);
        }

        // Get posts (limited for performance)
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.POSTS,
            Limit: 50
        }));

        let posts = (result.Items || []).sort((a, b) =>
            new Date(b.createdAt) - new Date(a.createdAt)
        );

        // Convert all media URLs to CloudFront
        posts = posts.map(convertPostMediaUrls);

        // Cache the result
        feedCache = posts;
        feedCacheTime = now;

        res.json(posts);
    } catch (err) {
        console.error('Feed error:', err);
        res.status(500).json({ error: 'Failed to load feed' });
    }
});

// Create post
router.post('/', verifyToken, async (req, res) => {
    try {
        const { content, mediaUrls, mediaType } = req.body;

        const postId = uuidv4();
        const now = new Date().toISOString();

        const post = {
            postId,
            userId: req.userId,
            content: content || '',
            mediaUrls: mediaUrls || [],
            mediaType: mediaType || 'text',
            likeCount: 0,
            commentCount: 0,
            viewCount: 0,
            createdAt: now,
            updatedAt: now
        };

        await docClient.send(new PutCommand({
            TableName: Tables.POSTS,
            Item: post
        }));

        res.status(201).json(post);
    } catch (err) {
        console.error('Create post error:', err);
        res.status(500).json({ error: 'Failed to create post' });
    }
});

// Get single post
router.get('/:postId', async (req, res) => {
    try {
        const result = await docClient.send(new GetCommand({
            TableName: Tables.POSTS,
            Key: { postId: req.params.postId }
        }));

        if (!result.Item) {
            return res.status(404).json({ error: 'Post not found' });
        }

        res.json(result.Item);
    } catch (err) {
        console.error('Get post error:', err);
        res.status(500).json({ error: 'Failed to get post' });
    }
});

// Like post
router.post('/:postId/like', verifyToken, async (req, res) => {
    try {
        const { postId } = req.params;

        // Increment like count
        await docClient.send(new UpdateCommand({
            TableName: Tables.POSTS,
            Key: { postId },
            UpdateExpression: 'SET likeCount = if_not_exists(likeCount, :zero) + :inc',
            ExpressionAttributeValues: { ':inc': 1, ':zero': 0 }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Like error:', err);
        res.status(500).json({ error: 'Failed to like post' });
    }
});

// Unlike post
router.delete('/:postId/like', verifyToken, async (req, res) => {
    try {
        const { postId } = req.params;

        await docClient.send(new UpdateCommand({
            TableName: Tables.POSTS,
            Key: { postId },
            UpdateExpression: 'SET likeCount = likeCount - :dec',
            ExpressionAttributeValues: { ':dec': 1 },
            ConditionExpression: 'likeCount > :zero',
            ExpressionAttributeValues: { ':dec': 1, ':zero': 0 }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Unlike error:', err);
        res.status(500).json({ error: 'Failed to unlike post' });
    }
});

// Get user posts
router.get('/user/:userId', async (req, res) => {
    try {
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.POSTS,
            FilterExpression: 'userId = :uid',
            ExpressionAttributeValues: { ':uid': req.params.userId }
        }));

        const posts = (result.Items || []).sort((a, b) =>
            new Date(b.createdAt) - new Date(a.createdAt)
        );

        res.json(posts);
    } catch (err) {
        console.error('Get user posts error:', err);
        res.status(500).json({ error: 'Failed to get posts' });
    }
});

// Delete post
router.delete('/:postId', verifyToken, async (req, res) => {
    try {
        // Verify post belongs to user
        const getResult = await docClient.send(new GetCommand({
            TableName: Tables.POSTS,
            Key: { postId: req.params.postId }
        }));

        if (!getResult.Item) {
            return res.status(404).json({ error: 'Post not found' });
        }

        if (getResult.Item.userId !== req.userId) {
            return res.status(403).json({ error: 'Not authorized' });
        }

        await docClient.send(new DeleteCommand({
            TableName: Tables.POSTS,
            Key: { postId: req.params.postId }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Delete post error:', err);
        res.status(500).json({ error: 'Failed to delete post' });
    }
});

// Save post - uses userId (PK) + postId (SK)
router.post('/:postId/save', verifyToken, async (req, res) => {
    try {
        await docClient.send(new PutCommand({
            TableName: Tables.SAVES,
            Item: {
                userId: req.userId,
                postId: req.params.postId,
                createdAt: new Date().toISOString()
            }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Save post error:', err);
        res.status(500).json({ error: 'Failed to save post' });
    }
});

// Unsave post - uses userId (PK) + postId (SK)
router.delete('/:postId/save', verifyToken, async (req, res) => {
    try {
        await docClient.send(new DeleteCommand({
            TableName: Tables.SAVES,
            Key: {
                userId: req.userId,
                postId: req.params.postId
            }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Unsave post error:', err);
        res.status(500).json({ error: 'Failed to unsave post' });
    }
});

// Get saved posts - use QueryCommand since userId is partition key
router.get('/saved/list', verifyToken, async (req, res) => {
    try {
        const result = await docClient.send(new QueryCommand({
            TableName: Tables.SAVES,
            KeyConditionExpression: 'userId = :uid',
            ExpressionAttributeValues: { ':uid': req.userId }
        }));

        const savedPostIds = (result.Items || []).map(item => item.postId);
        res.json(savedPostIds);
    } catch (err) {
        console.error('Get saved posts error:', err);
        res.status(500).json({ error: 'Failed to get saved posts' });
    }
});

module.exports = router;

