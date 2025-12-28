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

    // Set default NSFW flags (will be overridden by NSFW table lookup in feed endpoint)
    if (post.isNSFW === undefined) post.isNSFW = false;
    if (post.isSensitive === undefined) post.isSensitive = false;

    return post;
}

// Simple in-memory cache for feed
let feedCache = null;
let feedCacheTime = 0;
const CACHE_TTL = 5000; // 5 seconds - short TTL for fresh content

// Function to clear feed cache (called when NSFW flags change)
function clearFeedCache() {
    feedCache = null;
    feedCacheTime = 0;
    console.log('[Feed] Cache cleared');
}

// Algorithm server URL for personalized recommendations
const ALGORITHM_SERVER = 'http://54.235.126.186:8000';
const http = require('http');

// Helper to fetch recommendations from algorithm server
async function fetchRecommendations(userId, limit = 20) {
    return new Promise((resolve) => {
        const url = `${ALGORITHM_SERVER}/api/recommendations/${userId}?limit=${limit}`;

        const timeoutMs = 3000; // 3 second timeout
        const req = http.get(url, { timeout: timeoutMs }, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const result = JSON.parse(data);
                    resolve(result);
                } catch (e) {
                    console.log('[Feed] Algorithm parse error:', e.message);
                    resolve(null);
                }
            });
        });

        req.on('error', (e) => {
            console.log('[Feed] Algorithm server error:', e.message);
            resolve(null);
        });

        req.on('timeout', () => {
            console.log('[Feed] Algorithm server timeout');
            req.destroy();
            resolve(null);
        });
    });
}

// Get feed posts (PUBLIC - no auth required for browsing)
// Uses algorithm for logged-in users, chronological for guests
router.get('/feed', async (req, res) => {
    try {
        const now = Date.now();
        const userId = req.headers['x-user-id'] || req.query.userId;

        // Try algorithm-based recommendations for logged-in users
        let algorithmRecommendations = null;
        if (userId) {
            console.log(`[Feed] Fetching recommendations for user: ${userId.substring(0, 8)}...`);
            algorithmRecommendations = await fetchRecommendations(userId, 50);
        }

        // Get ALL posts from database using pagination (DynamoDB returns max 1MB per scan)
        let allPosts = [];
        let lastEvaluatedKey = undefined;
        do {
            const scanParams = {
                TableName: Tables.POSTS,
            };
            if (lastEvaluatedKey) {
                scanParams.ExclusiveStartKey = lastEvaluatedKey;
            }
            const postsResult = await docClient.send(new ScanCommand(scanParams));
            allPosts = allPosts.concat(postsResult.Items || []);
            lastEvaluatedKey = postsResult.LastEvaluatedKey;
            console.log(`[Feed] Fetched ${postsResult.Items?.length || 0} posts (total: ${allPosts.length})`);
        } while (lastEvaluatedKey);

        console.log(`[Feed] Total posts fetched: ${allPosts.length}`);

        // CRITICAL: Also fetch NSFW post IDs from NSFW table
        let nsfwPostIds = new Set();
        try {
            const nsfwResult = await docClient.send(new ScanCommand({
                TableName: 'Buddylynk_NSFW',
                FilterExpression: 'isAdult = :adult',
                ExpressionAttributeValues: { ':adult': true },
                ProjectionExpression: 'postId'
            }));
            nsfwPostIds = new Set((nsfwResult.Items || []).map(item => item.postId));
            console.log(`[Feed] Found ${nsfwPostIds.size} NSFW posts to flag`);
        } catch (nsfwErr) {
            console.error('[Feed] NSFW table query failed:', nsfwErr.message);
        }

        let posts = allPosts;

        // Convert all media URLs to CloudFront and apply NSFW flags
        posts = posts.map(post => {
            const converted = convertPostMediaUrls(post);
            // Override isNSFW if found in NSFW table
            if (nsfwPostIds.has(converted.postId)) {
                converted.isNSFW = true;
                converted.isSensitive = true;
            }
            return converted;
        });

        // Build a map for quick lookups
        const postMap = new Map(posts.map(p => [p.postId, p]));

        // Sort posts based on algorithm recommendations OR chronological
        let sortedPosts;
        if (algorithmRecommendations && algorithmRecommendations.recommendations && algorithmRecommendations.recommendations.length > 0) {
            // Use algorithm recommendations order
            console.log(`[Feed] Using ${algorithmRecommendations.recommendations.length} algorithm recommendations`);

            const recommendedIds = new Set(algorithmRecommendations.recommendations.map(r => r.contentId));

            // Put recommended posts first (in algorithm order), then rest chronologically
            const recommendedPosts = algorithmRecommendations.recommendations
                .map(r => postMap.get(r.contentId))
                .filter(p => p); // Filter out any null values

            const otherPosts = posts
                .filter(p => !recommendedIds.has(p.postId) && p.userId !== userId) // Exclude own posts
                .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

            sortedPosts = [...recommendedPosts, ...otherPosts];
        } else {
            // Fallback: Chronological with randomization for variety
            console.log('[Feed] Using chronological sort with variety');

            // Group by user to avoid too many posts from same user
            const postsByUser = {};
            posts.forEach(p => {
                if (!postsByUser[p.userId]) postsByUser[p.userId] = [];
                postsByUser[p.userId].push(p);
            });

            // Sort each user's posts by date
            Object.values(postsByUser).forEach(userPosts => {
                userPosts.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
            });

            // Interleave posts from different users for variety
            sortedPosts = [];
            const userIds = Object.keys(postsByUser);
            let round = 0;
            // NO LIMIT on rounds - include ALL posts
            while (sortedPosts.length < posts.length && round < 1000) {
                for (const uid of userIds) {
                    if (postsByUser[uid][round]) {
                        sortedPosts.push(postsByUser[uid][round]);
                    }
                }
                round++;
            }
        }

        // NO LIMIT - return all posts for unlimited feed
        // sortedPosts = sortedPosts.slice(0, 50); // REMOVED

        // Cache the result (short TTL for freshness)
        feedCache = sortedPosts;
        feedCacheTime = now;

        res.json(sortedPosts);
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

// Like post - adds user to likedBy array and increments likes count
router.post('/:postId/like', verifyToken, async (req, res) => {
    try {
        const { postId } = req.params;
        const userId = req.userId;

        // Add user to likedBy and increment likes count
        await docClient.send(new UpdateCommand({
            TableName: Tables.POSTS,
            Key: { postId },
            UpdateExpression: 'SET likes = if_not_exists(likes, :zero) + :inc, likedBy = list_append(if_not_exists(likedBy, :empty), :user)',
            ExpressionAttributeValues: {
                ':inc': 1,
                ':zero': 0,
                ':empty': [],
                ':user': [userId]
            }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Like error:', err);
        res.status(500).json({ error: 'Failed to like post' });
    }
});

// Unlike post - removes user from likedBy and decrements likes count
router.delete('/:postId/like', verifyToken, async (req, res) => {
    try {
        const { postId } = req.params;
        const userId = req.userId;

        // First get the post to find user index in likedBy
        const getResult = await docClient.send(new GetCommand({
            TableName: Tables.POSTS,
            Key: { postId }
        }));

        if (!getResult.Item) {
            return res.status(404).json({ error: 'Post not found' });
        }

        const likedBy = getResult.Item.likedBy || [];
        const userIndex = likedBy.indexOf(userId);

        if (userIndex === -1) {
            return res.json({ success: true }); // User hasn't liked this post
        }

        // Remove user from likedBy and decrement likes
        await docClient.send(new UpdateCommand({
            TableName: Tables.POSTS,
            Key: { postId },
            UpdateExpression: 'SET likes = likes - :dec REMOVE likedBy[' + userIndex + ']',
            ExpressionAttributeValues: { ':dec': 1 },
            ConditionExpression: 'likes > :zero',
            ExpressionAttributeValues: { ':dec': 1, ':zero': 0 }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Unlike error:', err);
        res.status(500).json({ error: 'Failed to unlike post' });
    }
});

// Share post - increments shares count (no auth required for sharing)
router.post('/:postId/share', async (req, res) => {
    try {
        const { postId } = req.params;

        await docClient.send(new UpdateCommand({
            TableName: Tables.POSTS,
            Key: { postId },
            UpdateExpression: 'SET shares = if_not_exists(shares, :zero) + :inc',
            ExpressionAttributeValues: { ':inc': 1, ':zero': 0 }
        }));

        res.json({ success: true });
    } catch (err) {
        console.error('Share error:', err);
        res.status(500).json({ error: 'Failed to share post' });
    }
});

// Add comment to post
router.post('/:postId/comment', verifyToken, async (req, res) => {
    try {
        const { postId } = req.params;
        const { text } = req.body;
        const userId = req.userId;

        if (!text || text.trim().length === 0) {
            return res.status(400).json({ error: 'Comment text is required' });
        }

        const comment = {
            commentId: uuidv4(),
            userId,
            text: text.trim(),
            createdAt: new Date().toISOString()
        };

        await docClient.send(new UpdateCommand({
            TableName: Tables.POSTS,
            Key: { postId },
            UpdateExpression: 'SET comments = list_append(if_not_exists(comments, :empty), :comment)',
            ExpressionAttributeValues: {
                ':empty': [],
                ':comment': [comment]
            }
        }));

        res.json({ success: true, comment });
    } catch (err) {
        console.error('Comment error:', err);
        res.status(500).json({ error: 'Failed to add comment' });
    }
});

// Get comments for post
router.get('/:postId/comments', async (req, res) => {
    try {
        const { postId } = req.params;

        const result = await docClient.send(new GetCommand({
            TableName: Tables.POSTS,
            Key: { postId },
            ProjectionExpression: 'comments'
        }));

        const comments = result.Item?.comments || [];
        res.json(comments);
    } catch (err) {
        console.error('Get comments error:', err);
        res.status(500).json({ error: 'Failed to get comments' });
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
module.exports.clearFeedCache = clearFeedCache;

