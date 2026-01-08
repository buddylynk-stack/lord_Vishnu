const express = require('express');
const router = express.Router();
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, GetCommand, PutCommand, UpdateCommand, QueryCommand, ScanCommand, DeleteCommand } = require('@aws-sdk/lib-dynamodb');
const { v4: uuidv4 } = require('uuid');

// Initialize DynamoDB
const client = new DynamoDBClient({ region: process.env.AWS_REGION || 'ap-south-1' });
const docClient = DynamoDBDocumentClient.from(client);

const OTT_TABLE = 'buddylynk-ott-videos';

// ============================================================================
// GET /api/ott/videos - Get video feed (paginated)
// ============================================================================
router.get('/videos', async (req, res) => {
    try {
        const { category, limit = 20, lastKey } = req.query;
        const userId = req.user?.userId;

        let params = {
            TableName: OTT_TABLE,
            Limit: parseInt(limit)
        };

        // Filter by category if provided
        if (category && category !== 'All') {
            params.FilterExpression = 'category = :cat';
            params.ExpressionAttributeValues = { ':cat': category };
        }

        if (lastKey) {
            params.ExclusiveStartKey = JSON.parse(Buffer.from(lastKey, 'base64').toString());
        }

        const result = await docClient.send(new ScanCommand(params));

        // Sort by createdAt descending
        const videos = (result.Items || []).sort((a, b) =>
            new Date(b.createdAt) - new Date(a.createdAt)
        );

        // Add isLiked status for current user
        const videosWithStatus = videos.map(video => ({
            ...video,
            isLiked: video.likedBy?.includes(userId) || false
        }));

        res.json({
            success: true,
            videos: videosWithStatus,
            lastKey: result.LastEvaluatedKey
                ? Buffer.from(JSON.stringify(result.LastEvaluatedKey)).toString('base64')
                : null
        });
    } catch (error) {
        console.error('Error fetching OTT videos:', error);
        res.status(500).json({ success: false, message: 'Failed to fetch videos' });
    }
});

// ============================================================================
// GET /api/ott/videos/:videoId - Get single video
// ============================================================================
router.get('/videos/:videoId', async (req, res) => {
    try {
        const { videoId } = req.params;
        const userId = req.user?.userId;

        const result = await docClient.send(new GetCommand({
            TableName: OTT_TABLE,
            Key: { videoId }
        }));

        if (!result.Item) {
            return res.status(404).json({ success: false, message: 'Video not found' });
        }

        const video = {
            ...result.Item,
            isLiked: result.Item.likedBy?.includes(userId) || false,
            isSaved: result.Item.savedBy?.includes(userId) || false
        };

        res.json({ success: true, video });
    } catch (error) {
        console.error('Error fetching video:', error);
        res.status(500).json({ success: false, message: 'Failed to fetch video' });
    }
});

// ============================================================================
// POST /api/ott/videos - Upload new video
// ============================================================================
router.post('/videos', async (req, res) => {
    try {
        const { title, description, videoUrl, thumbnailUrl, category, duration, tags } = req.body;
        const userId = req.user?.userId;
        const userName = req.user?.username || 'Unknown';
        const userAvatar = req.user?.avatarUrl;

        if (!title || !videoUrl) {
            return res.status(400).json({
                success: false,
                message: 'Title and video URL are required'
            });
        }

        const videoId = uuidv4();
        const now = new Date().toISOString();

        const video = {
            videoId,
            title,
            description: description || '',
            videoUrl,
            thumbnailUrl,
            duration: duration || 0,
            category: category || 'General',
            tags: tags || [],
            viewCount: 0,
            likeCount: 0,
            commentCount: 0,
            creatorId: userId,
            creatorName: userName,
            creatorAvatar: userAvatar,
            likedBy: [],
            savedBy: [],
            createdAt: now,
            updatedAt: now
        };

        await docClient.send(new PutCommand({
            TableName: OTT_TABLE,
            Item: video
        }));

        res.json({ success: true, video, message: 'Video uploaded successfully' });
    } catch (error) {
        console.error('Error uploading video:', error);
        res.status(500).json({ success: false, message: 'Failed to upload video' });
    }
});

// ============================================================================
// POST /api/ott/videos/:videoId/like - Like/unlike video
// ============================================================================
router.post('/videos/:videoId/like', async (req, res) => {
    try {
        const { videoId } = req.params;
        const userId = req.user?.userId;

        if (!userId) {
            return res.status(401).json({ success: false, message: 'Unauthorized' });
        }

        // Get current video
        const getResult = await docClient.send(new GetCommand({
            TableName: OTT_TABLE,
            Key: { videoId }
        }));

        if (!getResult.Item) {
            return res.status(404).json({ success: false, message: 'Video not found' });
        }

        const video = getResult.Item;
        const likedBy = video.likedBy || [];
        const isLiked = likedBy.includes(userId);

        let newLikedBy, newLikeCount;

        if (isLiked) {
            // Unlike
            newLikedBy = likedBy.filter(id => id !== userId);
            newLikeCount = Math.max(0, (video.likeCount || 1) - 1);
        } else {
            // Like
            newLikedBy = [...likedBy, userId];
            newLikeCount = (video.likeCount || 0) + 1;
        }

        await docClient.send(new UpdateCommand({
            TableName: OTT_TABLE,
            Key: { videoId },
            UpdateExpression: 'SET likedBy = :likedBy, likeCount = :likeCount, updatedAt = :now',
            ExpressionAttributeValues: {
                ':likedBy': newLikedBy,
                ':likeCount': newLikeCount,
                ':now': new Date().toISOString()
            }
        }));

        res.json({
            success: true,
            isLiked: !isLiked,
            likeCount: newLikeCount
        });
    } catch (error) {
        console.error('Error liking video:', error);
        res.status(500).json({ success: false, message: 'Failed to like video' });
    }
});

// ============================================================================
// POST /api/ott/videos/:videoId/view - Increment view count
// ============================================================================
router.post('/videos/:videoId/view', async (req, res) => {
    try {
        const { videoId } = req.params;

        await docClient.send(new UpdateCommand({
            TableName: OTT_TABLE,
            Key: { videoId },
            UpdateExpression: 'SET viewCount = if_not_exists(viewCount, :zero) + :inc',
            ExpressionAttributeValues: {
                ':zero': 0,
                ':inc': 1
            }
        }));

        res.json({ success: true });
    } catch (error) {
        console.error('Error incrementing view:', error);
        res.status(500).json({ success: false, message: 'Failed to increment view' });
    }
});

// ============================================================================
// GET /api/ott/trending - Get trending videos
// ============================================================================
router.get('/trending', async (req, res) => {
    try {
        const userId = req.user?.userId;
        const { limit = 10 } = req.query;

        const result = await docClient.send(new ScanCommand({
            TableName: OTT_TABLE,
            Limit: 50
        }));

        // Sort by view count + like count (trending score)
        const videos = (result.Items || [])
            .map(v => ({
                ...v,
                trendingScore: (v.viewCount || 0) + (v.likeCount || 0) * 2,
                isLiked: v.likedBy?.includes(userId) || false
            }))
            .sort((a, b) => b.trendingScore - a.trendingScore)
            .slice(0, parseInt(limit));

        res.json({ success: true, videos });
    } catch (error) {
        console.error('Error fetching trending:', error);
        res.status(500).json({ success: false, message: 'Failed to fetch trending' });
    }
});

// ============================================================================
// GET /api/ott/search - Search videos
// ============================================================================
router.get('/search', async (req, res) => {
    try {
        const { q, limit = 20 } = req.query;
        const userId = req.user?.userId;

        if (!q) {
            return res.json({ success: true, videos: [] });
        }

        const result = await docClient.send(new ScanCommand({
            TableName: OTT_TABLE,
            FilterExpression: 'contains(#title, :query) OR contains(#desc, :query)',
            ExpressionAttributeNames: {
                '#title': 'title',
                '#desc': 'description'
            },
            ExpressionAttributeValues: {
                ':query': q.toLowerCase()
            },
            Limit: parseInt(limit)
        }));

        const videos = (result.Items || []).map(v => ({
            ...v,
            isLiked: v.likedBy?.includes(userId) || false
        }));

        res.json({ success: true, videos });
    } catch (error) {
        console.error('Error searching videos:', error);
        res.status(500).json({ success: false, message: 'Failed to search videos' });
    }
});

// ============================================================================
// GET /api/ott/my-videos - Get current user's videos
// ============================================================================
router.get('/my-videos', async (req, res) => {
    try {
        const userId = req.user?.userId;

        if (!userId) {
            return res.status(401).json({ success: false, message: 'Unauthorized' });
        }

        const result = await docClient.send(new ScanCommand({
            TableName: OTT_TABLE,
            FilterExpression: 'creatorId = :userId',
            ExpressionAttributeValues: { ':userId': userId }
        }));

        const videos = (result.Items || []).sort((a, b) =>
            new Date(b.createdAt) - new Date(a.createdAt)
        );

        res.json({ success: true, videos });
    } catch (error) {
        console.error('Error fetching user videos:', error);
        res.status(500).json({ success: false, message: 'Failed to fetch videos' });
    }
});

// ============================================================================
// DELETE /api/ott/videos/:videoId - Delete video
// ============================================================================
router.delete('/videos/:videoId', async (req, res) => {
    try {
        const { videoId } = req.params;
        const userId = req.user?.userId;

        // Check ownership
        const getResult = await docClient.send(new GetCommand({
            TableName: OTT_TABLE,
            Key: { videoId }
        }));

        if (!getResult.Item) {
            return res.status(404).json({ success: false, message: 'Video not found' });
        }

        if (getResult.Item.creatorId !== userId) {
            return res.status(403).json({ success: false, message: 'Not authorized' });
        }

        await docClient.send(new DeleteCommand({
            TableName: OTT_TABLE,
            Key: { videoId }
        }));

        res.json({ success: true, message: 'Video deleted' });
    } catch (error) {
        console.error('Error deleting video:', error);
        res.status(500).json({ success: false, message: 'Failed to delete video' });
    }
});

module.exports = router;
