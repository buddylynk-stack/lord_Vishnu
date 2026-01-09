const express = require('express');
const router = express.Router();
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, GetCommand, PutCommand, UpdateCommand, QueryCommand, ScanCommand, DeleteCommand } = require('@aws-sdk/lib-dynamodb');
const { v4: uuidv4 } = require('uuid');
const { verifyToken } = require('../middleware/auth');

// Initialize DynamoDB
const client = new DynamoDBClient({ region: process.env.AWS_REGION || 'ap-south-1' });
const docClient = DynamoDBDocumentClient.from(client);

const OTT_TABLE = 'buddylynk-ott-videos';
const OTT_COMMENTS_TABLE = 'buddylynk-ott-comments';
const OTT_HISTORY_TABLE = 'buddylynk-ott-history';

// ============================================================================
// GET /api/ott/videos - Get video feed (paginated)
// ============================================================================
router.get('/videos', async (req, res) => {
    try {
        const { category, limit = 50, lastKey } = req.query;
        const userId = req.user?.userId;

        // Scan all videos (no limit on initial scan to get all items)
        let params = {
            TableName: OTT_TABLE
        };

        // Filter by category if provided
        if (category && category !== 'All') {
            params.FilterExpression = 'category = :cat';
            params.ExpressionAttributeValues = { ':cat': category };
        }

        if (lastKey) {
            params.ExclusiveStartKey = JSON.parse(Buffer.from(lastKey, 'base64').toString());
        }

        // Fetch all items (DynamoDB may paginate, so we handle that)
        let allItems = [];
        let lastEvaluatedKey = null;

        do {
            const result = await docClient.send(new ScanCommand(params));
            allItems = allItems.concat(result.Items || []);
            lastEvaluatedKey = result.LastEvaluatedKey;
            params.ExclusiveStartKey = lastEvaluatedKey;
        } while (lastEvaluatedKey && allItems.length < 100); // Get up to 100 videos

        // Sort by createdAt descending
        const videos = allItems.sort((a, b) =>
            new Date(b.createdAt) - new Date(a.createdAt)
        );

        // Apply limit after sorting
        const limitedVideos = videos.slice(0, parseInt(limit));

        // Add isLiked status for current user
        const videosWithStatus = limitedVideos.map(video => ({
            ...video,
            isLiked: video.likedBy?.includes(userId) || false
        }));

        res.json({
            success: true,
            videos: videosWithStatus,
            total: videos.length,
            lastKey: null // Simplified - no pagination for now
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
// DELETE /api/ott/videos/:videoId - Delete video (creator only) with S3 cleanup
// ============================================================================
router.delete('/videos/:videoId', verifyToken, async (req, res) => {
    try {
        const { videoId } = req.params;
        const userId = req.userId; // verifyToken sets req.userId

        console.log(`Delete request: videoId=${videoId}, userId=${userId}`);

        if (!userId) {
            return res.status(401).json({ success: false, message: 'Unauthorized - no user ID' });
        }

        // Get video to verify ownership and get URLs for S3 deletion
        const getResult = await docClient.send(new GetCommand({
            TableName: OTT_TABLE,
            Key: { videoId }
        }));

        if (!getResult.Item) {
            return res.status(404).json({ success: false, message: 'Video not found' });
        }

        const video = getResult.Item;
        console.log(`Video creatorId=${video.creatorId}, requesting userId=${userId}`);

        // Verify ownership - only creator can delete
        if (video.creatorId !== userId) {
            console.log(`Authorization failed: ${video.creatorId} !== ${userId}`);
            return res.status(403).json({ success: false, message: 'You can only delete your own videos' });
        }

        // Delete from S3 (video and thumbnail)
        const { S3Client, DeleteObjectCommand } = require('@aws-sdk/client-s3');
        const s3Client = new S3Client({ region: process.env.AWS_REGION || 'ap-south-1' });
        const bucketName = process.env.S3_BUCKET || 'buddylynk-uploads';

        // Extract S3 keys from URLs
        const extractS3Key = (url) => {
            if (!url) return null;
            try {
                // Handle CloudFront URLs: https://cdn.buddylynk.com/ott/file.mp4
                // Handle S3 URLs: https://bucket.s3.amazonaws.com/ott/file.mp4
                const urlObj = new URL(url);
                let key = urlObj.pathname.replace(/^\//, ''); // Remove leading slash
                return key || null;
            } catch (e) {
                return null;
            }
        };

        const videoKey = extractS3Key(video.videoUrl);
        const thumbnailKey = extractS3Key(video.thumbnailUrl);

        // Delete video file from S3
        if (videoKey) {
            try {
                await s3Client.send(new DeleteObjectCommand({
                    Bucket: bucketName,
                    Key: videoKey
                }));
                console.log(`Deleted video from S3: ${videoKey}`);
            } catch (s3Error) {
                console.error('Error deleting video from S3:', s3Error);
                // Continue even if S3 delete fails
            }
        }

        // Delete thumbnail from S3
        if (thumbnailKey) {
            try {
                await s3Client.send(new DeleteObjectCommand({
                    Bucket: bucketName,
                    Key: thumbnailKey
                }));
                console.log(`Deleted thumbnail from S3: ${thumbnailKey}`);
            } catch (s3Error) {
                console.error('Error deleting thumbnail from S3:', s3Error);
            }
        }

        // Delete video from DynamoDB
        await docClient.send(new DeleteCommand({
            TableName: OTT_TABLE,
            Key: { videoId }
        }));

        // Also delete all comments for this video
        try {
            const commentsResult = await docClient.send(new QueryCommand({
                TableName: OTT_COMMENTS_TABLE,
                KeyConditionExpression: 'videoId = :vid',
                ExpressionAttributeValues: { ':vid': videoId }
            }));

            for (const comment of (commentsResult.Items || [])) {
                await docClient.send(new DeleteCommand({
                    TableName: OTT_COMMENTS_TABLE,
                    Key: { videoId: comment.videoId, commentId: comment.commentId }
                }));
            }
        } catch (commentError) {
            console.error('Error deleting comments:', commentError);
        }

        res.json({ success: true, message: 'Video deleted successfully' });
    } catch (error) {
        console.error('Error deleting video:', error);
        res.status(500).json({ success: false, message: 'Failed to delete video' });
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
// ============================================================================
// POST /api/ott/videos/:videoId/save - Save/unsave video (bookmark)
// ============================================================================
router.post('/videos/:videoId/save', async (req, res) => {
    try {
        const { videoId } = req.params;
        const userId = req.user?.userId;

        if (!userId) {
            return res.status(401).json({ success: false, message: 'Unauthorized' });
        }

        const getResult = await docClient.send(new GetCommand({
            TableName: OTT_TABLE,
            Key: { videoId }
        }));

        if (!getResult.Item) {
            return res.status(404).json({ success: false, message: 'Video not found' });
        }

        const video = getResult.Item;
        const savedBy = video.savedBy || [];
        const isSaved = savedBy.includes(userId);

        let newSavedBy;
        if (isSaved) {
            newSavedBy = savedBy.filter(id => id !== userId);
        } else {
            newSavedBy = [...savedBy, userId];
        }

        await docClient.send(new UpdateCommand({
            TableName: OTT_TABLE,
            Key: { videoId },
            UpdateExpression: 'SET savedBy = :savedBy, updatedAt = :now',
            ExpressionAttributeValues: {
                ':savedBy': newSavedBy,
                ':now': new Date().toISOString()
            }
        }));

        res.json({ success: true, isSaved: !isSaved });
    } catch (error) {
        console.error('Error saving video:', error);
        res.status(500).json({ success: false, message: 'Failed to save video' });
    }
});

// ============================================================================
// GET /api/ott/saved - Get user's saved videos
// ============================================================================
router.get('/saved', async (req, res) => {
    try {
        const userId = req.user?.userId;

        if (!userId) {
            return res.status(401).json({ success: false, message: 'Unauthorized' });
        }

        const result = await docClient.send(new ScanCommand({
            TableName: OTT_TABLE,
            FilterExpression: 'contains(savedBy, :userId)',
            ExpressionAttributeValues: { ':userId': userId }
        }));

        const videos = (result.Items || []).sort((a, b) =>
            new Date(b.createdAt) - new Date(a.createdAt)
        );

        res.json({ success: true, videos });
    } catch (error) {
        console.error('Error fetching saved videos:', error);
        res.status(500).json({ success: false, message: 'Failed to fetch saved videos' });
    }
});

// ============================================================================
// GET /api/ott/categories - Get available categories
// ============================================================================
router.get('/categories', async (req, res) => {
    try {
        const categories = [
            { id: 'all', name: 'All', icon: '🎬' },
            { id: 'entertainment', name: 'Entertainment', icon: '🎭' },
            { id: 'music', name: 'Music', icon: '🎵' },
            { id: 'gaming', name: 'Gaming', icon: '🎮' },
            { id: 'sports', name: 'Sports', icon: '⚽' },
            { id: 'education', name: 'Education', icon: '📚' },
            { id: 'comedy', name: 'Comedy', icon: '😂' },
            { id: 'news', name: 'News', icon: '📰' },
            { id: 'lifestyle', name: 'Lifestyle', icon: '✨' },
            { id: 'tech', name: 'Technology', icon: '💻' }
        ];
        res.json({ success: true, categories });
    } catch (error) {
        res.status(500).json({ success: false, message: 'Failed to fetch categories' });
    }
});

// ============================================================================
// GET /api/ott/creator/:creatorId - Get videos by creator
// ============================================================================
router.get('/creator/:creatorId', async (req, res) => {
    try {
        const { creatorId } = req.params;
        const userId = req.user?.userId;

        const result = await docClient.send(new ScanCommand({
            TableName: OTT_TABLE,
            FilterExpression: 'creatorId = :creatorId',
            ExpressionAttributeValues: { ':creatorId': creatorId }
        }));

        const videos = (result.Items || [])
            .map(v => ({
                ...v,
                isLiked: v.likedBy?.includes(userId) || false,
                isSaved: v.savedBy?.includes(userId) || false
            }))
            .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

        res.json({ success: true, videos, count: videos.length });
    } catch (error) {
        console.error('Error fetching creator videos:', error);
        res.status(500).json({ success: false, message: 'Failed to fetch creator videos' });
    }
});

// ============================================================================
// POST /api/ott/videos/:videoId/comments - Add comment to video
// ============================================================================
router.post('/videos/:videoId/comments', async (req, res) => {
    try {
        const { videoId } = req.params;
        const { text } = req.body;
        const userId = req.user?.userId;
        const userName = req.user?.username || 'Unknown';
        const userAvatar = req.user?.avatarUrl;

        if (!userId) {
            return res.status(401).json({ success: false, message: 'Unauthorized' });
        }

        if (!text || text.trim().length === 0) {
            return res.status(400).json({ success: false, message: 'Comment text required' });
        }

        const commentId = uuidv4();
        const now = new Date().toISOString();

        const comment = {
            videoId,
            commentId,
            userId,
            userName,
            userAvatar,
            text: text.trim(),
            likeCount: 0,
            likedBy: [],
            createdAt: now
        };

        await docClient.send(new PutCommand({
            TableName: OTT_COMMENTS_TABLE,
            Item: comment
        }));

        // Update comment count on video
        await docClient.send(new UpdateCommand({
            TableName: OTT_TABLE,
            Key: { videoId },
            UpdateExpression: 'SET commentCount = if_not_exists(commentCount, :zero) + :inc',
            ExpressionAttributeValues: { ':zero': 0, ':inc': 1 }
        }));

        res.json({ success: true, comment });
    } catch (error) {
        console.error('Error adding comment:', error);
        res.status(500).json({ success: false, message: 'Failed to add comment' });
    }
});

// ============================================================================
// GET /api/ott/videos/:videoId/comments - Get comments for video
// ============================================================================
router.get('/videos/:videoId/comments', async (req, res) => {
    try {
        const { videoId } = req.params;
        const { limit = 50 } = req.query;

        const result = await docClient.send(new QueryCommand({
            TableName: OTT_COMMENTS_TABLE,
            KeyConditionExpression: 'videoId = :videoId',
            ExpressionAttributeValues: { ':videoId': videoId },
            Limit: parseInt(limit),
            ScanIndexForward: false // Newest first
        }));

        res.json({ success: true, comments: result.Items || [] });
    } catch (error) {
        console.error('Error fetching comments:', error);
        res.status(500).json({ success: false, message: 'Failed to fetch comments' });
    }
});

// ============================================================================
// DELETE /api/ott/comments/:commentId - Delete comment
// ============================================================================
router.delete('/videos/:videoId/comments/:commentId', async (req, res) => {
    try {
        const { videoId, commentId } = req.params;
        const userId = req.user?.userId;

        // Check ownership
        const getResult = await docClient.send(new GetCommand({
            TableName: OTT_COMMENTS_TABLE,
            Key: { videoId, commentId }
        }));

        if (!getResult.Item) {
            return res.status(404).json({ success: false, message: 'Comment not found' });
        }

        if (getResult.Item.userId !== userId) {
            return res.status(403).json({ success: false, message: 'Not authorized' });
        }

        await docClient.send(new DeleteCommand({
            TableName: OTT_COMMENTS_TABLE,
            Key: { videoId, commentId }
        }));

        // Decrement comment count
        await docClient.send(new UpdateCommand({
            TableName: OTT_TABLE,
            Key: { videoId },
            UpdateExpression: 'SET commentCount = commentCount - :dec',
            ExpressionAttributeValues: { ':dec': 1 }
        }));

        res.json({ success: true, message: 'Comment deleted' });
    } catch (error) {
        console.error('Error deleting comment:', error);
        res.status(500).json({ success: false, message: 'Failed to delete comment' });
    }
});

// ============================================================================
// POST /api/ott/videos/:videoId/history - Save watch history
// ============================================================================
router.post('/videos/:videoId/history', async (req, res) => {
    try {
        const { videoId } = req.params;
        const { progress, duration } = req.body;
        const userId = req.user?.userId;

        if (!userId) {
            return res.status(401).json({ success: false, message: 'Unauthorized' });
        }

        await docClient.send(new PutCommand({
            TableName: OTT_HISTORY_TABLE,
            Item: {
                userId,
                videoId,
                progress: progress || 0,
                duration: duration || 0,
                watchedAt: new Date().toISOString()
            }
        }));

        res.json({ success: true });
    } catch (error) {
        console.error('Error saving watch history:', error);
        res.status(500).json({ success: false, message: 'Failed to save history' });
    }
});

// ============================================================================
// GET /api/ott/history - Get user's watch history
// ============================================================================
router.get('/history', async (req, res) => {
    try {
        const userId = req.user?.userId;
        const { limit = 20 } = req.query;

        if (!userId) {
            return res.status(401).json({ success: false, message: 'Unauthorized' });
        }

        const result = await docClient.send(new QueryCommand({
            TableName: OTT_HISTORY_TABLE,
            KeyConditionExpression: 'userId = :userId',
            ExpressionAttributeValues: { ':userId': userId },
            Limit: parseInt(limit),
            ScanIndexForward: false
        }));

        // Fetch video details for each history item
        const historyWithVideos = await Promise.all(
            (result.Items || []).map(async (item) => {
                try {
                    const videoResult = await docClient.send(new GetCommand({
                        TableName: OTT_TABLE,
                        Key: { videoId: item.videoId }
                    }));
                    return { ...item, video: videoResult.Item };
                } catch {
                    return item;
                }
            })
        );

        res.json({ success: true, history: historyWithVideos });
    } catch (error) {
        console.error('Error fetching watch history:', error);
        res.status(500).json({ success: false, message: 'Failed to fetch history' });
    }
});

// ============================================================================
// POST /api/ott/videos/:videoId/report - Report video
// ============================================================================
router.post('/videos/:videoId/report', async (req, res) => {
    try {
        const { videoId } = req.params;
        const { reason } = req.body;
        const userId = req.user?.userId;

        if (!userId) {
            return res.status(401).json({ success: false, message: 'Unauthorized' });
        }

        // Add to reported list on video
        await docClient.send(new UpdateCommand({
            TableName: OTT_TABLE,
            Key: { videoId },
            UpdateExpression: 'SET reportedBy = list_append(if_not_exists(reportedBy, :empty), :report), reportCount = if_not_exists(reportCount, :zero) + :inc',
            ExpressionAttributeValues: {
                ':empty': [],
                ':report': [{ userId, reason: reason || 'Inappropriate content', reportedAt: new Date().toISOString() }],
                ':zero': 0,
                ':inc': 1
            }
        }));

        res.json({ success: true, message: 'Video reported' });
    } catch (error) {
        console.error('Error reporting video:', error);
        res.status(500).json({ success: false, message: 'Failed to report video' });
    }
});

module.exports = router;
