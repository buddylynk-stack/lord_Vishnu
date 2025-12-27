const express = require('express');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');
const { PutObjectCommand } = require('@aws-sdk/client-s3');
const { s3Client, S3_BUCKET } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');

const router = express.Router();

// Generate pre-signed URL for upload
router.post('/presign', verifyToken, async (req, res) => {
    try {
        const { filename, contentType, folder } = req.body;

        if (!filename || !contentType) {
            return res.status(400).json({ error: 'Filename and contentType required' });
        }

        const validFolders = ['profiles', 'posts', 'stories', 'messages', 'groups'];
        const safeFolder = validFolders.includes(folder) ? folder : 'uploads';

        // Generate unique key
        const timestamp = Date.now();
        const safeFilename = filename.replace(/[^a-zA-Z0-9.-]/g, '_');
        const key = `${safeFolder}/${req.userId}/${timestamp}-${safeFilename}`;

        // Create pre-signed URL (valid for 5 minutes)
        const command = new PutObjectCommand({
            Bucket: S3_BUCKET,
            Key: key,
            ContentType: contentType
        });

        const uploadUrl = await getSignedUrl(s3Client, command, { expiresIn: 300 });

        // Return the upload URL and the final file URL
        const fileUrl = `https://${S3_BUCKET}.s3.us-east-1.amazonaws.com/${key}`;

        res.json({ uploadUrl, fileUrl, key });
    } catch (err) {
        console.error('Presign error:', err);
        res.status(500).json({ error: 'Failed to generate upload URL' });
    }
});

module.exports = router;
