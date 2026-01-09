const express = require('express');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');
const { PutObjectCommand, CreateMultipartUploadCommand, UploadPartCommand, CompleteMultipartUploadCommand, AbortMultipartUploadCommand } = require('@aws-sdk/client-s3');
const { s3Client, S3_BUCKET } = require('../config/aws');
const { verifyToken } = require('../middleware/auth');

const router = express.Router();

const CLOUDFRONT_DOMAIN = 'd2cwas7x7omdpp.cloudfront.net';

// Generate pre-signed URL for upload
router.post('/presign', verifyToken, async (req, res) => {
    try {
        const { filename, contentType, folder } = req.body;

        if (!filename || !contentType) {
            return res.status(400).json({ error: 'Filename and contentType required' });
        }

        const validFolders = ['profiles', 'posts', 'stories', 'messages', 'groups', 'ott'];
        const safeFolder = validFolders.includes(folder) ? folder : 'uploads';

        // Generate unique key
        const timestamp = Date.now();
        const safeFilename = filename.replace(/[^a-zA-Z0-9.-]/g, '_');
        const key = `${safeFolder}/${req.userId}/${timestamp}-${safeFilename}`;

        // Create pre-signed URL (valid for 5 minutes)
        // ACL: public-read allows files to be accessed via CloudFront CDN
        const command = new PutObjectCommand({
            Bucket: S3_BUCKET,
            Key: key,
            ContentType: contentType,
            ACL: 'public-read'
        });

        const uploadUrl = await getSignedUrl(s3Client, command, { expiresIn: 300 });

        // Return the upload URL and the final CDN file URL (CloudFront for faster delivery)
        const fileUrl = `https://${CLOUDFRONT_DOMAIN}/${key}`;
        const s3Url = `https://${S3_BUCKET}.s3.us-east-1.amazonaws.com/${key}`;

        console.log(`Upload: S3=${s3Url}, CDN=${fileUrl}`);
        res.json({ uploadUrl, fileUrl, key, s3Url });
    } catch (err) {
        console.error('Presign error:', err);
        res.status(500).json({ error: 'Failed to generate upload URL' });
    }
});

// ============================================================================
// MULTIPART UPLOAD - For faster large file uploads (parallel chunk transfers)
// ============================================================================

// Start multipart upload - returns uploadId and key
router.post('/multipart/start', verifyToken, async (req, res) => {
    try {
        const { filename, contentType, folder } = req.body;

        if (!filename || !contentType) {
            return res.status(400).json({ error: 'Filename and contentType required' });
        }

        const validFolders = ['profiles', 'posts', 'stories', 'messages', 'groups', 'ott'];
        const safeFolder = validFolders.includes(folder) ? folder : 'uploads';

        const timestamp = Date.now();
        const safeFilename = filename.replace(/[^a-zA-Z0-9.-]/g, '_');
        const key = `${safeFolder}/${req.userId}/${timestamp}-${safeFilename}`;

        const command = new CreateMultipartUploadCommand({
            Bucket: S3_BUCKET,
            Key: key,
            ContentType: contentType,
            ACL: 'public-read'
        });

        const response = await s3Client.send(command);
        const fileUrl = `https://${CLOUDFRONT_DOMAIN}/${key}`;

        res.json({
            success: true,
            uploadId: response.UploadId,
            key,
            fileUrl
        });
    } catch (err) {
        console.error('Multipart start error:', err);
        res.status(500).json({ error: 'Failed to start multipart upload' });
    }
});

// Get presigned URLs for all parts at once (faster than one at a time)
router.post('/multipart/presign-parts', verifyToken, async (req, res) => {
    try {
        const { key, uploadId, partCount } = req.body;

        if (!key || !uploadId || !partCount) {
            return res.status(400).json({ error: 'key, uploadId, and partCount required' });
        }

        const partUrls = [];

        // Generate presigned URLs for all parts in parallel
        const promises = [];
        for (let partNumber = 1; partNumber <= partCount; partNumber++) {
            const command = new UploadPartCommand({
                Bucket: S3_BUCKET,
                Key: key,
                UploadId: uploadId,
                PartNumber: partNumber
            });
            promises.push(
                getSignedUrl(s3Client, command, { expiresIn: 3600 }).then(url => ({
                    partNumber,
                    url
                }))
            );
        }

        const results = await Promise.all(promises);
        res.json({ success: true, parts: results });
    } catch (err) {
        console.error('Multipart presign error:', err);
        res.status(500).json({ error: 'Failed to generate part URLs' });
    }
});

// Complete multipart upload
router.post('/multipart/complete', verifyToken, async (req, res) => {
    try {
        const { key, uploadId, parts } = req.body;

        if (!key || !uploadId || !parts || !Array.isArray(parts)) {
            return res.status(400).json({ error: 'key, uploadId, and parts array required' });
        }

        // Sort parts by part number
        const sortedParts = parts.sort((a, b) => a.PartNumber - b.PartNumber);

        const command = new CompleteMultipartUploadCommand({
            Bucket: S3_BUCKET,
            Key: key,
            UploadId: uploadId,
            MultipartUpload: {
                Parts: sortedParts.map(p => ({
                    PartNumber: p.PartNumber,
                    ETag: p.ETag
                }))
            }
        });

        await s3Client.send(command);
        const fileUrl = `https://${CLOUDFRONT_DOMAIN}/${key}`;

        res.json({ success: true, fileUrl });
    } catch (err) {
        console.error('Multipart complete error:', err);
        res.status(500).json({ error: 'Failed to complete multipart upload' });
    }
});

// Abort multipart upload (cleanup on error)
router.post('/multipart/abort', verifyToken, async (req, res) => {
    try {
        const { key, uploadId } = req.body;

        if (!key || !uploadId) {
            return res.status(400).json({ error: 'key and uploadId required' });
        }

        const command = new AbortMultipartUploadCommand({
            Bucket: S3_BUCKET,
            Key: key,
            UploadId: uploadId
        });

        await s3Client.send(command);
        res.json({ success: true });
    } catch (err) {
        console.error('Multipart abort error:', err);
        res.status(500).json({ error: 'Failed to abort multipart upload' });
    }
});

module.exports = router;

