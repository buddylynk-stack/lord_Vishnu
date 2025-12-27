// CloudFront CDN configuration for faster media delivery
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

// Convert all media URLs in a post object
function convertPostMediaUrls(post) {
    if (!post) return post;

    if (post.mediaUrl) post.mediaUrl = toCloudFrontUrl(post.mediaUrl);
    if (post.userAvatar) post.userAvatar = toCloudFrontUrl(post.userAvatar);

    if (post.media && Array.isArray(post.media)) {
        post.media = post.media.map(m => ({
            ...m,
            url: toCloudFrontUrl(m.url)
        }));
    }

    if (post.mediaUrls && Array.isArray(post.mediaUrls)) {
        post.mediaUrls = post.mediaUrls.map(url => toCloudFrontUrl(url));
    }

    return post;
}

// Convert all media URLs in a user object
function convertUserMediaUrls(user) {
    if (!user) return user;

    if (user.avatar) user.avatar = toCloudFrontUrl(user.avatar);
    if (user.profileImage) user.profileImage = toCloudFrontUrl(user.profileImage);
    if (user.coverImage) user.coverImage = toCloudFrontUrl(user.coverImage);
    if (user.imageUrl) user.imageUrl = toCloudFrontUrl(user.imageUrl);

    return user;
}

// Convert all media URLs in a message object
function convertMessageMediaUrls(message) {
    if (!message) return message;

    if (message.mediaUrl) message.mediaUrl = toCloudFrontUrl(message.mediaUrl);
    if (message.imageUrl) message.imageUrl = toCloudFrontUrl(message.imageUrl);
    if (message.attachmentUrl) message.attachmentUrl = toCloudFrontUrl(message.attachmentUrl);

    if (message.media && Array.isArray(message.media)) {
        message.media = message.media.map(m => ({
            ...m,
            url: toCloudFrontUrl(m.url)
        }));
    }

    return message;
}

// Convert all media URLs in a group object
function convertGroupMediaUrls(group) {
    if (!group) return group;

    if (group.imageUrl) group.imageUrl = toCloudFrontUrl(group.imageUrl);
    if (group.coverImage) group.coverImage = toCloudFrontUrl(group.coverImage);
    if (group.avatar) group.avatar = toCloudFrontUrl(group.avatar);

    return group;
}

// Convert all media URLs in a story object
function convertStoryMediaUrls(story) {
    if (!story) return story;

    if (story.mediaUrl) story.mediaUrl = toCloudFrontUrl(story.mediaUrl);
    if (story.imageUrl) story.imageUrl = toCloudFrontUrl(story.imageUrl);
    if (story.userAvatar) story.userAvatar = toCloudFrontUrl(story.userAvatar);

    return story;
}

module.exports = {
    toCloudFrontUrl,
    convertPostMediaUrls,
    convertUserMediaUrls,
    convertMessageMediaUrls,
    convertGroupMediaUrls,
    convertStoryMediaUrls,
    CLOUDFRONT_DOMAIN
};
