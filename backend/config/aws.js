const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient } = require('@aws-sdk/lib-dynamodb');
const { S3Client } = require('@aws-sdk/client-s3');

const dynamoClient = new DynamoDBClient({ region: process.env.AWS_REGION || 'us-east-1' });
const docClient = DynamoDBDocumentClient.from(dynamoClient);
const s3Client = new S3Client({ region: process.env.AWS_REGION || 'us-east-1' });

const Tables = {
    USERS: 'Buddylynk_Users',
    POSTS: 'Buddylynk_Posts',
    MESSAGES: 'Buddylynk_Messages',
    FOLLOWS: 'Buddylynk_Follows',
    STORIES: 'Buddylynk_Stories',
    ACTIVITIES: 'Buddylynk_Activities',
    NOTIFICATIONS: 'Buddylynk_Notifications',
    BLOCKS: 'Buddylynk_Blocks',
    REPORTS: 'Buddylynk_Reports',
    EVENTS: 'Buddylynk_Events',
    SAVES: 'Buddylynk_Saves',
    GROUPS: 'Buddylynk_Groups',
    POST_VIEWS: 'Buddylynk_PostViews'
};

const S3_BUCKET = process.env.S3_BUCKET || 'buddylynk-media-bucket-2024';

module.exports = { docClient, s3Client, Tables, S3_BUCKET };
