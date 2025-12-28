const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, ScanCommand, UpdateCommand } = require('@aws-sdk/lib-dynamodb');
const client = DynamoDBDocumentClient.from(new DynamoDBClient({ region: 'us-east-1' }));

async function removeNsfwColumn() {
    console.log('Scanning for posts with isNsfw attribute...');
    const result = await client.send(new ScanCommand({
        TableName: 'Buddylynk_Posts',
        ProjectionExpression: 'postId, isNsfw',
        FilterExpression: 'attribute_exists(isNsfw)'
    }));

    console.log('Found', result.Items?.length || 0, 'posts with isNsfw');

    for (const item of result.Items || []) {
        await client.send(new UpdateCommand({
            TableName: 'Buddylynk_Posts',
            Key: { postId: item.postId },
            UpdateExpression: 'REMOVE isNsfw'
        }));
        console.log('Removed isNsfw from', item.postId.substring(0, 8));
    }

    console.log('DONE - isNsfw column removed from all posts');
}

removeNsfwColumn().catch(console.error);
