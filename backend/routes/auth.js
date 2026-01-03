const express = require('express');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const { PutCommand, GetCommand, QueryCommand, ScanCommand } = require('@aws-sdk/lib-dynamodb');
const { docClient, Tables } = require('../config/aws');
const { generateToken } = require('../middleware/auth');

const router = express.Router();

// Avatar color palette based on first letter (A-Z)
// Same colors as Android app for consistency across web/app
const avatarColors = {
    'A': '#E91E63', 'B': '#9C27B0', 'C': '#673AB7', 'D': '#3F51B5',
    'E': '#2196F3', 'F': '#03A9F4', 'G': '#00BCD4', 'H': '#009688',
    'I': '#4CAF50', 'J': '#8BC34A', 'K': '#CDDC39', 'L': '#FFEB3B',
    'M': '#FFC107', 'N': '#FF9800', 'O': '#FF5722', 'P': '#795548',
    'Q': '#607D8B', 'R': '#F44336', 'S': '#9C27B0', 'T': '#3F51B5',
    'U': '#00BCD4', 'V': '#4CAF50', 'W': '#FF9800', 'X': '#E91E63',
    'Y': '#673AB7', 'Z': '#2196F3'
};

function getAvatarColorForName(name) {
    const letter = (name || 'U').charAt(0).toUpperCase();
    return avatarColors[letter] || '#757575';
}

// Register
router.post('/register', async (req, res) => {
    try {
        const { email, username, password, fullName } = req.body;

        if (!email || !username || !password) {
            return res.status(400).json({ error: 'Email, username, and password required' });
        }

        // Check if email exists using Scan (no GSI required)
        const existingUser = await docClient.send(new ScanCommand({
            TableName: Tables.USERS,
            FilterExpression: 'email = :email',
            ExpressionAttributeValues: { ':email': email.toLowerCase() }
        }));

        if (existingUser.Items && existingUser.Items.length > 0) {
            return res.status(409).json({ error: 'Email already registered' });
        }

        // Hash password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Create user
        const userId = uuidv4();
        const now = new Date().toISOString();

        // Get avatar color based on first letter of username
        const avatarColor = getAvatarColorForName(username);

        const user = {
            userId,
            email: email.toLowerCase(),
            username: username.toLowerCase(),
            password: hashedPassword,
            fullName: fullName || username,
            avatar: null,
            avatarColor, // Hex color for default avatar (e.g. "#E91E63")
            bio: '',
            followerCount: 0,
            followingCount: 0,
            postCount: 0,
            createdAt: now,
            updatedAt: now
        };

        await docClient.send(new PutCommand({
            TableName: Tables.USERS,
            Item: user
        }));

        const token = generateToken(userId);

        // Remove password from response
        delete user.password;

        res.status(201).json({ user, token });

    } catch (err) {
        console.error('Register error:', err);
        res.status(500).json({ error: 'Registration failed' });
    }
});

// Login
router.post('/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        if (!email || !password) {
            return res.status(400).json({ error: 'Email and password required' });
        }

        // Find user by email using Scan (no GSI required)
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.USERS,
            FilterExpression: 'email = :email',
            ExpressionAttributeValues: { ':email': email.toLowerCase() }
        }));

        if (!result.Items || result.Items.length === 0) {
            return res.status(401).json({ error: 'Invalid email or password' });
        }

        const user = result.Items[0];

        if (!user.password) {
            return res.status(401).json({ error: 'This account uses Google Sign-In. Please login with Google.' });
        }

        // Verify password with bcrypt
        const validPassword = await bcrypt.compare(password, user.password);
        if (!validPassword) {
            return res.status(401).json({ error: 'Invalid email or password' });
        }

        const token = generateToken(user.userId);

        // Remove password from response
        delete user.password;

        res.json({ user, token });

    } catch (err) {
        res.status(500).json({ error: 'Login failed' });
    }
});

// Google Auth - create or find user by email and return JWT
router.post('/google', async (req, res) => {
    try {
        const { email, displayName, avatar } = req.body;

        if (!email) {
            return res.status(400).json({ error: 'Email required' });
        }

        // Find user by email using Scan (no GSI required)
        const result = await docClient.send(new ScanCommand({
            TableName: Tables.USERS,
            FilterExpression: 'email = :email',
            ExpressionAttributeValues: { ':email': email.toLowerCase() }
        }));

        let user;
        let isNewUser = false;

        if (result.Items && result.Items.length > 0) {
            // User exists
            user = result.Items[0];

            // Update avatar if changed
            if (avatar && user.avatar !== avatar) {
                user.avatar = avatar;
                await docClient.send(new PutCommand({
                    TableName: Tables.USERS,
                    Item: user
                }));
            }
        } else {
            // Create new user
            const userId = uuidv4();
            const now = new Date().toISOString();

            user = {
                userId,
                email: email.toLowerCase(),
                username: displayName?.toLowerCase() || email.split('@')[0],
                fullName: displayName || email.split('@')[0],
                avatar: avatar || null,
                bio: '',
                followerCount: 0,
                followingCount: 0,
                postCount: 0,
                createdAt: now,
                updatedAt: now
            };

            await docClient.send(new PutCommand({
                TableName: Tables.USERS,
                Item: user
            }));

            isNewUser = true;
        }

        const token = generateToken(user.userId);

        // Remove password from response if exists
        delete user.password;

        res.json({ user, token, isNewUser });

    } catch (err) {
        console.error('Google auth error:', err);
        res.status(500).json({ error: 'Google authentication failed' });
    }
});

module.exports = router;
