require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const http = require('http');
const WebSocket = require('ws');
const { Server } = require('socket.io');
const path = require('path');

const authRoutes = require('./routes/auth');
const userRoutes = require('./routes/users');
const postRoutes = require('./routes/posts');
const uploadRoutes = require('./routes/upload');
const storyRoutes = require('./routes/stories');
const followRoutes = require('./routes/follows');
const messageRoutes = require('./routes/messages');
const groupRoutes = require('./routes/groups');
const blockRoutes = require('./routes/blocks');
const nsfwRoutes = require('./routes/nsfw');
const behaviorRoutes = require('./routes/behavior');
const { initializeSignaling } = require('./services/signaling');

const app = express();
const server = http.createServer(app);

// Serve assetlinks.json for Android App Links verification
app.use('/.well-known', express.static(path.join(__dirname, '.well-known')));

// Socket.io for WebRTC call signaling - SECURITY: Restrict CORS
const allowedOrigins = process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : [];
const io = new Server(server, {
    cors: { origin: allowedOrigins.length > 0 ? allowedOrigins : false, methods: ['GET', 'POST'] },
    path: '/socket.io'
});
initializeSignaling(io);

// WebSocket for real-time chat (keeping for backwards compatibility)
const wss = new WebSocket.Server({ server, path: '/ws' });
const clients = new Map();

wss.on('connection', (ws) => {
    ws.on('message', (msg) => {
        try {
            const data = JSON.parse(msg);
            if (data.type === 'login' && data.userId) {
                ws.userId = data.userId;
                clients.set(data.userId, ws);
                ws.send(JSON.stringify({ type: 'login_success' }));
            } else if (data.targetUserId) {
                const target = clients.get(data.targetUserId);
                if (target && target.readyState === WebSocket.OPEN) {
                    target.send(JSON.stringify({ ...data, senderId: ws.userId }));
                }
            }
        } catch (e) { }
    });
    ws.on('close', () => {
        if (ws.userId) clients.delete(ws.userId);
    });
});

// Security
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            imgSrc: ["'self'", "data:", "https:"],
            scriptSrc: ["'self'"]
        }
    },
    hsts: {
        maxAge: 31536000,
        includeSubDomains: true
    }
}));
app.use(cors({
    origin: process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : '*',
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json({ limit: '10mb' }));

// Security middleware - request logging
const { requestLogger } = require('./middleware/security');
app.use(requestLogger);

// Rate limiting (global) - stricter limits
const globalLimiter = rateLimit({
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
    max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 500,
    message: { error: 'Too many requests, please try again later' },
    standardHeaders: true,
    legacyHeaders: false
});
app.use(globalLimiter);

// Stricter rate limiting for auth endpoints
const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // 10 attempts per window
    message: { error: 'Too many login attempts, please try again later' }
});

// Routes
app.use('/api/auth', authLimiter, authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/posts', postRoutes);
app.use('/api/upload', uploadRoutes);
app.use('/api/stories', storyRoutes);
app.use('/api/follows', followRoutes);
app.use('/api/messages', messageRoutes);
app.use('/api/groups', groupRoutes);
app.use('/api/blocks', blockRoutes);
app.use('/api/nsfw', nsfwRoutes); // Admin NSFW content flagging
app.use('/api/behavior', behaviorRoutes); // MindFlow algorithm behavior tracking

// Deep Link routes - redirect to app or show landing page
app.get('/post/:postId', (req, res) => {
    const postId = req.params.postId;
    const userAgent = req.headers['user-agent'] || '';
    const isAndroid = userAgent.toLowerCase().includes('android');

    if (isAndroid) {
        // Try to open app with intent fallback
        const intentUrl = `intent://post/${postId}#Intent;scheme=buddylynk;package=com.orignal.buddylynk;S.browser_fallback_url=https://play.google.com/store/apps/details?id=com.orignal.buddylynk;end`;
        res.redirect(intentUrl);
    } else {
        // Show landing page for other devices
        res.send(`
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>BuddyLynk - View Post</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; color: white; }
                    .container { text-align: center; padding: 40px; }
                    h1 { font-size: 2.5rem; margin-bottom: 20px; background: linear-gradient(90deg, #8B5CF6, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
                    p { color: #a0aec0; margin-bottom: 30px; }
                    .btn { display: inline-block; padding: 15px 40px; background: linear-gradient(90deg, #8B5CF6, #EC4899); color: white; text-decoration: none; border-radius: 30px; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>BuddyLynk</h1>
                    <p>Download the app to view this post</p>
                    <a class="btn" href="https://play.google.com/store/apps/details?id=com.orignal.buddylynk">Get the App</a>
                </div>
            </body>
            </html>
        `);
    }
});

app.get('/user/:userId', (req, res) => {
    const userId = req.params.userId;
    const userAgent = req.headers['user-agent'] || '';
    const isAndroid = userAgent.toLowerCase().includes('android');

    if (isAndroid) {
        const intentUrl = `intent://user/${userId}#Intent;scheme=buddylynk;package=com.orignal.buddylynk;S.browser_fallback_url=https://play.google.com/store/apps/details?id=com.orignal.buddylynk;end`;
        res.redirect(intentUrl);
    } else {
        res.send(`
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>BuddyLynk - View Profile</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; color: white; }
                    .container { text-align: center; padding: 40px; }
                    h1 { font-size: 2.5rem; margin-bottom: 20px; background: linear-gradient(90deg, #8B5CF6, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
                    p { color: #a0aec0; margin-bottom: 30px; }
                    .btn { display: inline-block; padding: 15px 40px; background: linear-gradient(90deg, #8B5CF6, #EC4899); color: white; text-decoration: none; border-radius: 30px; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>BuddyLynk</h1>
                    <p>Download the app to view this profile</p>
                    <a class="btn" href="https://play.google.com/store/apps/details?id=com.orignal.buddylynk">Get the App</a>
                </div>
            </body>
            </html>
        `);
    }
});

app.get('/health', (req, res) => res.json({ status: 'ok' }));

const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Socket.io enabled for WebRTC signaling`);
});
