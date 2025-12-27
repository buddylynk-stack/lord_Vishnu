/**
 * Security Middleware for Buddylynk API
 * Production-ready input validation and sanitization
 */

// Input validation helper
function sanitizeString(str) {
    if (typeof str !== 'string') return '';
    // Remove any HTML/script tags
    return str.replace(/<[^>]*>/g, '').trim();
}

// Validate email format
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Validate username (alphanumeric, underscore, 3-20 chars)
function isValidUsername(username) {
    const usernameRegex = /^[a-zA-Z0-9_]{3,20}$/;
    return usernameRegex.test(username);
}

// Validate password strength (min 6 chars)
function isValidPassword(password) {
    return password && password.length >= 6;
}

// Request logging middleware
function requestLogger(req, res, next) {
    const timestamp = new Date().toISOString();
    const method = req.method;
    const url = req.originalUrl;
    const userId = req.userId || 'anonymous';
    console.log(`[${timestamp}] ${method} ${url} - User: ${userId}`);
    next();
}

// Validate registration input
function validateRegistration(req, res, next) {
    const { email, username, password } = req.body;

    if (!email || !isValidEmail(email)) {
        return res.status(400).json({ error: 'Invalid email format' });
    }

    if (!username || !isValidUsername(username)) {
        return res.status(400).json({ error: 'Username must be 3-20 alphanumeric characters' });
    }

    if (!password || !isValidPassword(password)) {
        return res.status(400).json({ error: 'Password must be at least 6 characters' });
    }

    // Sanitize inputs
    req.body.email = sanitizeString(email).toLowerCase();
    req.body.username = sanitizeString(username).toLowerCase();
    req.body.fullName = sanitizeString(req.body.fullName || username);

    next();
}

// Validate content (for posts, messages)
function validateContent(req, res, next) {
    if (req.body.content !== undefined) {
        req.body.content = sanitizeString(req.body.content);
        if (req.body.content.length > 5000) {
            return res.status(400).json({ error: 'Content too long (max 5000 chars)' });
        }
    }
    next();
}

// Rate limiting by user (in-memory, consider Redis for production)
const userRequestCounts = new Map();

function perUserRateLimit(maxRequests = 100, windowMs = 60000) {
    return (req, res, next) => {
        const userId = req.userId || req.ip;
        const now = Date.now();

        if (!userRequestCounts.has(userId)) {
            userRequestCounts.set(userId, { count: 1, windowStart: now });
        } else {
            const userData = userRequestCounts.get(userId);

            if (now - userData.windowStart > windowMs) {
                // Reset window
                userData.count = 1;
                userData.windowStart = now;
            } else {
                userData.count++;
            }

            if (userData.count > maxRequests) {
                return res.status(429).json({ error: 'Too many requests. Slow down!' });
            }
        }

        next();
    };
}

module.exports = {
    sanitizeString,
    isValidEmail,
    isValidUsername,
    isValidPassword,
    requestLogger,
    validateRegistration,
    validateContent,
    perUserRateLimit
};
