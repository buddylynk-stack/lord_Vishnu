const jwt = require('jsonwebtoken');

// CRITICAL: JWT_SECRET must be set in environment variables!
// Generate a strong secret: node -e "console.log(require('crypto').randomBytes(64).toString('hex'))"
const JWT_SECRET = process.env.JWT_SECRET;

if (!JWT_SECRET || JWT_SECRET === 'change-this-secret' || JWT_SECRET.length < 32) {
    console.error('SECURITY ERROR: JWT_SECRET is not properly configured!');
    console.error('Set a strong JWT_SECRET in your environment variables (minimum 32 characters)');
    process.exit(1);
}

function generateToken(userId) {
    // SECURITY: Shorter token expiry (7 days instead of 30)
    return jwt.sign({ userId }, JWT_SECRET, { expiresIn: '7d' });
}

function verifyToken(req, res, next) {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'No token provided' });
    }

    const token = authHeader.split(' ')[1];
    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        req.userId = decoded.userId;
        next();
    } catch (err) {
        return res.status(401).json({ error: 'Invalid token' });
    }
}

module.exports = { generateToken, verifyToken };
