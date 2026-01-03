#!/bin/bash
# Deploy BuddyLynk Backend to EC2
# Run this script on the EC2 instance

echo "=== BuddyLynk Backend Deployment ==="

# Install Node.js if not present
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Create app directory
sudo mkdir -p /var/www/buddylynk-api
cd /var/www/buddylynk-api

# Copy files (assuming they're uploaded)
echo "Installing dependencies..."
npm install

# Create .env file (EDIT THESE VALUES!)
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
AWS_REGION=us-east-1
JWT_SECRET=YOUR_SECURE_JWT_SECRET
PORT=3000
S3_BUCKET=buddylynk-media-bucket-2024
EOF

echo "IMPORTANT: Edit /var/www/buddylynk-api/.env with your real secrets!"

# Install PM2 globally
sudo npm install -g pm2

# Start the server
pm2 start server.js --name buddylynk-api

# Save PM2 config for auto-restart
pm2 save
pm2 startup

echo "=== Deployment Complete ==="
echo "Server running on http://52.0.95.126:3000"
echo "Test: curl http://localhost:3000/health"
