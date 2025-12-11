# Deployment Guide

## Cloud Deployment Options

### 1. AWS Lambda (Serverless)

**Advantages:**
- Pay only for execution time
- Auto-scaling
- No server management
- Perfect for hourly/daily predictions

**Setup:**

```bash
# Install serverless
npm install -g serverless

# Create serverless configuration
serverless create --template aws-python-torch

# Deploy
serverless deploy
```

### 2. AWS EC2 (Traditional VM)

**Setup:**

```bash
# Launch Ubuntu EC2 instance
# SSH into instance

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv git

# Clone and setup
git clone <repo-url>
cd crypto-price-predictor
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run as daemon
nohup python main.py > logs/output.log 2>&1 &
```

### 3. Google Cloud Run (Containerized)

**Setup:**

```bash
# Build and push Docker image
docker build -t crypto-predictor:latest .

# Authenticate with GCP
gcloud auth configure-docker

# Push to Container Registry
docker tag crypto-predictor:latest gcr.io/<project-id>/crypto-predictor:latest
docker push gcr.io/<project-id>/crypto-predictor:latest

# Deploy to Cloud Run
gcloud run deploy crypto-predictor \
  --image gcr.io/<project-id>/crypto-predictor:latest \
  --platform managed \
  --region us-central1 \
  --set-env-vars DISCORD_BOT_TOKEN=<token>
```

### 4. Docker Swarm

**Setup:**

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml crypto-predictor

# Monitor
docker service ls
docker service logs crypto-predictor_crypto-predictor
```

### 5. Kubernetes

**Create deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crypto-predictor
spec:
  replicas: 1
  selector:
    matchLabels:
      app: crypto-predictor
  template:
    metadata:
      labels:
        app: crypto-predictor
    spec:
      containers:
      - name: crypto-predictor
        image: crypto-predictor:latest
        env:
        - name: DISCORD_BOT_TOKEN
          valueFrom:
            secretKeyRef:
              name: crypto-secret
              key: discord-token
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

**Deploy:**

```bash
kubectl apply -f deployment.yaml
```

## VPS Deployment (DigitalOcean, Linode, etc.)

### 1. Initial Setup

```bash
# SSH into server
ssh root@<ip>

# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Add user to docker group
usermod -aG docker $USER
```

### 2. Deploy with Docker Compose

```bash
# Clone repository
git clone <repo-url>
cd crypto-price-predictor

# Create .env with credentials
echo "DISCORD_BOT_TOKEN=<token>" > .env
echo "DISCORD_CHANNEL_ID=<id>" >> .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 3. Setup SSL/HTTPS (Optional)

```bash
# Install Certbot
apt-get install certbot

# Get certificate
certbot certonly --standalone -d your-domain.com
```

## Monitoring and Maintenance

### Uptime Monitoring

```bash
# Use UptimeRobot or similar service
# Configure health check endpoint
```

### Log Monitoring

```bash
# Real-time logs
docker-compose logs -f crypto-predictor

# Log rotation
# Add to logrotate
```

### Auto-restart

```bash
# Docker restart policy
# In docker-compose.yml:
restart_policy:
  condition: on-failure
  delay: 5s
  max_attempts: 5
  window: 120s
```

## Performance Optimization

### 1. Database Caching

```bash
# Enable Redis for caching
docker-compose up -d redis

# Configure in config.yaml
redis_url: redis://localhost:6379
```

### 2. GPU Support

```bash
# Docker GPU support
docker run --gpus all crypto-predictor
```

### 3. Load Balancing

Use Nginx for multiple instances:

```nginx
upstream crypto_predictor {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://crypto_predictor;
    }
}
```

## Backup and Recovery

### Backup Strategy

```bash
# Daily backup
0 2 * * * docker-compose exec -T redis redis-cli SAVE > /backups/redis_$(date +\%Y\%m\%d).rdb

# Model backup
0 3 * * * tar -czf /backups/models_$(date +\%Y\%m\%d).tar.gz models/
```

### Disaster Recovery

```bash
# Restore from backup
tar -xzf /backups/models_20240101.tar.gz

# Update docker containers
docker-compose down
docker-compose up -d
```

## Security

### 1. Environment Variables

```bash
# Never commit .env file
echo ".env" >> .gitignore

# Use secrets management
echo "discord_token" | docker secret create discord_token -
```

### 2. Firewall

```bash
# UFW rules
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

### 3. API Keys

- Rotate keys regularly
- Use different keys for different environments
- Monitor API usage

## Troubleshooting

### Container won't start

```bash
docker-compose logs crypto-predictor
# Check for configuration errors
```

### High memory usage

```bash
# Monitor resources
docker stats

# Adjust memory limits in docker-compose.yml
```

### Slow predictions

```bash
# Check GPU usage
gpustat

# Reduce batch size
# Optimize model
```
