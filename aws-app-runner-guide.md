# AWS App Runner Deployment Guide

## Deploy to AWS App Runner (Recommended)

AWS App Runner is the easiest way to deploy your Flask app to AWS. It automatically builds and deploys your containerized application.

### Prerequisites
- AWS Account
- AWS CLI installed and configured
- Docker installed locally

### Step 1: Prepare Your Repository
Ensure your repository is on GitHub and contains:
- `Dockerfile`
- `requirements.txt`
- `app.py`
- `train.py`

### Step 2: Deploy via AWS Console

1. **Go to AWS App Runner Console**
   - Navigate to AWS App Runner in your AWS Console
   - Click "Create service"

2. **Source Configuration**
   - Choose "Source code repository"
   - Connect your GitHub account
   - Select your repository
   - Choose the branch (usually `main`)

3. **Build Configuration**
   - Build command: Leave empty (uses Dockerfile)
   - Port: `5000`
   - Runtime: `Docker`

4. **Service Configuration**
   - Service name: `breast-cancer-prediction-api`
   - CPU: `1 vCPU`
   - Memory: `2 GB`
   - Environment variables (optional):
     ```
     FLASK_ENV=production
     PORT=5000
     ```

5. **Create Service**
   - Click "Create & deploy"
   - Wait for deployment (5-10 minutes)

### Step 3: Access Your API
- App Runner will provide a URL like: `https://abc123.us-east-1.awsapprunner.com`
- Your API endpoints will be available at:
  - `https://abc123.us-east-1.awsapprunner.com/` (Web interface)
  - `https://abc123.us-east-1.awsapprunner.com/predict` (API endpoint)
  - `https://abc123.us-east-1.awsapprunner.com/health` (Health check)

### Step 4: Test Your Deployment
```bash
# Test the health endpoint
curl https://abc123.us-east-1.awsapprunner.com/health

# Test prediction
curl -X POST https://abc123.us-east-1.awsapprunner.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 15.0,
    "mean_texture": 20.0,
    "mean_smoothness": 0.1,
    "model_type": "gaussian"
  }'
```

## Advanced Configuration

### Custom Domain (Optional)
1. Go to your App Runner service
2. Click "Custom domains"
3. Add your domain and configure DNS

### Environment Variables
You can add environment variables in the App Runner console:
- `FLASK_ENV=production`
- `PORT=5000`
- `MODEL_PATH=/app/models`

### Auto Scaling
App Runner automatically scales based on traffic, but you can configure:
- Min instances: 1
- Max instances: 10
- Concurrency: 50 requests per instance

## Cost Estimation
- **Free tier**: 750 hours/month for first 12 months
- **After free tier**: ~$0.064/hour for 1 vCPU, 2 GB memory
- **Estimated monthly cost**: ~$45-50 for 24/7 operation

## Troubleshooting

### Common Issues
1. **Build fails**: Check Dockerfile and requirements.txt
2. **Port issues**: Ensure app listens on port 5000
3. **Model loading**: Ensure models are trained during build

### Logs
- View logs in App Runner console
- Use CloudWatch for detailed monitoring

### Health Checks
Your app includes a health check endpoint at `/health` that App Runner can use. 