# AWS Deployment Guide for Breast Cancer Prediction API

## Overview

This guide covers multiple ways to deploy your Flask API to AWS. Choose the option that best fits your needs:

1. **AWS App Runner** (Recommended) - Easiest, fully managed
2. **AWS ECS** - Container orchestration, more control
3. **AWS Elastic Beanstalk** - Platform as a Service, good for beginners

## Prerequisites

### Required Tools
- AWS Account
- AWS CLI installed and configured
- Docker (for containerized deployments)
- Git repository with your code

### AWS CLI Setup
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS CLI
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-east-1)
# Enter your output format (json)
```

## Option 1: AWS App Runner (Recommended)

### Why App Runner?
- **Easiest deployment** - Just connect your GitHub repo
- **Fully managed** - No server management
- **Auto-scaling** - Handles traffic automatically
- **HTTPS included** - SSL certificates managed
- **Cost-effective** - Pay only for what you use

### Quick Deployment Steps

1. **Prepare Your Repository**
   ```bash
   # Ensure these files are in your repo:
   # - Dockerfile
   # - requirements.txt
   # - app.py
   # - train.py
   ```

2. **Deploy via AWS Console**
   ```bash
   # Use the deployment script
   ./deploy-aws.sh app-runner
   ```

3. **Manual Steps in AWS Console**
   - Go to AWS App Runner Console
   - Click "Create service"
   - Choose "Source code repository"
   - Connect GitHub and select your repo
   - Configure:
     - Build command: (leave empty)
     - Port: `5000`
     - Runtime: `Docker`
   - Service configuration:
     - CPU: `1 vCPU`
     - Memory: `2 GB`
   - Click "Create & deploy"

4. **Access Your API**
   - App Runner provides a URL like: `https://abc123.us-east-1.awsapprunner.com`
   - Test endpoints:
     ```bash
     # Health check
     curl https://abc123.us-east-1.awsapprunner.com/health
     
     # Make prediction
     curl -X POST https://abc123.us-east-1.awsapprunner.com/predict \
       -H "Content-Type: application/json" \
       -d '{
         "mean_radius": 15.0,
         "mean_texture": 20.0,
         "mean_smoothness": 0.1,
         "model_type": "gaussian"
       }'
     ```

### Cost Estimation
- **Free tier**: 750 hours/month for first 12 months
- **After free tier**: ~$0.064/hour for 1 vCPU, 2 GB memory
- **Monthly cost**: ~$45-50 for 24/7 operation

## Option 2: AWS ECS (Elastic Container Service)

### Why ECS?
- **Container orchestration** - Manage multiple containers
- **High availability** - Multi-AZ deployment
- **Load balancing** - Distribute traffic
- **Service discovery** - Internal communication
- **More complex** - Requires more AWS knowledge

### Deployment Steps

1. **Setup AWS Resources**
   ```bash
   ./deploy-aws.sh setup-aws
   ```

2. **Build and Push to ECR**
   ```bash
   ./deploy-aws.sh build-ecr
   ```

3. **Deploy to ECS**
   ```bash
   ./deploy-aws.sh ecs
   ```

4. **Create ECS Service** (Manual step)
   ```bash
   # You'll need to create a service in the ECS console
   # or use the AWS CLI with proper VPC configuration
   aws ecs create-service \
     --cluster breast-cancer-cluster \
     --service-name breast-cancer-service \
     --task-definition breast-cancer-api \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}
   ```

### Cost Estimation
- **Fargate**: ~$0.04048 per vCPU per hour + $0.004445 per GB per hour
- **Load Balancer**: ~$16/month
- **Monthly cost**: ~$50-70 for 24/7 operation

## Option 3: AWS Elastic Beanstalk

### Why Elastic Beanstalk?
- **Platform as a Service** - AWS manages the platform
- **Easy deployment** - Simple CLI commands
- **Auto-scaling** - Built-in scaling policies
- **Health monitoring** - Automatic health checks
- **Less control** - Platform limitations

### Deployment Steps

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Deploy**
   ```bash
   ./deploy-aws.sh eb
   ```

3. **Manual Steps**
   ```bash
   # Initialize EB application
   eb init breast-cancer-api --platform python-3.9 --region us-east-1
   
   # Create environment
   eb create breast-cancer-api-env --instance-type t3.small --single-instance
   
   # Deploy
   eb deploy
   ```

### Cost Estimation
- **t3.small instance**: ~$15/month
- **Load Balancer**: ~$16/month
- **Monthly cost**: ~$30-40 for 24/7 operation

## Advanced Configuration

### Environment Variables
All deployment options support environment variables:

```bash
FLASK_ENV=production
PORT=5000
MODEL_PATH=/app/models
```

### Custom Domain
1. **App Runner**: Use the Custom domains section in console
2. **ECS**: Use Application Load Balancer with Route 53
3. **EB**: Use the Custom domains section in console

### Monitoring and Logging
- **CloudWatch**: All options integrate with CloudWatch
- **Health Checks**: Your app includes `/health` endpoint
- **Logs**: View logs in respective AWS consoles

## Troubleshooting

### Common Issues

1. **Build Fails**
   ```bash
   # Check Dockerfile
   docker build -t test-image .
   
   # Check requirements.txt
   pip install -r requirements.txt
   ```

2. **Port Issues**
   - Ensure app listens on port 5000
   - Check security groups allow port 5000

3. **Model Loading**
   - Ensure models are trained during build
   - Check model file paths

4. **Memory Issues**
   - Increase memory allocation
   - Optimize Docker image size

### Debugging Commands

```bash
# Check AWS CLI configuration
aws sts get-caller-identity

# Check ECR login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin

# Check ECS cluster
aws ecs describe-clusters --clusters breast-cancer-cluster

# Check EB status
eb status
```

## Performance Optimization

### Docker Optimization
```dockerfile
# Use multi-stage builds
FROM python:3.9-slim as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
# ... rest of Dockerfile
```

### Application Optimization
- Use gunicorn for production
- Implement caching for model predictions
- Add request rate limiting

## Security Best Practices

1. **IAM Roles**: Use least privilege principle
2. **Security Groups**: Restrict access to necessary ports
3. **Secrets Management**: Use AWS Secrets Manager for sensitive data
4. **HTTPS**: All options provide HTTPS by default
5. **VPC**: Use private subnets for ECS deployments

## Scaling Strategies

### App Runner
- Automatic scaling based on CPU/memory
- Configure min/max instances

### ECS
- Use Application Auto Scaling
- Configure target tracking policies

### Elastic Beanstalk
- Configure auto-scaling groups
- Set scaling triggers

## Cost Optimization

1. **Use Spot Instances** (ECS/EB)
2. **Right-size instances**
3. **Use reserved instances** for long-term deployments
4. **Monitor with CloudWatch**
5. **Set up billing alerts**

## Next Steps

1. **Choose your deployment option**
2. **Follow the deployment steps**
3. **Test your API endpoints**
4. **Set up monitoring and alerts**
5. **Configure custom domain (optional)**
6. **Set up CI/CD pipeline (optional)**

Your breast cancer prediction API will be production-ready and scalable! 