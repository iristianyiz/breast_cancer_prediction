#!/bin/bash

# AWS Deployment Script for Breast Cancer Prediction API
# Supports App Runner, ECS, and Elastic Beanstalk

set -e

echo "AWS Deployment Script for Breast Cancer Prediction API"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo -e "${BLUE}Usage: $0 [OPTION]${NC}"
    echo "Options:"
    echo "  app-runner    - Deploy to AWS App Runner (Recommended)"
    echo "  ecs          - Deploy to AWS ECS"
    echo "  eb           - Deploy to AWS Elastic Beanstalk"
    echo "  build-ecr    - Build and push to ECR"
    echo "  setup-aws    - Setup AWS resources (IAM, ECR, etc.)"
    echo "  help         - Show this help message"
}

# Function to check AWS CLI
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}❌ AWS CLI is not installed.${NC}"
        echo "Please install AWS CLI: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}❌ AWS credentials not configured.${NC}"
        echo "Please run 'aws configure' first."
        exit 1
    fi
    
    echo -e "${GREEN}AWS CLI configured${NC}"
}

# Function to get AWS account info
get_aws_info() {
    export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    export AWS_REGION=$(aws configure get region)
    echo -e "${BLUE}AWS Account: ${AWS_ACCOUNT_ID}${NC}"
    echo -e "${BLUE}AWS Region: ${AWS_REGION}${NC}"
}

# Function to setup AWS resources
setup_aws() {
    echo -e "${YELLOW}Setting up AWS resources...${NC}"
    
    # Create ECR repository
    echo "Creating ECR repository..."
    aws ecr create-repository --repository-name breast-cancer-api --region $AWS_REGION || true
    
    # Create CloudWatch log group
    echo "Creating CloudWatch log group..."
    aws logs create-log-group --log-group-name /ecs/breast-cancer-api --region $AWS_REGION || true
    
    # Create IAM roles (if they don't exist)
    echo "Setting up IAM roles..."
    
    # ECS Task Execution Role
    aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }' || true
    
    aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy || true
    
    # ECS Task Role
    aws iam create-role --role-name ecsTaskRole --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }' || true
    
    echo -e "${GREEN}AWS resources setup complete${NC}"
}

# Function to build and push to ECR
build_ecr() {
    echo -e "${YELLOW}Building and pushing to ECR...${NC}"
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    # Build image
    docker build -t breast-cancer-api .
    
    # Tag image
    docker tag breast-cancer-api:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/breast-cancer-api:latest
    
    # Push image
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/breast-cancer-api:latest
    
    echo -e "${GREEN}Image pushed to ECR${NC}"
}

# Function to deploy to App Runner
deploy_app_runner() {
    echo -e "${YELLOW}Deploying to AWS App Runner...${NC}"
    
    echo -e "${BLUE}Please follow these steps in the AWS Console:${NC}"
    echo "1. Go to AWS App Runner Console"
    echo "2. Click 'Create service'"
    echo "3. Choose 'Source code repository'"
    echo "4. Connect your GitHub account"
    echo "5. Select this repository"
    echo "6. Configure:"
    echo "   - Build command: (leave empty)"
    echo "   - Port: 5000"
    echo "   - Runtime: Docker"
    echo "7. Service configuration:"
    echo "   - CPU: 1 vCPU"
    echo "   - Memory: 2 GB"
    echo "8. Click 'Create & deploy'"
    
    echo -e "${GREEN}App Runner deployment instructions provided${NC}"
    echo -e "${BLUE}Your app will be available at the URL provided by App Runner${NC}"
}

# Function to deploy to ECS
deploy_ecs() {
    echo -e "${YELLOW}Deploying to AWS ECS...${NC}"
    
    # Build and push to ECR first
    build_ecr
    
    # Update task definition with correct account ID and region
    sed "s/YOUR_ACCOUNT_ID/$AWS_ACCOUNT_ID/g; s/YOUR_REGION/$AWS_REGION/g" aws-ecs-task-definition.json > task-definition-updated.json
    
    # Register task definition
    aws ecs register-task-definition --cli-input-json file://task-definition-updated.json --region $AWS_REGION
    
    # Create ECS cluster
    aws ecs create-cluster --cluster-name breast-cancer-cluster --region $AWS_REGION || true
    
    # Create service (requires VPC and security group setup)
    echo -e "${YELLOW}⚠️  ECS service creation requires VPC setup${NC}"
    echo "Please create the service manually in the ECS console or use:"
    echo "aws ecs create-service --cluster breast-cancer-cluster --service-name breast-cancer-service --task-definition breast-cancer-api --desired-count 1 --launch-type FARGATE --network-configuration awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
    
    echo -e "${GREEN}ECS task definition registered${NC}"
}

# Function to deploy to Elastic Beanstalk
deploy_eb() {
    echo -e "${YELLOW}Deploying to AWS Elastic Beanstalk...${NC}"
    
    # Check if EB CLI is installed
    if ! command -v eb &> /dev/null; then
        echo -e "${RED}❌ Elastic Beanstalk CLI is not installed.${NC}"
        echo "Please install it: pip install awsebcli"
        exit 1
    fi
    
    # Initialize EB application
    eb init breast-cancer-api --platform python-3.9 --region $AWS_REGION || true
    
    # Create environment
    eb create breast-cancer-api-env --instance-type t3.small --single-instance || true
    
    # Deploy
    eb deploy
    
    echo -e "${GREEN}Elastic Beanstalk deployment complete${NC}"
    echo -e "${BLUE}Your app is available at: $(eb status | grep CNAME | awk '{print $2}')${NC}"
}

# Function to run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    python -m pytest tests/ -v
    echo -e "${GREEN}Tests passed!${NC}"
}

# Main script logic
case "${1:-help}" in
    "app-runner")
        check_aws_cli
        get_aws_info
        run_tests
        deploy_app_runner
        ;;
    "ecs")
        check_aws_cli
        get_aws_info
        setup_aws
        run_tests
        deploy_ecs
        ;;
    "eb")
        check_aws_cli
        get_aws_info
        run_tests
        deploy_eb
        ;;
    "build-ecr")
        check_aws_cli
        get_aws_info
        build_ecr
        ;;
    "setup-aws")
        check_aws_cli
        get_aws_info
        setup_aws
        ;;
    "help"|*)
        usage
        ;;
esac 