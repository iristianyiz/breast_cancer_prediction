#!/bin/bash

# Deployment script for Breast Cancer Prediction API
# Supports Heroku, AWS, and local Docker deployment

set -e

echo "🚀 Breast Cancer Prediction API Deployment Script"
echo "=================================================="

# Function to display usage
usage() {
    echo "Usage: $0 [OPTION]"
    echo "Options:"
    echo "  local     - Deploy locally using Docker"
    echo "  heroku    - Deploy to Heroku"
    echo "  aws       - Deploy to AWS (requires AWS CLI setup)"
    echo "  build     - Build Docker image only"
    echo "  test      - Run tests before deployment"
    echo "  help      - Show this help message"
}

# Function to run tests
run_tests() {
    echo "🧪 Running tests..."
    python -m pytest tests/ -v
    echo "✅ Tests passed!"
}

# Function to build Docker image
build_docker() {
    echo "🐳 Building Docker image..."
    docker build -t breast-cancer-api .
    echo "✅ Docker image built successfully!"
}

# Function to deploy locally
deploy_local() {
    echo "🏠 Deploying locally..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Build and run with docker-compose
    docker-compose up --build -d
    
    echo "✅ Local deployment successful!"
    echo "🌐 API is running at: http://localhost:5000"
    echo "📊 Health check: http://localhost:5000/health"
}

# Function to deploy to Heroku
deploy_heroku() {
    echo "☁️  Deploying to Heroku..."
    
    # Check if Heroku CLI is installed
    if ! command -v heroku &> /dev/null; then
        echo "❌ Heroku CLI is not installed. Please install it first."
        echo "   Visit: https://devcenter.heroku.com/articles/heroku-cli"
        exit 1
    fi
    
    # Check if logged in to Heroku
    if ! heroku auth:whoami &> /dev/null; then
        echo "❌ Not logged in to Heroku. Please run 'heroku login' first."
        exit 1
    fi
    
    # Create Heroku app if it doesn't exist
    if ! heroku apps:info &> /dev/null; then
        echo "📱 Creating new Heroku app..."
        heroku create
    fi
    
    # Deploy to Heroku
    echo "🚀 Deploying to Heroku..."
    git add .
    git commit -m "Deploy to Heroku" || true
    git push heroku main
    
    echo "✅ Heroku deployment successful!"
    echo "🌐 Your app is running at: $(heroku info -s | grep web_url | cut -d= -f2)"
}

# Function to deploy to AWS
deploy_aws() {
    echo "☁️  Deploying to AWS..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        echo "❌ AWS CLI is not installed. Please install it first."
        echo "   Visit: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    # Check if AWS credentials are configured
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "❌ AWS credentials not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    echo "⚠️  AWS deployment requires manual setup."
    echo "   Please follow these steps:"
    echo "   1. Create an ECS cluster"
    echo "   2. Create a task definition"
    echo "   3. Create a service"
    echo "   4. Or use AWS App Runner for easier deployment"
    echo ""
    echo "   For App Runner deployment:"
    echo "   1. Go to AWS App Runner console"
    echo "   2. Create new service"
    echo "   3. Connect your GitHub repository"
    echo "   4. Use the Dockerfile in this project"
}

# Main script logic
case "${1:-help}" in
    "local")
        run_tests
        deploy_local
        ;;
    "heroku")
        run_tests
        deploy_heroku
        ;;
    "aws")
        run_tests
        deploy_aws
        ;;
    "build")
        build_docker
        ;;
    "test")
        run_tests
        ;;
    "help"|*)
        usage
        ;;
esac 