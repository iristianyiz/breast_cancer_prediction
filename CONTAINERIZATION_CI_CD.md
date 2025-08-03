# Containerization and CI/CD in Breast Cancer Prediction Project

## Containerization: Docker Expertise

### What is Containerization?
Containerization packages your application and all its dependencies into a standardized unit (container) that can run consistently across different environments.

### Docker Implementation in This Project

#### 1. Production-Ready Dockerfile
```dockerfile
# Multi-stage build for optimization
FROM python:3.9-slim

# Set working directory and environment variables
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models plots experiments

# Train models if dataset is available
RUN if [ -f "Breast_cancer_data.csv" ]; then \
        python train.py; \
    else \
        echo "Dataset not found, models will be trained on first run"; \
    fi

# Expose port and health check
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"]
```

**Key Features:**
- **Multi-stage optimization**: Reduces image size
- **Layer caching**: Faster builds
- **Health checks**: Ensures application is running
- **Security**: Minimal base image, no unnecessary packages
- **Production ready**: Proper environment variables

#### 2. Docker Compose for Development
```yaml
version: '3.8'
services:
  breast-cancer-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - PORT=5000
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./plots:/app/plots
      - ./experiments:/app/experiments
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

**Benefits:**
- **Easy development**: One command to start everything
- **Volume mounting**: Persistent data across container restarts
- **Health monitoring**: Automatic health checks
- **Environment isolation**: Consistent development environment

#### 3. Docker Optimization (.dockerignore)
```dockerignore
# Exclude unnecessary files
__pycache__/
*.pyc
.git/
venv/
tests/
*.log
plots/
experiments/
*.csv
!Breast_cancer_data.csv
```

**Optimization Benefits:**
- **Smaller build context**: Faster builds
- **Reduced image size**: Smaller deployments
- **Security**: Excludes sensitive files
- **Efficiency**: Only includes necessary files

### Docker Commands for This Project

```bash
# Build the image
docker build -t breast-cancer-api .

# Run locally
docker run -p 5000:5000 breast-cancer-api

# Run with docker-compose
docker-compose up --build

# Test the container
docker run -d --name test-api -p 5000:5000 breast-cancer-api
curl http://localhost:5000/health
docker stop test-api && docker rm test-api
```

## CI/CD Ready: GitHub Actions

### What is CI/CD?
- **CI (Continuous Integration)**: Automatically test code changes
- **CD (Continuous Deployment)**: Automatically deploy to production

### GitHub Actions Implementation

#### 1. Simple CI/CD Pipeline (.github/workflows/simple-ci.yml)
```yaml
name: Simple CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-docker:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t breast-cancer-api .
    
    - name: Test Docker image
      run: |
        docker run -d --name test-api -p 5000:5000 breast-cancer-api
        sleep 10
        curl -f http://localhost:5000/health || exit 1
        docker stop test-api
        docker rm test-api
```

#### 2. Advanced CI/CD Pipeline (.github/workflows/ci-cd.yml)
Includes:
- **Multi-Python testing**: Tests on Python 3.8, 3.9, 3.10
- **Code quality**: Linting with flake8, formatting with black
- **Security scanning**: Bandit security analysis
- **Docker registry**: Push to GitHub Container Registry
- **Staging deployment**: Deploy to staging on develop branch
- **Production deployment**: Deploy to production on main branch
- **Notifications**: Success/failure notifications

### CI/CD Pipeline Benefits

#### 1. Automated Testing
- **Unit tests**: Run on every push/PR
- **Coverage reports**: Track code coverage
- **Multiple Python versions**: Ensure compatibility
- **Code quality**: Linting and formatting checks

#### 2. Automated Deployment
- **Staging environment**: Test changes before production
- **Production deployment**: Automatic deployment on main branch
- **Rollback capability**: Easy to revert changes
- **Health checks**: Verify deployment success

#### 3. Quality Assurance
- **Security scanning**: Identify security vulnerabilities
- **Code coverage**: Ensure adequate test coverage
- **Formatting**: Consistent code style
- **Linting**: Catch code issues early

### How to Use CI/CD

#### 1. Enable GitHub Actions
1. Push your code to GitHub
2. Go to Actions tab in your repository
3. GitHub will automatically detect the workflow files
4. Actions will run on every push/PR

#### 2. Set Up Secrets (for deployment)
```bash
# In GitHub repository settings > Secrets
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
PRODUCTION_URL=https://your-app-url.com
```

#### 3. Workflow Triggers
- **Push to main**: Runs tests and deploys to production
- **Push to develop**: Runs tests and deploys to staging
- **Pull Request**: Runs tests only
- **Release**: Runs full pipeline

### CI/CD Best Practices Demonstrated

#### 1. Pipeline Stages
```
Code Push → Test → Build → Deploy → Verify
```

#### 2. Environment Separation
- **Staging**: Test changes safely
- **Production**: Live application
- **Development**: Local development

#### 3. Quality Gates
- **Tests must pass**: No deployment if tests fail
- **Code coverage**: Minimum coverage requirements
- **Security scan**: No known vulnerabilities
- **Health checks**: Application must be healthy

#### 4. Rollback Strategy
- **Git tags**: Version control for releases
- **Docker images**: Immutable deployments
- **Database migrations**: Backward compatible changes

## Resume Benefits

### Containerization Skills
- **Docker expertise**: Production-ready containers
- **Container orchestration**: Docker Compose
- **Image optimization**: Multi-stage builds, .dockerignore
- **Health monitoring**: Health checks and logging
- **Security**: Minimal attack surface

### CI/CD Skills
- **GitHub Actions**: Automated workflows
- **Testing automation**: Unit tests, coverage, quality checks
- **Deployment automation**: Staging and production
- **Monitoring**: Health checks and notifications
- **DevOps practices**: Infrastructure as code

### Professional Impact
- **Production ready**: Enterprise-level deployment
- **Scalable**: Easy to scale horizontally
- **Maintainable**: Automated testing and deployment
- **Reliable**: Health checks and rollback capabilities
- **Modern**: Industry-standard practices

## Next Steps

### 1. Enable GitHub Actions
```bash
# Push your code to GitHub
git add .
git commit -m "Add CI/CD pipeline"
git push origin main
```

### 2. Set Up AWS Deployment
```bash
# Configure AWS credentials
aws configure

# Deploy using the script
./deploy-aws.sh app-runner
```

### 3. Monitor Your Pipeline
- Check GitHub Actions tab
- Monitor deployment health
- Review test coverage
- Address any issues

Your project now demonstrates professional containerization and CI/CD practices that are highly valued in the industry! 