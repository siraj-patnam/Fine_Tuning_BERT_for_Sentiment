# Twitter Sentiment Analysis API - Simple Setup Guide

This is a simplified setup for deploying a Twitter sentiment analysis API using FastAPI, Docker, and Nginx.

## Project Structure

```
sentiment-analysis-app/
│
├── app/
│   ├── main.py         # FastAPI application
│   └── model_utils.py  # Utilities for safe model loading
│
├── Dockerfile          # Docker configuration for the API
├── requirements.txt    # Python dependencies
├── nginx.conf          # Nginx configuration
└── docker-compose.yml  # Docker Compose configuration
```

## Quick Start

1. Make sure Docker and Docker Compose are installed on your system.

2. Create the files as shown in the provided artifacts.

3. Build and start the services:
   ```bash
   docker-compose up -d
   ```

4. The API will be accessible at:
   - API Endpoints: http://localhost/api
   - API Documentation: http://localhost/api/docs
   - Health Check: http://localhost/api/health

## Usage Example


curl -X 'POST' \
  'http://localhost/api/sentiment' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "I absolutely love this new product! It'\''s amazing."
}'


## Scaling

To run multiple instances of the API:

docker-compose up -d --scale api=3
```

This will start 3 instances of the API, and Nginx will automatically load balance between them.

## Notes

- This setup focuses only on the core functionality (Docker and Nginx).
- The model `bert-base-uncased-sentiment-model` should be available or replaced with an appropriate model path.
- For production use, consider adding volume mounts for model persistence.