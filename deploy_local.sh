#!/bin/bash

echo "🔄 Pulling latest image from Docker Hub..."
docker compose pull

echo "🧼 Stopping and removing existing containers..."
docker compose down

echo "🚀 Starting the housing-api service using Docker Compose..."
docker compose up -d

echo "✅ Housing API should now be running at: http://localhost:8000/docs"

