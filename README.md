# AI Model Deployment - Lab 4

Iris flower classification API using FastAPI and Docker.

## Project files
- app/: FastAPI application and tests
- models/: trained Iris classification model
- requirements.txt: Python dependencies
- Dockerfile.baseline: baseline Docker configuration

## API endpoints
- GET /health
- POST /v1/predict

## Run tests
python -m pytest -v app/test_main.py
