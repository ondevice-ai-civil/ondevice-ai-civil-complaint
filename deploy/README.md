# Deploy Configuration

This directory contains all deployment-related configuration files.

## Structure

| Directory | Contents |
|-----------|----------|
| `docker/` | Dockerfiles for production, CI, and HF Space |
| `compose/` | Docker Compose files for different environments |
| `env/` | Environment variable templates (`.env.example` variants) |
| `nginx/` | Nginx reverse proxy configuration |

## Quick Start

```bash
# From the project root:
cp deploy/env/.env.example .env
# Edit .env with your settings

docker compose -f deploy/compose/docker-compose.yml up -d --build
```

## Docker Build

Dockerfiles are located in `deploy/docker/` but must be built with the project root as context:

```bash
docker build -f deploy/docker/Dockerfile -t govon-backend .
```

## Compose Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Default GPU development |
| `docker-compose.ci.yml` | CI smoke tests (no ML packages) |
| `docker-compose.offline.yml` | Air-gapped / offline deployment |
| `docker-compose.prod.yml` | Production Blue/Green deployment |
