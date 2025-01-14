#!/bin/bash
set -e

VENV_DIR=".venv"

check_port() {
    local port=$1
    if lsof -i ":$port" >/dev/null 2>&1; then
        local new_port=$((port + 1000))
        while lsof -i ":$new_port" >/dev/null 2>&1; do
            new_port=$((new_port + 1))
        done
        >&2 echo "Port $port is in use. Using port $new_port instead..."
        echo "$new_port"
    fi
    echo "$port"
}

setup_dev_env() {
    echo "Creating virtual environment..."
    if ! command -v uv &> /dev/null; then
        echo "uv is not installed. Installing uv..."
        python3 -m pip install --user uv
    fi

    uv venv "$VENV_DIR"
     source "$VENV_DIR"/bin/activate
    echo "Installing development dependencies..."
    uv pip install -e ".[dev]"
}

setup_docker_services() {
    local service_name="${1:-all}"

    if ! docker compose version >/dev/null 2>&1; then
        echo "Docker Compose not found. Please install Docker Desktop or docker-compose-plugin."
        exit 1
    fi

    # Check and set alternative ports if needed
    export APP_PORT=$(check_port 8501)
    export NEO4J_HTTP_PORT=$(check_port 7474)
    export NEO4J_BOLT_PORT=$(check_port 7687)

    # Determine which services to update
    case "$service_name" in
        all)
            echo "Starting all services..."
            docker compose up -d --build
            ;;
        app)
            echo "Updating app service..."
            docker compose up -d --build app
            ;;
        neo4j)
            echo "Updating Neo4j service..."
            docker compose up -d --build neo4j
            ;;
        *)
            echo "Unknown service: $service_name"
            echo "Available services: all, app, neo4j"
            exit 1
            ;;
    esac

    echo "Waiting for services to be healthy..."
    docker compose ps --format json | grep -q '"Health": "healthy"' || sleep 10
}

# Process command line arguments
FRESH=false
WITH_DOCKER=false
SERVICE_NAME="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh) FRESH=true ;;
        --with-docker) WITH_DOCKER=true ;;
        --service)
            shift
            SERVICE_NAME="$1"
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

# Setup development environment
if [ "$FRESH" = true ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ] || [ "$FRESH" = true ]; then
    setup_dev_env
else
    echo "Using existing virtual environment."
    source "$VENV_DIR"/bin/activate
fi

# Setup Docker services if requested
[ "$WITH_DOCKER" = true ] && setup_docker_services "$SERVICE_NAME"

# Create default .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating default .env file..."
    cp .env.example .env
fi

echo "
Development environment ready! 🚀

Services:
- App: http://localhost:${APP_PORT:-8501}
- Neo4j:
  Browser: http://localhost:${NEO4J_HTTP_PORT:-7474}
  Bolt: localhost:${NEO4J_BOLT_PORT:-7687}
  Credentials: neo4j/taxrag_dev_password

Management:
- Start: docker compose up -d neo4j
- Stop: docker compose down neo4j
- Logs: docker compose logs -f neo4j    
- Status: docker compose ps neo4j
"
