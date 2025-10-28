# Show help message
help:
	@echo "Production Commands"
	@echo "===================="
	@echo ""
	@echo "Poetry Package Management:"
	@echo "  make add PKG=<package>          - Add a new package"
	@echo "  make add PKG=<package> WITH=<group>  - Add package to specific group"
	@echo "  make install                    - Install all dependencies"
	@echo "  make install WITH=<group>       - Install dependencies for specific group"
	@echo "  make update                     - Update all dependencies"
	@echo "  make run                        - Run Streamlit app"
	@echo ""
	@echo "Docker (Production):"
	@echo "  make docker-prod-build          - Build images"
	@echo "  make docker-prod-up             - Start containers"
	@echo "  make docker-prod-down           - Stop containers"
	@echo ""
	@echo "Langfuse (Observability):"
	@echo "  make langfuse-up                - Start Langfuse services"
	@echo "  make langfuse-down              - Stop Langfuse services"
	@echo "  make langfuse-restart           - Restart Langfuse services"
	@echo "  make langfuse-logs              - View Langfuse logs"
	@echo "  make langfuse-logs-web          - View web service logs"
	@echo "  make langfuse-ps                - Show Langfuse service status"
	@echo "  make langfuse-reset             - Reset Langfuse (delete all data)"


add:
	@if [ -z "$(PKG)" ]; then \
		echo "Error: Package name is required. Example: make add PKG=torch"; \
		exit 1; \
	fi
	poetry add $(PKG) $(if $(WITH), --group $(WITH),)

install:
	poetry lock
	poetry install $(if $(WITH), --with $(WITH),)

update:
	poetry update

run:
	PYTHONPATH=. poetry run streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0

docker-prod-up:
	docker compose -f ./docker/docker-compose.yaml --profile prod up -d

docker-prod-build:
	DOCKER_BUILDKIT=1 docker compose -f ./docker/docker-compose.yaml --profile prod build

docker-prod-down:
	docker compose -f ./docker/docker-compose.yaml --profile prod down

langfuse-up:
	docker compose -f ./langfuse/docker-compose.yml up -d

langfuse-down:
	docker compose -f ./langfuse/docker-compose.yml down

langfuse-restart:
	docker compose -f ./langfuse/docker-compose.yml restart

langfuse-logs:
	docker compose -f ./langfuse/docker-compose.yml logs -f

langfuse-logs-web:
	docker compose -f ./langfuse/docker-compose.yml logs -f web

langfuse-ps:
	docker compose -f ./langfuse/docker-compose.yml ps

langfuse-reset:
	@echo "WARNING: This will delete all Langfuse data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose -f ./langfuse/docker-compose.yml down -v; \
		echo "Langfuse data has been reset."; \
	else \
		echo "Reset cancelled."; \
	fi
