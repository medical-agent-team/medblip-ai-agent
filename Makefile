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
	poetry run streamlit run app/main.py

docker-prod-up:
	docker compose -f ./docker/docker-compose.yaml --profile prod up -d

docker-prod-build:
	docker compose -f ./docker/docker-compose.yaml --profile prod build

docker-prod-down:
	docker compose -f ./docker/docker-compose.yaml --profile prod down
