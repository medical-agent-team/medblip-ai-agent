# Show help message
help:
	@echo "Medical Agent Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Poetry Package Management:"
	@echo "  make add PKG=<package>          - Add a new package"
	@echo "  make add PKG=<package> WITH=<group>  - Add package to specific group"
	@echo "  make install                    - Install all dependencies"
	@echo "  make install WITH=<group>       - Install dependencies for specific group"
	@echo "  make update                     - Update all dependencies"
	@echo "  make run                        - Run Streamlit app"
	@echo "  make test                       - Run integration tests"
	@echo ""
	@echo "Docker Commands:"


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
	poetry run streamlit run app/first_service.py

test:
	poetry run python test_medblip.py

docker-dev-up:
	docker compose -f ./docker/docker-compose.yaml --profile dev up -d

docker-dev-build:
	docker compose -f ./docker/docker-compose.yaml --profile dev build

docker-dev-down:
	docker compose -f ./docker/docker-compose.yaml --profile dev down

docker-prod-up:
	docker compose -f ./docker/docker-compose.yaml --profile prod up -d

docker-prod-down:
	docker compose -f ./docker/docker-compose.yaml --profile prod down
