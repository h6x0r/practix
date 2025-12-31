SHELL := /bin/sh
DOCKER_COMPOSE ?= docker compose

.PHONY: migrate-up vet build start start-docker db-reset db-seed db-migrate db-studio db-refresh \
        start-all stop-all clean clean-deep disk clean-cache piston-install-langs piston-langs piston-status

migrate-up:
	cd server && npx prisma migrate deploy

vet:
	npm run build
	cd server && npm run build

build:
	$(DOCKER_COMPOSE) build

start:
	@echo "Starting frontend (Vite) and backend (Nest) locally..."
	@set -e; \
	trap "exit" INT TERM; \
	trap "kill 0" EXIT; \
	( cd server && npm run start:dev ) & \
	npm run dev

start-docker:
	$(DOCKER_COMPOSE) up --build

# Database commands for Docker environment
db-reset:
	@echo "🗑️  Resetting database in Docker..."
	cd server && npx prisma migrate reset --force --skip-seed

db-seed:
	@echo "🌱 Seeding database in Docker..."
	cd server && npm run seed

db-migrate:
	@echo "📊 Running migrations in Docker..."
	cd server && npx prisma migrate deploy

db-studio:
	@echo "🎨 Opening Prisma Studio..."
	cd server && npx prisma studio

# Combined command: reset DB and seed
db-refresh: db-reset db-seed
	@echo "✅ Database refreshed successfully!"

# ============================================================================
# Full Stack Commands
# ============================================================================

# Start everything
start-all:
	@echo "🚀 Starting full stack..."
	$(DOCKER_COMPOSE) up --build -d
	@echo "✅ Full stack is running!"
	@echo "   Frontend: http://localhost:3000"
	@echo "   Backend:  http://localhost:8080"
	@echo "   Piston:   http://localhost:2000"

# Stop everything
stop-all:
	@echo "🛑 Stopping all services..."
	$(DOCKER_COMPOSE) down
	@echo "✅ All services stopped"

# ============================================================================
# Docker Cleanup & Maintenance
# ============================================================================

# Quick cleanup (safe, removes only unused)
clean:
	@echo "🧹 Quick Docker cleanup..."
	docker system prune -f
	@echo "✅ Cleanup complete!"
	@docker system df

# Deep cleanup (removes everything unused including volumes)
clean-deep:
	@echo "⚠️  Deep Docker cleanup (includes unused volumes)..."
	docker system prune -af --volumes
	@echo "✅ Deep cleanup complete!"
	@docker system df

# Show Docker disk usage
disk:
	@echo "📊 Docker Disk Usage:"
	@docker system df
	@echo ""
	@echo "🖼️  Top 5 largest images:"
	@docker images --format "{{.Size}}\t{{.Repository}}:{{.Tag}}" | sort -hr | head -5

# Clean build cache only
clean-cache:
	@echo "🗑️  Cleaning build cache..."
	docker builder prune -af
	@echo "✅ Build cache cleared!"

# ============================================================================
# Piston Language Management
# ============================================================================

# Install additional languages in Piston
piston-install-langs:
	@echo "📦 Installing additional languages in Piston..."
	docker exec kodla_piston piston ppman install rust || true
	docker exec kodla_piston piston ppman install c++ || true
	docker exec kodla_piston piston ppman install javascript || true
	docker exec kodla_piston piston ppman install ruby || true
	@echo "✅ Languages installed!"

# List installed Piston languages
piston-langs:
	@echo "📋 Installed Piston languages:"
	@curl -s http://localhost:2000/api/v2/runtimes | python3 -c "import sys,json; [print(f'  {r[\"language\"]} {r[\"version\"]}') for r in json.load(sys.stdin)]" 2>/dev/null || echo "❌ Piston not running"

# Check Piston status
piston-status:
	@echo "🔍 Piston Status:"
	@curl -s http://localhost:2000/api/v2/runtimes | python3 -c "import sys,json; langs=json.load(sys.stdin); print(f'  ✅ Running with {len(langs)} runtimes')" 2>/dev/null || echo "  ❌ Not running"
