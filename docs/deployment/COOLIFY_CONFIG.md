# Coolify Production Configuration

## Server Access

| Parameter | Value |
|-----------|-------|
| **Server IP** | `5.189.182.153` |
| **SSH** | `ssh root@5.189.182.153` |
| **Coolify Dashboard** | http://5.189.182.153:8000 |

---

## Practix Services

### Frontend
| Parameter | Value |
|-----------|-------|
| **Container** | `nwk0wwo0gw0g0oso0g04gwwc-131935658892` |
| **URL** | https://nwk0wwo0gw0g0oso0g04gwwc.5.189.182.153.sslip.io |
| **Internal Port** | 80 |
| **VITE_API_URL** | https://wsggcg0s80cccw044s4k884c.5.189.182.153.sslip.io |

### Backend
| Parameter | Value |
|-----------|-------|
| **Container** | `wsggcg0s80cccw044s4k884c-132821394360` |
| **URL** | https://wsggcg0s80cccw044s4k884c.5.189.182.153.sslip.io |
| **Internal Port** | 8080 |
| **Database** | `postgresql://kodla:KodlaDB2026Secure@oo8ss0ockw04kcs0sswok8kw:5432/kodla` |
| **Redis** | `redis://default:KodlaRedis2026Secure@vo04w88gkkkw4w8w88skcw40:6379/0` |
| **Judge0** | `http://practix_judge0:2358` |

### Database (PostgreSQL)
| Parameter | Value |
|-----------|-------|
| **Container** | `oo8ss0ockw04kcs0sswok8kw` |
| **Internal Port** | 5432 |
| **Database** | `kodla` |
| **User** | `kodla` |
| **Password** | `KodlaDB2026Secure` |

### Redis
| Parameter | Value |
|-----------|-------|
| **Container** | `vo04w88gkkkw4w8w88skcw40` |
| **Internal Port** | 6379 |
| **Password** | `KodlaRedis2026Secure` |

### Judge0 (Code Execution)
| Parameter | Value |
|-----------|-------|
| **Server Container** | `practix_judge0` |
| **Workers Container** | `practix_judge0_workers` |
| **DB Container** | `practix_judge0_db` |
| **Redis Container** | `practix_judge0_redis` |
| **Network** | `kodla-starter_default` |
| **External Port** | 2358 |
| **Internal URL** | `http://practix_judge0:2358` |

---

## Other Projects on Server

### Kodla (Legacy)
| Parameter | Value |
|-----------|-------|
| **Frontend** | `kodla-frontend` (port 3000) |
| **Backend** | `kodla-backend` (port 8081) |

---

## Quick Commands

```bash
# SSH to server
ssh root@5.189.182.153

# Check all containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check Practix backend logs (container name changes on redeploy)
docker logs $(docker ps --format '{{.Names}}' | grep wsgg) --tail 100 -f

# Check Practix frontend logs
docker logs $(docker ps --format '{{.Names}}' | grep nwk0) --tail 100 -f

# Check Judge0 logs
docker logs practix_judge0 --tail 100 -f

# Check Judge0 workers
docker logs practix_judge0_workers --tail 100 -f

# Test Judge0 health
curl http://localhost:2358/languages

# Test backend health
curl https://wsggcg0s80cccw044s4k884c.5.189.182.153.sslip.io/health

# Restart Practix backend
docker restart $(docker ps --format '{{.Names}}' | grep wsgg)

# Restart Practix frontend
docker restart $(docker ps --format '{{.Names}}' | grep nwk0)
```

---

## E2E Testing on Production

```bash
# Run E2E tests against production
E2E_BASE_URL=https://nwk0wwo0gw0g0oso0g04gwwc.5.189.182.153.sslip.io \
E2E_API_URL=https://wsggcg0s80cccw044s4k884c.5.189.182.153.sslip.io \
E2E_TIER=QUICK \
npx playwright test e2e/tests/task-validation/go-tasks.spec.ts

# Full validation
E2E_BASE_URL=https://nwk0wwo0gw0g0oso0g04gwwc.5.189.182.153.sslip.io \
E2E_API_URL=https://wsggcg0s80cccw044s4k884c.5.189.182.153.sslip.io \
E2E_TIER=FULL \
npx playwright test e2e/tests/task-validation/
```

---

## Network Configuration

**IMPORTANT:** Backend must be connected to `kodla-starter_default` network to access Judge0!

```bash
# Connect backend to Judge0 network (required after each redeploy!)
docker network connect kodla-starter_default $(docker ps --format '{{.Names}}' | grep wsgg)

# Verify connection
docker exec $(docker ps --format '{{.Names}}' | grep wsgg) curl -s http://practix_judge0:2358/languages | head -3
```

### Networks
| Network | Purpose |
|---------|---------|
| `coolify` | Coolify managed services (frontend, backend, db, redis) |
| `kodla-starter_default` | Judge0 services (server, workers, db, redis) |

---

## Notes

- All Coolify services use `coolify` network
- Judge0 uses separate `kodla-starter_default` network
- Backend must be manually connected to `kodla-starter_default` after each redeploy
- sslip.io provides automatic SSL certificates via Let's Encrypt
- Frontend uses VITE_API_URL env var set at build time in Coolify

---

*Last updated: 2026-02-15*
