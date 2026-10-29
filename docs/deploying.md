# Deploying (phase 2: Docker and AWS)

One server runs everything with Docker compose: PostgreSQL, the API, the worker, Ollama
with the checked Qwen file, and Caddy for HTTPS ([decision 3](decisions/0003-one-server-shared-model.md)).
Projects call `https://<AGENTS_DOMAIN>/v1/...` with their keys ([API guide](api-guide.md)).

| File | What it is |
|---|---|
| `Dockerfile` | One image for the API, the worker and `init-db` |
| `compose.yaml` | The whole service on one machine |
| `deploy/Caddyfile` | HTTPS in front of the API |
| `deploy/fetch-model.sh` | Downloads and checks the Qwen Q8_0 file, writes `models/Modelfile` |
| `deploy/compose.gpu.yaml` | Ollama on an NVIDIA GPU |

## 1. Try it on the laptop first

Ollama keeps running on Windows; the containers reach it at `host.docker.internal`.

```
copy .env.example .env
```

In `.env` set `AGENTS_KEY_PEPPER` and `POSTGRES_PASSWORD` (two different values from
`agents generate-secret`) and `AGENTS_DOCKER_MODEL_URL=http://host.docker.internal:11434/v1`.

```
docker compose up -d --build
docker compose exec api agents model-check
docker compose exec api agents create-key --project test --agent cv
curl http://127.0.0.1:8000/health/ready
```

`docker compose down` stops it; `docker compose down -v` also deletes its database.

## 2. AWS server

**Server.** An EC2 instance with Ubuntu 24.04 and a 60 GB gp3 disk.

| Choice | Instance | Speed of one CV (estimate) |
|---|---|---|
| CPU, to start | c7i.2xlarge or c7a.2xlarge (8 vCPU, 16 GB) | about 1-3 minutes |
| GPU, many CVs | g6.xlarge (NVIDIA L4, 16 GB) | seconds |

Check the hourly price in your region before choosing. The model needs about 4 GB of memory.

**Network.** A security group with port 22 from your office address only, and ports 80 and
443 from the projects that call the API (or from anywhere). An Elastic IP, and a DNS `A`
record for `AGENTS_DOMAIN` pointing at it.

**Install Docker** (Ubuntu 24.04):

```
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER   # log out and in again
```

GPU servers: use an AWS Deep Learning Base GPU AMI (driver and toolkit installed), or
install the NVIDIA driver and `nvidia-container-toolkit`.

**Get the code and the model:**

```
git clone -b production-ready <repository URL> agents && cd agents
sh deploy/fetch-model.sh
cp .env.example .env && chmod 600 .env
```

In `.env` set `AGENTS_KEY_PEPPER`, `POSTGRES_PASSWORD`, `AGENTS_DOMAIN`, `AGENTS_ENABLED_AGENTS`
and any lookup agent keys. Leave `AGENTS_DOCKER_MODEL_URL` empty. Never set `AGENTS_DEBUG_DIR` here.

**Start:**

```
docker compose --profile ollama --profile https up -d --build
# GPU: docker compose -f compose.yaml -f deploy/compose.gpu.yaml --profile ollama --profile https up -d --build
docker compose exec api agents model-check
docker compose exec api agents accuracy
docker compose exec api agents create-key --project <project> --agent cv
```

Check from your laptop: `curl https://<AGENTS_DOMAIN>/health/ready`.

## 3. Running it

| Task | Command |
|---|---|
| Update | `git pull && docker compose --profile ollama --profile https up -d --build` |
| Logs | `docker compose logs -f api worker` |
| More workers | `AGENTS_WORKERS=2` in `.env`, then the start command again |
| Usage | `docker compose exec api agents stats` |
| Database backup | `docker compose exec db pg_dump -U agents agents > backup.sql` (copy it off the server, e.g. to S3) |

Later steps, settings only: the database on Amazon RDS (`AGENTS_DATABASE_URL`), uploaded files in
S3 (`AGENTS_STORAGE=s3`), and the model on a separate GPU server (`AGENTS_DOCKER_MODEL_URL`).
