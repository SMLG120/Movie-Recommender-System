import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[0]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.yml"

def deploy():
    logging.info("Rebuilding recommender service image...")
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "build"],
        cwd=REPO_ROOT,
        check=True,
    )

    logging.info("Stopping existing service (if any)...")
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
        cwd=REPO_ROOT,
        check=False,  # allow no-op if nothing running
    )

    logging.info("Restarting recommender service...")
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"],
        cwd=REPO_ROOT,
        check=True,
    )

    logging.info("Deployment complete (canary handled at model level)")

if __name__ == "__main__":
    deploy()
