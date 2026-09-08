"""
Docker Compose & Container Stack Verification Script for SpaceNetra.
Validates the presence, structure, and service definitions of docker-compose.yml, Dockerfiles, and init scripts.
"""

import sys
from pathlib import Path
import yaml


def verify_docker_stack() -> bool:
    root_dir = Path(__file__).resolve().parent.parent
    errors = []

    # 1. Check docker-compose.yml existence and syntax
    compose_path = root_dir / "docker-compose.yml"
    if not compose_path.exists():
        errors.append("docker-compose.yml is missing.")
    else:
        try:
            with open(compose_path, "r", encoding="utf-8") as f:
                compose_config = yaml.safe_load(f)
            
            services = compose_config.get("services", {})
            required_services = ["frontend", "backend", "model-service", "qdrant", "postgres"]
            for svc in required_services:
                if svc not in services:
                    errors.append(f"Missing required service '{svc}' in docker-compose.yml.")
            print(f"[SUCCESS] Verified docker-compose.yml with services: {list(services.keys())}")
        except Exception as e:
            errors.append(f"Failed to parse docker-compose.yml: {e}")

    # 2. Check Backend Dockerfile
    backend_df = root_dir / "Dockerfile"
    if not backend_df.exists():
        errors.append("Backend Dockerfile is missing.")
    else:
        print("[SUCCESS] Verified Backend Dockerfile.")

    # 3. Check Frontend Dockerfile & Nginx Config
    frontend_df = root_dir / "frontend" / "Dockerfile"
    nginx_cfg = root_dir / "frontend" / "nginx.conf"
    if not frontend_df.exists():
        errors.append("frontend/Dockerfile is missing.")
    if not nginx_cfg.exists():
        errors.append("frontend/nginx.conf is missing.")
    if frontend_df.exists() and nginx_cfg.exists():
        print("[SUCCESS] Verified frontend container setup (Dockerfile & nginx.conf).")

    # 4. Check Database init.sql
    init_sql = root_dir / "database" / "init.sql"
    if not init_sql.exists():
        errors.append("database/init.sql is missing.")
    else:
        print("[SUCCESS] Verified database/init.sql PostGIS script.")

    if errors:
        print("\n--- Verification Errors ---")
        for err in errors:
            print(f"[ERROR] {err}")
        return False

    print("\n[OK] SpaceNetra Docker Compose Container Stack verification PASSED!")
    return True


if __name__ == "__main__":
    success = verify_docker_stack()
    sys.exit(0 if success else 1)
