"""
Unit tests for SpaceNetra Docker Stack setup (Phase 26).
"""

from pathlib import Path
import yaml
import pytest


def test_docker_compose_structure():
    root_dir = Path(__file__).resolve().parent.parent
    compose_path = root_dir / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist"

    with open(compose_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "services" in config, "docker-compose.yml must define services"
    services = config["services"]
    
    expected_services = ["frontend", "backend", "model-service", "qdrant", "postgres"]
    for svc in expected_services:
        assert svc in services, f"Service {svc} must be defined in docker-compose.yml"


def test_dockerfiles_exist():
    root_dir = Path(__file__).resolve().parent.parent
    assert (root_dir / "Dockerfile").exists(), "Root Dockerfile must exist"
    assert (root_dir / "frontend" / "Dockerfile").exists(), "Frontend Dockerfile must exist"
    assert (root_dir / "frontend" / "nginx.conf").exists(), "Frontend nginx.conf must exist"
    assert (root_dir / "database" / "init.sql").exists(), "Database init.sql must exist"
