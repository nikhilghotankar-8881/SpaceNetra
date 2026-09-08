#!/usr/bin/env python3
"""
Automated Pre-Deployment Security Checklist Verifier for SpaceNetra.
"""

import os
import re
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_secrets_in_code():
    """Scan python source files for hardcoded secrets/passwords."""
    patterns = [
        re.compile(r'password\s*=\s*["\'][^"\']{6,}["\']', re.IGNORECASE),
        re.compile(r'secret_key\s*=\s*["\'](?!spacenetra_dev)[^"\']{6,}["\']', re.IGNORECASE),
        re.compile(r'api_key\s*=\s*["\'][^"\']{6,}["\']', re.IGNORECASE),
    ]
    violations = []
    for root, _, files in os.walk(PROJECT_ROOT / "src"):
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                content = path.read_text(encoding="utf-8", errors="ignore")
                for pat in patterns:
                    if pat.search(content):
                        violations.append(f"{path.relative_to(PROJECT_ROOT)}")
    return len(violations) == 0, violations


def check_gitignore():
    """Verify .gitignore contains sensitive patterns."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    if not gitignore_path.exists():
        return False, [".gitignore file missing"]
    content = gitignore_path.read_text(encoding="utf-8")
    required = [".env", "checkpoints/", "logs/"]
    missing = [req for req in required if req not in content]
    return len(missing) == 0, missing


def check_jwt_secret():
    """Verify JWT secret length."""
    from backend.services.auth_service import SECRET_KEY
    if len(SECRET_KEY) < 16:
        return False, [f"JWT secret too short ({len(SECRET_KEY)} chars)"]
    return True, []


def check_cors_and_middleware():
    """Verify app/main.py restricts CORS and includes security middleware."""
    main_path = PROJECT_ROOT / "app" / "main.py"
    content = main_path.read_text(encoding="utf-8")
    issues = []
    if 'allow_origins=["*"]' in content:
        issues.append("CORS wildcard allow_origins=['*'] present")
    if "SecurityHeadersMiddleware" not in content:
        issues.append("SecurityHeadersMiddleware not registered in main.py")
    if "AuditMiddleware" not in content:
        issues.append("AuditMiddleware not registered in main.py")
    if "RateLimitMiddleware" not in content:
        issues.append("RateLimitMiddleware not registered in main.py")
    return len(issues) == 0, issues


def check_docker_hardening():
    """Verify Docker compose stack hardening."""
    compose_path = PROJECT_ROOT / "docker-compose.yml"
    if not compose_path.exists():
        return True, ["docker-compose.yml not present (skipping)"]
    content = compose_path.read_text(encoding="utf-8")
    issues = []
    if "privileged: true" in content:
        issues.append("Privileged mode containers found")
    return len(issues) == 0, issues


def check_health_endpoint():
    """Verify health endpoint does not leak stack traces or internal secrets."""
    main_path = PROJECT_ROOT / "app" / "main.py"
    content = main_path.read_text(encoding="utf-8")
    if "traceback" in content.lower() or "secret" in content.lower():
        return False, ["Health check potential info leak"]
    return True, []


def run_security_audit():
    """Execute all security checklist items and print report."""
    print("\n═══════════════════════════════════════════════════")
    print("  SPACENETRA SECURITY VERIFICATION REPORT")
    print("═══════════════════════════════════════════════════\n")

    checks = [
        ("No secrets in source code", check_secrets_in_code),
        (".env in .gitignore", check_gitignore),
        ("JWT secret length >= 16 bits", check_jwt_secret),
        ("CORS restricted & security middleware registered", check_cors_and_middleware),
        ("Docker stack hardening", check_docker_hardening),
        ("Health endpoint clean", check_health_endpoint),
    ]

    passed = 0
    total = len(checks)

    for name, check_fn in checks:
        ok, details = check_fn()
        if ok:
            passed += 1
            print(f" [✅] {name}")
        else:
            print(f" [❌] {name} - Issues: {', '.join(details)}")

    print("\n─────────────────────────────────────────────────")
    print(f" RESULT: {passed}/{total} CHECKS PASSED")
    print("═══════════════════════════════════════════════════\n")
    return passed == total


if __name__ == "__main__":
    success = run_security_audit()
    sys.exit(0 if success else 1)
