# 🔐 Security Architecture — SpaceNetra

> Security design for an offline, defence-oriented satellite intelligence system

---

## Table of Contents

- [Security Principles](#security-principles)
- [Threat Model](#threat-model)
- [Network Security](#network-security)
- [Authentication & Authorization](#authentication--authorization)
- [Data Protection](#data-protection)
- [Model Security](#model-security)
- [Audit & Logging](#audit--logging)
- [Container Security](#container-security)
- [Operational Security](#operational-security)
- [Security Checklist](#security-checklist)

---

## Security Principles

> **Offline ≠ Automatically Secure**

The system operates in a defence-oriented environment. Security is not optional.

### Core Principles

| Principle | Implementation |
|-----------|---------------|
| **Zero External Dependency** | No external AI APIs, no telemetry, no analytics |
| **Defense in Depth** | Multiple security layers, no single point of failure |
| **Least Privilege** | Each component has minimum required permissions |
| **Audit Everything** | All actions logged, all decisions traceable |
| **Data Sovereignty** | All data stays on-premise, encrypted at rest |
| **Model Integrity** | Verified checkpoints, version-controlled models |

---

## Threat Model

### Attack Surface

```
┌──────────────────────────────────────────────────────┐
│                   THREAT MODEL                        │
│                                                      │
│  EXTERNAL THREATS (mitigated by air-gap):            │
│  ├── Remote code execution          → Air-gapped     │
│  ├── Network-based attacks          → No internet    │
│  ├── Supply chain (runtime)         → Offline ops    │
│  └── Data exfiltration (network)    → No egress      │
│                                                      │
│  INTERNAL THREATS (must address):                    │
│  ├── Unauthorized physical access   → Auth + RBAC    │
│  ├── Insider data theft             → Audit logs     │
│  ├── Model tampering               → Checksums       │
│  ├── Privilege escalation           → Least privilege │
│  ├── SQL injection                  → Parameterized  │
│  ├── Input manipulation             → Validation     │
│  └── Container escape               → Hardening      │
│                                                      │
│  DATA INTEGRITY THREATS:                             │
│  ├── Corrupted training data        → Validation     │
│  ├── Adversarial inputs             → Input checks   │
│  ├── Model poisoning                → Checksum verify │
│  └── Label manipulation             → Audit trail    │
└──────────────────────────────────────────────────────┘
```

---

## Network Security

### Air-Gap Architecture

```
┌─────────────────────────────────────────┐
│         OPERATIONAL NETWORK              │
│         (Air-gapped / Isolated)          │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  SpaceNetra Docker Stack          │  │
│  │                                   │  │
│  │  Only internal Docker network     │  │
│  │  No outbound connections          │  │
│  │  No DNS resolution needed         │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Access: Authorized terminals only     │
│  Protocol: HTTPS (self-signed cert)    │
└─────────────────────────────────────────┘
         │
    Physical gap
    (no cable)
         │
┌─────────────────────────────────────────┐
│         DATA TRANSFER NETWORK            │
│         (Controlled access)              │
│                                         │
│  • Download satellite imagery           │
│  • Download model updates               │
│  • Transfer via encrypted USB           │
└─────────────────────────────────────────┘
```

### Network Rules

| Rule | Detail |
|------|--------|
| No outbound internet | All containers blocked from external access |
| No external API calls | No OpenAI, no Google APIs, no telemetry |
| Internal network only | Docker bridge network, no host networking |
| TLS for inter-service | All internal communication encrypted |
| No DNS dependency | Services referenced by container name |

---

## Authentication & Authorization

### Authentication Methods

```
┌──────────────────────────────────────────────┐
│           AUTHENTICATION FLOW                 │
│                                              │
│  User/Analyst                                │
│       │                                      │
│       ▼                                      │
│  ┌──────────┐                                │
│  │  LOGIN   │  username + password           │
│  └────┬─────┘                                │
│       │                                      │
│       ▼                                      │
│  ┌──────────┐                                │
│  │ VALIDATE │  bcrypt hash comparison        │
│  └────┬─────┘                                │
│       │                                      │
│       ▼                                      │
│  ┌──────────┐                                │
│  │ JWT TOKEN│  Signed with server secret     │
│  │ (1h exp) │  Contains: user_id, role       │
│  └────┬─────┘                                │
│       │                                      │
│       ▼                                      │
│  ┌──────────┐                                │
│  │API CALLS │  Bearer token in header        │
│  └──────────┘                                │
└──────────────────────────────────────────────┘
```

### Role-Based Access Control (RBAC)

| Role | Permissions |
|------|------------|
| **Viewer** | View map, browse results, view events |
| **Analyst** | All viewer + submit feedback, run searches |
| **Operator** | All analyst + trigger change detection, manage data |
| **Admin** | All operator + manage users, view audit logs, system config |

### API Key Management

- API keys stored as bcrypt hashes in database
- Keys rotatable without downtime
- Per-key rate limiting and usage tracking
- No API keys in source code (environment variables only)

---

## Data Protection

### Encryption

| Layer | Method |
|-------|--------|
| **At Rest** | AES-256 for sensitive data, filesystem encryption |
| **In Transit** | TLS 1.3 for all inter-service communication |
| **Database** | PostgreSQL with encrypted tablespaces |
| **Backups** | Encrypted backup files |
| **Model files** | SHA-256 checksums for integrity |

### Sensitive Data Handling

```
┌──────────────────────────────────────────────────┐
│              DATA CLASSIFICATION                  │
│                                                  │
│  PUBLIC:                                         │
│  ├── Open satellite imagery (Sentinel-2)         │
│  ├── LEVIR-CD training data                      │
│  └── System documentation                       │
│                                                  │
│  INTERNAL:                                       │
│  ├── Model weights and checkpoints               │
│  ├── Vector embeddings                           │
│  ├── System configuration                        │
│  └── Processing logs                             │
│                                                  │
│  RESTRICTED:                                     │
│  ├── Analyst feedback and decisions              │
│  ├── Change event assessments                    │
│  ├── User credentials                            │
│  └── Audit logs                                  │
│                                                  │
│  CLASSIFIED (if applicable):                     │
│  ├── Operational satellite imagery               │
│  ├── Intelligence assessments                    │
│  └── Location-specific analysis                  │
└──────────────────────────────────────────────────┘
```

### No Secrets in Code

```
# ❌ NEVER
DATABASE_URL = "postgresql://user:password@localhost/db"
API_KEY = "sk-1234567890"

# ✅ ALWAYS
DATABASE_URL = os.environ["DATABASE_URL"]
API_KEY = os.environ["API_KEY"]
```

All secrets via:
- Environment variables
- Docker secrets
- `.env` file (never committed to git)

---

## Model Security

### Model Integrity

```
┌──────────────────────────────────────────────┐
│           MODEL INTEGRITY CHAIN               │
│                                              │
│  Training Environment                        │
│       │                                      │
│       ▼                                      │
│  Model Checkpoint                            │
│       │                                      │
│       ▼                                      │
│  SHA-256 Hash                                │
│  Version Tag                                 │
│  Training Config Hash                        │
│       │                                      │
│       ▼                                      │
│  Signed Manifest                             │
│  {                                           │
│    "model": "siamese_unet_v1.0.0",          │
│    "hash": "sha256:a3f2b1c4...",            │
│    "trained_on": "LEVIR-CD",                │
│    "date": "2026-01-15",                    │
│    "metrics": {"f1": 0.891, "iou": 0.804}   │
│  }                                           │
│       │                                      │
│       ▼                                      │
│  Deployment: Verify hash before loading      │
└──────────────────────────────────────────────┘
```

### Verification on Load

```python
# Before loading any model checkpoint:
expected_hash = manifest["hash"]
actual_hash = sha256(checkpoint_file)

if actual_hash != expected_hash:
    raise SecurityError("Model checkpoint integrity check failed!")
```

### Training Data Governance

- Only use authorized/public training datasets for prototype
- No sensitive operational imagery in training data
- Document all training data sources in provenance
- Version control training configurations

---

## Audit & Logging

### What Gets Logged

| Event | Logged Data |
|-------|-------------|
| **Authentication** | User, timestamp, success/failure, IP |
| **Search queries** | User, query text, timestamp, result count |
| **Change detection** | User, tile IDs, timestamp, model version |
| **Feedback** | User, event ID, decision, timestamp |
| **Data ingestion** | Scene ID, timestamp, processing steps |
| **System changes** | Config changes, model loads, index rebuilds |
| **Errors** | Full stack trace, request context |

### Log Format

```json
{
  "timestamp": "2026-01-21T10:15:30.123Z",
  "level": "INFO",
  "event": "SEARCH_QUERY",
  "user_id": "analyst_001",
  "details": {
    "query": "construction activity",
    "results_count": 10,
    "search_time_ms": 45
  },
  "request_id": "req_abc123",
  "ip": "192.168.1.100"
}
```

### Log Retention

| Log Type | Retention |
|----------|----------|
| Authentication | 1 year |
| API access | 6 months |
| System events | 1 year |
| Error logs | 3 months |
| Audit trail | Permanent |

---

## Container Security

### Hardening Measures

| Measure | Implementation |
|---------|---------------|
| **Non-root processes** | All containers run as non-root user |
| **Read-only filesystem** | Where possible, mount as read-only |
| **Minimal base images** | `python:3.10-slim`, `alpine` variants |
| **No shell access** | Disable interactive shells in production |
| **Resource limits** | CPU/memory limits per container |
| **Health checks** | All containers have health endpoints |
| **No privileged mode** | Never `--privileged` |

### Docker Security Settings

```yaml
services:
  backend:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

---

## Operational Security

### Access Controls

| Control | Detail |
|---------|--------|
| Physical access | Server in secured location |
| Terminal access | Authorized personnel only |
| USB ports | Controlled for data transfer |
| Screen locking | Auto-lock after inactivity |
| Password policy | Minimum 12 chars, complexity requirements |

### Incident Response

```
SECURITY INCIDENT
       │
       ▼
┌──────────────┐
│ 1. DETECT    │  Audit logs, anomaly detection
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 2. CONTAIN   │  Isolate affected service
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 3. ANALYZE   │  Review logs, identify scope
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 4. REMEDIATE │  Fix vulnerability, update
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 5. REPORT    │  Document and communicate
└──────────────┘
```

---

## Security Checklist

### Pre-Deployment

- [ ] No API keys or secrets in source code
- [ ] All environment variables documented in `.env.example`
- [ ] `.env` file in `.gitignore`
- [ ] All containers run as non-root
- [ ] Database credentials use strong passwords
- [ ] TLS configured for inter-service communication
- [ ] JWT secret is cryptographically strong (256+ bits)
- [ ] Model checksums generated and stored
- [ ] Audit logging enabled for all services
- [ ] Rate limiting configured on all endpoints
- [ ] Input validation on all API endpoints
- [ ] SQL injection prevention (parameterized queries)
- [ ] CORS configured (restrict to frontend origin)
- [ ] Health check endpoints don't leak sensitive info
- [ ] Container images scanned for vulnerabilities
- [ ] No unnecessary ports exposed

### Post-Deployment

- [ ] All services accessible only from internal network
- [ ] No outbound internet connections verified
- [ ] Test authentication with invalid credentials
- [ ] Verify audit logs are recording
- [ ] Verify model integrity check on startup
- [ ] Test rate limiting under load
- [ ] Verify backup encryption
- [ ] Review access control roles
- [ ] Penetration testing (if applicable)
