# GitLab CI/CD Setup Guide

## Required GitLab CI/CD Variables

Before the pipeline can run successfully, you need to configure these variables in GitLab:

### Navigation
**Settings → CI/CD → Variables → Expand → Add Variable**

---

## 1. Deployment Variables (Required for Deploy Stage)

### `DEPLOY_HOST`
- **Value**: `fall2025-comp585-6.cs.mcgill.ca`
- **Type**: Variable
- **Protected**: ✅ Yes (only available on protected branches)
- **Masked**: ❌ No (hostname is not sensitive)
- **Description**: Production server hostname

### `DEPLOY_USER`
- **Value**: Your SSH username on the McGill server
- **Type**: Variable
- **Protected**: ✅ Yes
- **Masked**: ❌ No
- **Description**: SSH user for deployment

### `DEPLOY_SSH_KEY`
- **Value**: Your private SSH key (entire content including headers)
- **Type**: Variable
- **Protected**: ✅ Yes
- **Masked**: ✅ Yes (sensitive data)
- **Description**: SSH private key for server access
- **Format**:
  ```
  -----BEGIN OPENSSH PRIVATE KEY-----
  b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAA...
  ... (your key content) ...
  -----END OPENSSH PRIVATE KEY-----
  ```

---

## 2. Generate SSH Key Pair (if you don't have one)

On your local machine:

```powershell
# Generate new SSH key pair
ssh-keygen -t ed25519 -C "gitlab-ci-deploy" -f ~/.ssh/gitlab_deploy_key -N ""

# Copy public key to deployment server
ssh-copy-id -i ~/.ssh/gitlab_deploy_key.pub your_username@fall2025-comp585-6.cs.mcgill.ca

# Display private key (copy this to GitLab variable)
Get-Content ~/.ssh/gitlab_deploy_key
```

---

## 3. Optional Variables (Auto-configured by GitLab)

These are automatically available and don't need manual configuration:

- `CI_REGISTRY` - GitLab container registry URL
- `CI_REGISTRY_IMAGE` - Full image path for your project
- `CI_REGISTRY_USER` - Registry username (gitlab-ci-token)
- `CI_REGISTRY_PASSWORD` - Registry password (auto-generated)
- `CI_COMMIT_SHA` - Current commit hash
- `CI_COMMIT_SHORT_SHA` - Short commit hash (for image tags)
- `CI_COMMIT_BRANCH` - Current branch name
- `CI_DEFAULT_BRANCH` - Default branch (usually "main")

---

## 4. Pipeline Behavior

### On Merge Request (MR):
✅ **Runs automatically**:
- `unit_tests` - Tests `tests/unittests/`
- `integration_tests` - Tests `tests/integration/`
- `code_coverage` - Coverage report

🔧 **Manual trigger**:
- `docker_build` - Optional: Verify Docker build works

❌ **Does not run**:
- `deploy_production` - Only runs on main branch

### On Main Branch (after merge):
✅ **Runs automatically**:
- `unit_tests`
- `integration_tests`
- `code_coverage`
- `docker_build` - Builds and pushes image to GitLab registry

🔧 **Manual trigger**:
- `deploy_production` - Deploy to production server (requires approval)
- `evaluate_models` - Run offline evaluation

---

## 5. Testing the Pipeline Locally

### Test Unit Tests:
```powershell
# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/unittests/ -v
```

### Test Integration Tests:
```powershell
pytest tests/integration/ -v
```

### Test Docker Build:
```powershell
# Build image
docker build -f docker/Dockerfile -t movie-recommender:test .

# Test run
docker run -p 8080:8080 movie-recommender:test

# Health check (in another terminal)
curl http://localhost:8080/health
```

---

## 6. Deployment Checklist

Before triggering the deploy job:

- [ ] SSH key added to GitLab variables
- [ ] Public key copied to deployment server
- [ ] Docker installed on deployment server
- [ ] Port 8080 open on deployment server
- [ ] Can SSH manually: `ssh $DEPLOY_USER@$DEPLOY_HOST`
- [ ] Tests passing on main branch
- [ ] Docker image built successfully

---

## 7. Troubleshooting

### Pipeline fails at docker_build:
```
Error: Cannot connect to Docker daemon
```
**Solution**: Ensure `services: [docker:24-dind]` is present (already configured)

### Pipeline fails at deploy_production:
```
Permission denied (publickey)
```
**Solution**: 
1. Verify `DEPLOY_SSH_KEY` variable contains the complete private key
2. Ensure public key is in `~/.ssh/authorized_keys` on server
3. Test SSH manually: `ssh -i ~/.ssh/gitlab_deploy_key $DEPLOY_USER@$DEPLOY_HOST`

### Docker image not found during deployment:
```
Error: manifest unknown
```
**Solution**: Check that `docker_build` job completed successfully and pushed image

### Health check fails after deployment:
```
❌ Deployment failed - Health check failed
```
**Solution**:
1. Check container logs: `ssh $DEPLOY_USER@$DEPLOY_HOST "docker logs movie-recommender"`
2. Verify model files exist in container
3. Check if port 8080 is already in use

---

## 8. Pipeline Stages Overview

```
┌─────────────────────────────────────────────────────────┐
│                    MERGE REQUEST                        │
├─────────────────────────────────────────────────────────┤
│  Stage: test (parallel)                                 │
│    • unit_tests           ✅ Auto                       │
│    • integration_tests    ✅ Auto                       │
│    • code_coverage        ✅ Auto                       │
│    • evaluate_models      🔧 Manual (optional)          │
│                                                          │
│  Stage: build                                           │
│    • docker_build         🔧 Manual (verification only) │
│                                                          │
│  Stage: deploy                                          │
│    • (skipped on MR)      ❌                            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     MAIN BRANCH                         │
├─────────────────────────────────────────────────────────┤
│  Stage: test (parallel)                                 │
│    • unit_tests           ✅ Auto                       │
│    • integration_tests    ✅ Auto                       │
│    • code_coverage        ✅ Auto                       │
│    • evaluate_models      🔧 Manual (optional)          │
│                                                          │
│  Stage: build                                           │
│    • docker_build         ✅ Auto                       │
│                                                          │
│  Stage: deploy                                          │
│    • deploy_production    🔧 Manual (requires approval) │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Next Steps

1. **Add SSH variables** to GitLab (see section 1)
2. **Push this pipeline** to a feature branch
3. **Create a Merge Request** to test the pipeline
4. **Verify tests pass** in the MR pipeline
5. **Merge to main** after approval
6. **Manually trigger** `deploy_production` job
7. **Verify deployment** at http://fall2025-comp585-6.cs.mcgill.ca:8080/health

---

## 10. Important Notes

⚠️ **Security**:
- Always use **Protected** and **Masked** for sensitive variables (SSH keys, passwords)
- Never commit secrets to the repository
- Rotate SSH keys periodically

⚠️ **Cost**:
- Docker builds can be slow on first run (~5-10 minutes)
- Subsequent builds use cache (~1-2 minutes)
- Consider using GitLab's shared runners or set up project-specific runners

⚠️ **Testing**:
- Always test pipeline changes in a feature branch first
- Use manual triggers for deploy until confident in automation
- Keep test data in repository or use mocks for faster CI runs
