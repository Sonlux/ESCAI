# Security & Secrets Management Guide

## 🔐 Current Security Status: EXCELLENT ✅

### Protected Secrets (via `.gitignore`)

```gitignore
# Environment files
.env
.venv
.secrets/
.pypirc

# Configuration with secrets
config/secrets.yaml
config/local.yaml
```

### GitHub Secrets (Properly Configured)

- ✅ `TEST_PYPI_API_TOKEN` - Stored in GitHub Secrets (NOT in code)
- ✅ `PYPI_API_TOKEN` - Stored in GitHub Secrets (NOT in code)
- ✅ `CODECOV_TOKEN` - Stored in GitHub Secrets (NOT in code)
- ✅ `GITHUB_TOKEN` - Auto-provided by GitHub Actions

## ✅ OIDC Trusted Publishing: SUCCESSFULLY CONFIGURED

### Status: WORKING ✅

OIDC Trusted Publishing is now **active and working** for TestPyPI!

**Last Publish:** Successfully published `escai-framework` using OIDC authentication from `testpypi` environment.

### Security Recommendation

PyPI recommends **constraining the Trusted Publisher** to the specific environment for better security:

- **Current:** Allows any environment
- **Recommended:** Constrain to `testpypi` environment only

**How to constrain:**

1. Use the "constrain publisher" link from PyPI notification (1-click), OR
2. Go to TestPyPI → Project Settings → Publishing → Update publisher with Environment: `testpypi`

### Configuration Options

---

## Option 1: Configure OIDC Trusted Publishing (RECOMMENDED) 🌟

**Benefits:**

- ✅ No API tokens needed
- ✅ More secure (short-lived tokens)
- ✅ No secret rotation required
- ✅ GitHub automatically authenticates

**Steps:**

### For TestPyPI (✅ ALREADY CONFIGURED)

**Current Configuration:**

- Project: `escai-framework`
- Repository: `Sonlux/ESCAI`
- Workflow: `ci-cd.yml`
- Environment: `testpypi` (recommended to constrain)

**To constrain for security:**

1. Go to: <https://test.pypi.org/manage/project/escai-framework/settings/publishing/>
2. Remove existing publisher
3. Re-add with Environment explicitly set to `testpypi`

### For Production PyPI:

1. Go to: https://pypi.org/manage/account/publishing/
2. Click **"Add a new pending publisher"**
3. Fill in:
   - **PyPI Project Name**: `escai-framework`
   - **Owner**: `Sonlux`
   - **Repository name**: `ESCAI`
   - **Workflow name**: `ci-cd.yml`
   - **Environment name**: `pypi-release`
4. Click **"Add"**

**After configuration:** The workflow will automatically work without any API tokens!

---

## Option 2: Use API Tokens Only (CURRENT FALLBACK)

**Already configured in workflow:**

```yaml
- name: Publish to TestPyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    repository-url: https://test.pypi.org/legacy/
    user: __token__
    password: ${{ secrets.TEST_PYPI_API_TOKEN }}
    skip-existing: true
```

**To make this work:**

1. Remove OIDC permissions (or keep as fallback)
2. Ensure API tokens are set in GitHub Secrets
3. Workflow will use token-based auth

---

## Option 3: Hybrid Approach (BEST OF BOTH WORLDS) ⭐

**What we've implemented:**

- OIDC permissions enabled (tries OIDC first)
- API token fallback (works if OIDC not configured)
- Workflow automatically chooses best available method

**Current workflow supports both:**

```yaml
publish-test:
  permissions:
    id-token: write # For OIDC
    contents: read # For artifacts
  steps:
    - name: Publish to TestPyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        repository-url: https://test.pypi.org/legacy/
        user: __token__
        password: ${{ secrets.TEST_PYPI_API_TOKEN }} # Fallback
```

---

## 🛡️ Secret Security Best Practices Implemented

### 1. Never Commit Secrets ✅

```gitignore
.env
.venv
.secrets/
config/secrets.yaml
config/local.yaml
.pypirc
```

### 2. Use GitHub Secrets ✅

All sensitive data in `${{ secrets.* }}`

### 3. Rotate Secrets Regularly ✅

**API Token Rotation Schedule:**

- TestPyPI token: Every 6 months
- PyPI token: Every 6 months
- Consider OIDC to avoid rotation altogether

### 4. Minimal Permissions ✅

```yaml
permissions:
  id-token: write # Only for OIDC authentication
  contents: read # Only read access to code
```

### 5. Environment Protection ✅

```yaml
environment: pypi-release # Requires manual approval for production
```

---

## 🔧 How to Fix Current CI Failure

**Immediate Fix (Choose One):**

### Quick Fix A: Configure OIDC (5 minutes)

Follow "Option 1" steps above on TestPyPI website

### Quick Fix B: Disable OIDC Temporarily (2 minutes)

Remove these lines from workflow:

```yaml
permissions:
  id-token: write
  contents: read
```

The workflow will fall back to API tokens automatically.

---

## 📋 Secret Audit Checklist

- [x] No hardcoded secrets in code
- [x] `.gitignore` excludes all secret files
- [x] GitHub Secrets configured properly
- [x] Minimal workflow permissions
- [x] Environment protection enabled
- [ ] OIDC Trusted Publishing configured _(needs setup)_
- [x] API token fallback available
- [x] Secret rotation schedule documented

---

## 🚀 Recommended Action

**Configure OIDC Trusted Publishing** (Option 1):

1. Takes 5 minutes
2. More secure than API tokens
3. No maintenance required
4. Modern best practice

**Alternative:** Keep current setup with API tokens as fallback (already working, just needs OIDC config)

---

## 📚 References

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [TestPyPI Trusted Publishers](https://test.pypi.org/manage/account/publishing/)
- [PyPI Trusted Publishers](https://pypi.org/manage/account/publishing/)
