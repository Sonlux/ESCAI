# TestPyPI Upload Instructions

## Prerequisites

1. Create account at https://test.pypi.org/account/register/
2. Verify your email address
3. Generate API token at https://test.pypi.org/manage/account/#api-tokens

## Upload Command

```bash
python -m twine upload --repository testpypi dist/*
```

## When prompted for API token:

- Enter the token you generated from TestPyPI (starts with `pypi-`)
- The token should have scope for the entire account or specific to escai-framework

## Expected Output

```
Uploading distributions to https://test.pypi.org/legacy/
Uploading escai_framework-0.2.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uploading escai_framework-0.2.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

View at:
https://test.pypi.org/project/escai-framework/0.2.0/
```

## Verification Steps

1. Visit https://test.pypi.org/project/escai-framework/
2. Check that version 0.2.0 appears
3. Verify metadata displays correctly
4. Check that description, license, and dependencies are shown properly
