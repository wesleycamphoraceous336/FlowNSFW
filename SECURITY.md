# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report

1. **Email**: Send details to [your-email@example.com]
2. **Expected Response Time**: Within 48 hours
3. **Disclosure Timeline**: We aim to patch within 7 days

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

### After Reporting

- You will receive an acknowledgment within 48 hours
- We will investigate and provide updates every 72 hours
- Once patched, we will coordinate disclosure timing with you
- Credit will be given in the release notes (unless you prefer anonymity)

## Security Best Practices

### When Using FlowNSFW

1. **Input Validation**: Always validate video inputs before processing
2. **Resource Limits**: Set timeouts and memory limits to prevent DoS
3. **Model Weights**: Verify checksum of downloaded weights (see releases)
4. **API Keys**: Never commit API keys or credentials to version control
5. **Access Control**: Restrict model access in production environments

### Known Limitations

- **Adversarial Examples**: Model may be vulnerable to adversarial inputs
- **Out-of-Distribution**: Performance degrades on non-standard video formats
- **Privacy**: Video frames are processed in-memory but not stored by default

### Secure Deployment Checklist

- [ ] Model runs in isolated container (Docker recommended)
- [ ] Input video size/length limits enforced
- [ ] Output logged for audit purposes
- [ ] HTTPS used for any API endpoints
- [ ] Regular security updates applied

## CVE References

No CVEs have been assigned to this project yet.

## PGP Key (Optional)

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
[Your PGP public key for encrypted vulnerability reports]
-----END PGP PUBLIC KEY BLOCK-----
```

## Acknowledgments

We thank the following security researchers for responsible disclosure:

- [Name] - [Vulnerability] - [Date]

---

**Last Updated**: 2026-06-11
