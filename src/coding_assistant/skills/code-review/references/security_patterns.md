# Security Patterns for Code Review

## Input Validation
- **Always validate**: All user input (form data, query params, headers, cookies, file uploads)
- **Whitelist over blacklist**: Allow known good, reject everything else
- **Type checking**: Ensure expected types (int, string, etc.)
- **Length limits**: Prevent buffer overflows, DoS
- **Canonicalization**: Normalize paths, URLs before validation

## Authentication & Authorization
- **Never trust client**: Authentication decisions must happen server-side
- **Session management**: Secure cookies (HttpOnly, Secure, SameSite)
- **JWT best practices**: Short expiration, proper signature verification
- **RBAC/ABAC**: Enforce role-based or attribute-based access control
- **Least privilege**: Minimal permissions required

## SQL Injection
- **Prepared statements**: Always use parameterized queries
- **ORM safety**: Ensure ORM uses parameterization, not string concatenation
- **Stored procedures**: Use with caution (can still be vulnerable)
- **No dynamic SQL**: Avoid building SQL strings with user input

## XSS (Cross-Site Scripting)
- **Output encoding**: Encode based on context (HTML, JS, CSS, URL)
- **Content Security Policy (CSP)**: Implement strict CSP headers
- **Sanitize HTML**: Use DOMPurify or similar for rich content
- **Angular/React**: Built-in protections still need careful usage

## CSRF (Cross-Site Request Forgery)
- **Anti-CSRF tokens**: Use synchronizer token pattern
- **SameSite cookies**: Set to Strict or Lax
- **Origin/Referer checks**: Validate request origin

## Sensitive Data
- **Logging**: Never log passwords, tokens, PII
- **Encryption**: Encrypt at rest, TLS in transit
- **Secrets management**: Use vaults (HashiCorp Vault, AWS Secrets Manager)
- **Hardcoded secrets**: Scan with `git secrets`, `truffleHog`

## File Uploads
- **File type validation**: Check MIME type, extension
- **Virus scanning**: Scan uploaded files
- **Storage location**: Outside web root, proper permissions
- **Filename sanitization**: Prevent path traversal

## Dependencies
- **Vulnerability scanning**: Regular `npm audit`, `safety`, `gem audit`
- **Update strategy**: Patch known vulnerabilities promptly
- **License compliance**: Check licenses of dependencies

## Error Handling
- **Don't expose stack traces**: To users in production
- **Generic error messages**: Don't leak system info
- **Logging**: Log errors for monitoring, but not sensitive data

## Cryptography
- **Don't roll your own crypto**: Use established libraries
- **Key management**: Proper rotation, storage
- **Hashing**: Use bcrypt/scrypt/argon2 for passwords
- **Randomness**: Use cryptographically secure random generators

## API Security
- **Rate limiting**: Prevent brute force, DoS
- **API keys**: Rotate regularly, scope permissions
- **OAuth**: Use PKCE for public clients
- **CORS**: Configure appropriately, don't use wildcard in production

## Cloud & Infrastructure
- **IAM roles**: Least privilege
- **Security groups**: Restrict inbound/outbound
- **Environment isolation**: Separate dev/staging/prod credentials
- **Infrastructure as Code**: Security scanning (tfsec, checkov)

## Common Vulnerabilities (OWASP Top 10 2023)
1. **Broken Access Control**
2. **Cryptographic Failures**
3. **Injection**
4. **Insecure Design**
5. **Security Misconfiguration**
6. **Vulnerable and Outdated Components**
7. **Identification and Authentication Failures**
8. **Software and Data Integrity Failures**
9. **Security Logging and Monitoring Failures**
10. **Server-Side Request Forgery**

## Tools to Use
- **SAST**: Semgrep, SonarQube, CodeQL
- **DAST**: OWASP ZAP, Burp Suite
- **SCA**: Snyk, Dependabot, OWASP Dependency Check
- **Secrets scanning**: GitGuardian, TruffleHog

## Quick Questions to Ask
1. Where does data originate? Can it be malicious?
2. Is data validated at every trust boundary?
3. Are permissions checked at the source of data access?
4. Are dependencies up to date and scanned?
5. Are secrets properly managed?
6. Are errors handled without leaking information?