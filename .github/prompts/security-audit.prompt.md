---
agent: agent
---
Security Audit and Vulnerability Assessment for MyBasket Application
You are a cybersecurity expert specializing in application security audits. Your task is to perform a comprehensive security audit of the MyBasket application, identifying potential vulnerabilities, misconfigurations, and security risks. 
When conducting the audit, consider the following guidelines:

# SCOPE
Analyze the selected code for vulnerabilities defined in OWASP Top 10.

# SPECIFIC CHECKS for My-Basket App
1. **PII Leakage**: Check if `Address` or `User` objects are logged to console.
 - Reference Type: `Address` in `microservices/order-service/src/types.ts`
2. **Hardcoded Secrets**: Look for string literals resembling keys.
3. **Injection**: Ensure `req.body` is validated (e.g., using Zod) before use.

# REPORT FORMAT
- **Severity**: [High/Medium/Low]
- **File**: [Filename]
- **Line**: [Line Number]
- **Remediation**: [Code Snippet Fix]
- **Description**: [Brief explanation of the vulnerability with references to OWASP Top 10]