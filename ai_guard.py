#!/usr/bin/env python3
"""
Local AI Guard - Code Quality & Security Auditor
Uses Ollama with a local LLM to audit files for technical debt and security risks.
Ensures sensitive data never leaves the developer's machine.
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from datetime import datetime

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
DEFAULT_TIMEOUT = 120  # seconds

# Audit severity levels
SEVERITY_REJECT = "REJECT"
SEVERITY_FLAG = "FLAG"
SEVERITY_PASS = "PASS"


class AIGuard:
    """Local AI Guard for code auditing using Ollama."""
    
    def __init__(self, model: str = MODEL_NAME, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self.timeout = DEFAULT_TIMEOUT
        
    def _check_ollama_connection(self) -> bool:
        """Verify Ollama service is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _validate_file_exists(self, file_path: str) -> bool:
        """Check if the target file exists."""
        return Path(file_path).exists() and Path(file_path).is_file()
    
    def read_file(self, file_path: str) -> Tuple[bool, str]:
        """Read file content safely."""
        if not self._validate_file_exists(file_path):
            return False, f"File not found: {file_path}"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return True, content
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def _build_audit_prompt(self, file_path: str, content: str, file_type: str) -> str:
        """Build a comprehensive audit prompt for the AI."""
        extension = Path(file_path).suffix.lower()
        
        prompt = f"""You are a Security Expert specialized in detecting hardcoded secrets and passwords.

AUDIT FILE: {file_path}
FILE TYPE: {file_type}
DATE: {datetime.now().strftime('%Y-%m-%d')}

CRITICAL: Look for ANY of these patterns:
1. Hardcoded passwords (password=, passwd=, pwd=, pass=)
2. API Keys and tokens (api_key=, apikey=, token=, secret=, api-token=)
3. Database credentials (db_password=, database_password=, db_user=)
4. AWS/Cloud credentials (aws_secret=, aws_access_key=, azure_key=)
5. Private keys or certificates (-----BEGIN, private_key=, pem)
6. Authentication tokens (jwt=, bearer=, oauth=)
7. Encryption keys (encryption_key=, cipher_key=)
8. Any variable names containing: secret, password, passwd, pwd, key, credential, token, apikey

---CODE TO AUDIT---
{content}
---END CODE---

RESPOND WITH ONLY ONE OF THESE VERDICTS:

If you find ANY hardcoded secrets, credentials, or passwords:
REJECT: [exact line number] - [type of secret found] - [exact line of code]

If you find code issues but no secrets:
FLAG: [issue type] - [description]

If no secrets or significant issues:
PASS

Examples of responses:
- REJECT: Line 5 - Hardcoded API key detected - API_KEY=sk_test_abc123def456
- REJECT: Line 3 - Database password found - DATABASE_URL=postgresql://user:fakepassword123@localhost
- FLAG: Fragile test pattern - waitForTimeout detected
- PASS

YOUR VERDICT (respond with ONLY the verdict, nothing else):"""
        
        return prompt
    
    def audit_file(self, file_path: str) -> Dict:
        """Audit a file using Ollama's local LLM."""
        # Validate file exists
        file_exists, content = self.read_file(file_path)
        if not file_exists:
            return {
                "status": SEVERITY_REJECT,
                "error": content,
                "file": file_path,
                "timestamp": datetime.now().isoformat()
            }
        
        # Check Ollama connection
        if not self._check_ollama_connection():
            return {
                "status": SEVERITY_REJECT,
                "error": f"Ollama service not available at {self.base_url}. Please ensure Ollama is running: ollama serve",
                "file": file_path,
                "timestamp": datetime.now().isoformat()
            }
        
        # Determine file type
        extension = Path(file_path).suffix.lower()
        file_type_map = {
            '.ts': 'TypeScript/Playwright',
            '.js': 'JavaScript',
            '.py': 'Python',
            '.tsx': 'React/TypeScript',
            '.jsx': 'React/JavaScript',
            '.test.ts': 'TypeScript Test',
            '.spec.ts': 'TypeScript Test/Playwright',
            '.env': 'Environment Variables (HIGH PRIORITY FOR SECRETS)',
        }
        file_type = file_type_map.get(extension, 'Code')
        
        # Build audit prompt
        prompt = self._build_audit_prompt(file_path, content, file_type)
        
        # Call Ollama API
        try:
            print(f"\n--- Auditing File: {file_path} ---")
            print("Sending to Local Ollama LLM for analysis...\n")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,  # Lower temperature for more deterministic security checks
                    "top_p": 0.9,
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                return {
                    "status": SEVERITY_REJECT,
                    "error": f"Ollama API error: {response.status_code} - {response.text}",
                    "file": file_path,
                    "timestamp": datetime.now().isoformat()
                }
            
            response_data = response.json()
            report = response_data.get("response", "").strip()
            
            # Parse verdict from report - STRICT parsing for security
            verdict = SEVERITY_PASS
            
            if report.startswith("REJECT"):
                verdict = SEVERITY_REJECT
                print("="*60)
                print("🚨 SECURITY ALERT: SECRETS DETECTED")
                print("="*60)
                print(report)
            elif report.startswith("FLAG"):
                verdict = SEVERITY_FLAG
                print("⚠️  FLAG: Issues found but no critical secrets")
                print(report)
            else:
                print("✅ PASS: No security issues detected")
                print(report)
            
            return {
                "status": verdict,
                "file": file_path,
                "report": report,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.exceptions.Timeout:
            return {
                "status": SEVERITY_REJECT,
                "error": f"Ollama request timed out after {self.timeout}s. File audit could not complete - blocking commit for safety.",
                "file": file_path,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": SEVERITY_REJECT,
                "error": f"Unexpected error during audit: {str(e)}",
                "file": file_path,
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_audit(self, file_paths: List[str]) -> List[Dict]:
        """Audit multiple files."""
        results = []
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{len(file_paths)}] Processing: {file_path}")
            result = self.audit_file(file_path)
            results.append(result)
        return results


def print_audit_report(result: Dict):
    """Pretty print an audit result."""
    status = result.get("status", "UNKNOWN")
    file = result.get("file", "Unknown file")
    
    # Color codes for terminal
    colors = {
        SEVERITY_REJECT: '\033[91m',  # Red
        SEVERITY_FLAG: '\033[93m',    # Yellow
        SEVERITY_PASS: '\033[92m',    # Green
    }
    reset = '\033[0m'
    
    color = colors.get(status, '')
    
    print(f"\n{'='*60}")
    print(f"{color}[{status}]{reset} {file}")
    print(f"{'='*60}")
    
    if "error" in result:
        print(f"Error: {result['error']}")
    elif "report" in result:
        print(result["report"])
    
    print(f"Timestamp: {result.get('timestamp', 'N/A')}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python ai_guard.py <file_path> [file_path2] ...")
        print("\nExample:")
        print("  python ai_guard.py src/tests/existing.test.ts")
        print("  python ai_guard.py auth.ts security.ts")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    
    # Initialize AI Guard
    guard = AIGuard()
    
    print("\n" + "="*60)
    print("LOCAL AI GUARD - Code Quality & Security Auditor")
    print("="*60)
    print(f"Model: {guard.model}")
    print(f"Ollama: {guard.base_url}")
    print(f"Files to audit: {len(file_paths)}")
    print("="*60)
    
    # Audit files
    results = guard.batch_audit(file_paths)
    
    # Print reports
    for result in results:
        print_audit_report(result)
    
    # Summary
    rejects = sum(1 for r in results if r.get("status") == SEVERITY_REJECT)
    flags = sum(1 for r in results if r.get("status") == SEVERITY_FLAG)
    passes = sum(1 for r in results if r.get("status") == SEVERITY_PASS)
    
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)
    print(f"Total files audited: {len(results)}")
    print(f"✗ REJECT: {rejects}")
    print(f"⚠ FLAG: {flags}")
    print(f"✓ PASS: {passes}")
    print("="*60 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if rejects == 0 else 1)


if __name__ == "__main__":
    main()
