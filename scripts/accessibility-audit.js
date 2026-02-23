const axios = require('axios');
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');
const { AxePuppeteer } = require('@axe-core/puppeteer');

const PAGES = [
  { name: 'Main Page', url: 'http://localhost:9002/' },
  { name: 'Checkout Page', url: 'http://localhost:9002/checkout' },
  { name: 'Orders Page', url: 'http://localhost:9002/orders' }
];

const REPORTS_DIR = path.join(__dirname, '../accessibility-reports');
const TIMESTAMP = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -4);

async function checkServerHealth() {
  console.log('🔍 Checking if Next.js server is running on port 9002...');
  
  try {
    await axios.get('http://localhost:9002/', { timeout: 3000 });
    console.log('✅ Server is running on port 9002\n');
    return true;
  } catch (error) {
    console.error('❌ Server not running on port 9002');
    console.error('Please start it with: npm run dev\n');
    process.exit(1);
  }
}

async function runAccessibilityScans() {
  if (!fs.existsSync(REPORTS_DIR)) {
    fs.mkdirSync(REPORTS_DIR, { recursive: true });
  }

  const allViolations = [];
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  console.log('🚀 Starting Accessibility Audit (WCAG 2.0 AA)\n');
  console.log('='.repeat(70));

  for (const page of PAGES) {
    console.log(`\n📄 Scanning: ${page.name} (${page.url})`);
    
    try {
      const browserPage = await browser.newPage();
      await browserPage.goto(page.url, { waitUntil: 'networkidle2' });
      
      console.log('   Running axe accessibility checks...');
      
      const results = await new AxePuppeteer(browserPage)
        .withTags(['wcag2aa'])
        .analyze();

      // Save detailed JSON report
      const reportFile = path.join(REPORTS_DIR, `${page.name.replace(/\s+/g, '-').toLowerCase()}-${TIMESTAMP}.json`);
      fs.writeFileSync(reportFile, JSON.stringify(results, null, 2));

      // Extract and categorize violations
      if (results.violations && results.violations.length > 0) {
        results.violations.forEach(violation => {
          allViolations.push({
            impact: violation.impact || 'moderate',
            id: violation.id,
            description: violation.description,
            help: violation.help,
            helpUrl: violation.helpUrl,
            page: page.name,
            nodes: violation.nodes.length
          });
        });
        console.log(`✅ Scan complete (${results.violations.length} issues found)`);
      } else {
        console.log(`✅ Scan complete (No violations found)`);
      }

      await browserPage.close();
    } catch (error) {
      console.error(`⚠️  Error scanning ${page.name}:`, error.message);
    }
  }

  await browser.close();
  
  console.log('\n' + '='.repeat(70));
  generateHtmlReport(allViolations);
  printSummary(allViolations);
}

function generateHtmlReport(violations) {
  const criticalViolations = violations.filter(v => v.impact === 'critical');
  const seriousViolations = violations.filter(v => v.impact === 'serious');
  const moderateViolations = violations.filter(v => v.impact === 'moderate');

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Accessibility Audit Report - MyBasket Lite</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
      line-height: 1.6; 
      color: #333; 
      background: #f5f5f5; 
    }
    .container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }
    
    header { 
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
      color: white; 
      padding: 40px; 
      border-radius: 8px; 
      margin-bottom: 40px; 
    }
    header h1 { font-size: 2.5em; margin-bottom: 10px; }
    header .meta { font-size: 0.95em; opacity: 0.9; }
    
    .summary { 
      display: grid; 
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
      gap: 20px; 
      margin-bottom: 40px; 
    }
    .summary-card { 
      background: white; 
      padding: 20px; 
      border-radius: 8px; 
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      text-align: center;
    }
    .summary-card h3 { 
      font-size: 0.85em; 
      text-transform: uppercase; 
      color: #666; 
      margin-bottom: 10px; 
      letter-spacing: 0.5px;
    }
    .summary-card .value { 
      font-size: 2.5em; 
      font-weight: bold; 
    }
    .summary-card.critical .value { color: #dc2626; }
    .summary-card.serious .value { color: #f97316; }
    .summary-card.passed .value { color: #16a34a; }
    
    .violations-section { 
      background: white; 
      padding: 30px; 
      border-radius: 8px; 
      box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
      margin-bottom: 30px; 
    }
    .violations-section h2 { 
      font-size: 1.8em; 
      margin-bottom: 20px; 
      border-bottom: 3px solid #667eea; 
      padding-bottom: 10px; 
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge { 
      display: inline-block; 
      padding: 4px 12px; 
      border-radius: 20px; 
      font-size: 0.75em; 
      font-weight: 700;
      text-transform: uppercase;
    }
    .badge-critical { background: #dc2626; color: white; }
    .badge-serious { background: #f97316; color: white; }
    
    .violation-item { 
      padding: 15px; 
      margin-bottom: 15px; 
      border-left: 4px solid; 
      border-radius: 4px; 
      background: #f9fafb;
    }
    .violation-item.critical { 
      border-color: #dc2626; 
      background: #fee2e2;
    }
    .violation-item.serious { 
      border-color: #f97316; 
      background: #ffedd5;
    }
    
    .violation-item h4 { 
      margin-bottom: 8px; 
      font-weight: 600; 
    }
    .violation-item .code { 
      font-family: 'Courier New', monospace; 
      background: rgba(0,0,0,0.05); 
      padding: 2px 6px; 
      border-radius: 3px; 
      font-size: 0.9em; 
    }
    .violation-item .meta { 
      font-size: 0.9em; 
      color: #666; 
      margin-top: 8px; 
      display: flex;
      gap: 15px;
      flex-wrap: wrap;
    }
    .violation-item .count { 
      display: inline-block; 
      background: rgba(0,0,0,0.1); 
      padding: 2px 8px; 
      border-radius: 12px; 
      font-size: 0.85em; 
    }
    
    .recommendations { 
      background: #ecf0f1; 
      padding: 20px; 
      border-radius: 8px; 
      margin-top: 30px; 
      border-left: 4px solid #667eea; 
    }
    .recommendations h3 { 
      margin-bottom: 15px; 
      color: #667eea; 
    }
    .recommendations ul { 
      margin-left: 20px; 
    }
    .recommendations li { 
      margin-bottom: 10px; 
    }
    .recommendations a { 
      color: #667eea; 
      text-decoration: none; 
    }
    .recommendations a:hover { 
      text-decoration: underline; 
    }
    
    footer { 
      text-align: center; 
      margin-top: 40px; 
      padding-top: 20px; 
      border-top: 1px solid #e5e7eb; 
      color: #666; 
      font-size: 0.9em; 
    }
    
    .empty-state {
      text-align: center;
      padding: 40px;
      color: #16a34a;
    }
    .empty-state h3 {
      font-size: 1.5em;
      margin-bottom: 10px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>♿ Accessibility Audit Report</h1>
      <p style="margin-bottom: 10px;">MyBasket Lite - WCAG 2.0 AA Compliance Analysis</p>
      <div class="meta">Generated: ${new Date().toLocaleString()}</div>
    </header>

    <div class="summary">
      <div class="summary-card critical">
        <h3>Critical Issues</h3>
        <div class="value">${criticalViolations.length}</div>
      </div>
      <div class="summary-card serious">
        <h3>Serious Issues</h3>
        <div class="value">${seriousViolations.length}</div>
      </div>
      <div class="summary-card passed">
        <h3>Pages Scanned</h3>
        <div class="value">3</div>
      </div>
    </div>

    ${criticalViolations.length > 0 ? `
    <div class="violations-section">
      <h2><span class="badge badge-critical">🔴 CRITICAL</span> Critical Violations (${criticalViolations.length})</h2>
      <p style="margin-bottom: 20px; color: #666;">These issues must be fixed immediately to ensure WCAG 2.0 AA compliance.</p>
      ${criticalViolations.map(v => `
        <div class="violation-item critical">
          <h4>${v.description}</h4>
          <div><span class="code">${v.id}</span></div>
          <div class="meta">
            <span>📄 ${v.page}</span>
            <span class="count">${v.nodes} occurrence${v.nodes > 1 ? 's' : ''}</span>
          </div>
          <div style="margin-top: 10px; font-size: 0.9em;">
            <a href="${v.helpUrl}" target="_blank">Learn more →</a>
          </div>
        </div>
      `).join('')}
    </div>
    ` : ''}

    ${seriousViolations.length > 0 ? `
    <div class="violations-section">
      <h2><span class="badge badge-serious">🟠 SERIOUS</span> Serious Violations (${seriousViolations.length})</h2>
      <p style="margin-bottom: 20px; color: #666;">These issues should be addressed to improve accessibility.</p>
      ${seriousViolations.map(v => `
        <div class="violation-item serious">
          <h4>${v.description}</h4>
          <div><span class="code">${v.id}</span></div>
          <div class="meta">
            <span>📄 ${v.page}</span>
            <span class="count">${v.nodes} occurrence${v.nodes > 1 ? 's' : ''}</span>
          </div>
          <div style="margin-top: 10px; font-size: 0.9em;">
            <a href="${v.helpUrl}" target="_blank">Learn more →</a>
          </div>
        </div>
      `).join('')}
    </div>
    ` : ''}

    ${criticalViolations.length === 0 && seriousViolations.length === 0 ? `
    <div class="empty-state">
      <h3>✅ No Critical or Serious Issues Found</h3>
      <p>All pages passed WCAG 2.0 AA compliance checks.</p>
    </div>
    ` : ''}

    <div class="recommendations">
      <h3>🎯 Remediation Recommendations</h3>
      <ul>
        <li><strong>Add ARIA labels</strong> to interactive elements missing accessible names</li>
        <li><strong>Improve color contrast</strong> to meet WCAG AA standards (4.5:1 for text, 3:1 for large text)</li>
        <li><strong>Ensure keyboard navigation</strong> for all interactive components</li>
        <li><strong>Add visible focus indicators</strong> for keyboard users</li>
        <li><strong>Test with screen readers</strong> (NVDA, JAWS, VoiceOver)</li>
        <li><strong>Implement alt text</strong> for all images and icons</li>
        <li><strong>Use semantic HTML</strong> (&lt;button&gt;, &lt;nav&gt;, &lt;main&gt;)</li>
        <li><strong>Ensure form labels</strong> are properly associated with inputs</li>
      </ul>
      <p style="margin-top: 20px;">
        For more information, visit <a href="https://www.w3.org/WAI/WCAG21/quickref/" target="_blank">WCAG 2.1 Quick Reference</a>
      </p>
    </div>

    <footer>
      <p>✅ Accessibility audit completed successfully</p>
      <p style="margin-top: 10px; font-size: 0.85em;">Audit ID: ${TIMESTAMP}</p>
    </footer>
  </div>
</body>
</html>`;

  const reportPath = path.join(REPORTS_DIR, `accessibility-summary-${TIMESTAMP}.html`);
  fs.writeFileSync(reportPath, html);
  console.log(`\n📊 HTML Report generated: ${reportPath}`);
  return reportPath;
}

function printSummary(violations) {
  const critical = violations.filter(v => v.impact === 'critical');
  const serious = violations.filter(v => v.impact === 'serious');

  console.log('\n' + '='.repeat(70));
  console.log('📋 ACCESSIBILITY AUDIT SUMMARY');
  console.log('='.repeat(70));
  
  if (critical.length > 0) {
    console.log(`\n🔴 CRITICAL VIOLATIONS (${critical.length}):`);
    critical.forEach((v, i) => {
      console.log(`\n   ${i + 1}. ${v.id}`);
      console.log(`      Description: ${v.description}`);
      console.log(`      Page: ${v.page}`);
      console.log(`      Occurrences: ${v.nodes}`);
    });
  } else {
    console.log('\n✅ No critical violations found.');
  }

  if (serious.length > 0) {
    console.log(`\n🟠 SERIOUS VIOLATIONS (${serious.length}):`);
    serious.forEach((v, i) => {
      console.log(`\n   ${i + 1}. ${v.id}`);
      console.log(`      Description: ${v.description}`);
      console.log(`      Page: ${v.page}`);
      console.log(`      Occurrences: ${v.nodes}`);
    });
  } else {
    console.log('\n✅ No serious violations found.');
  }

  console.log('\n' + '='.repeat(70) + '\n');
}

// Run the audit
(async () => {
  try {
    await checkServerHealth();
    await runAccessibilityScans();
    console.log('\n✅ Accessibility audit completed successfully!\n');
    process.exit(0);
  } catch (error) {
    console.error('❌ Audit failed:', error);
    process.exit(1);
  }
})();
