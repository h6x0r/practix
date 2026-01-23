import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'cert-ceh-prep',
	title: 'CEH Certification Practice',
	difficulty: 'hard',
	tags: ['security', 'certification', 'ceh', 'ethical-hacking', 'typescript'],
	estimatedTime: '40m',
	isPremium: true,
	youtubeUrl: '',
	description: `Practice CEH (Certified Ethical Hacker) concepts through coding.

**CEH Overview:**

CEH focuses on offensive security - thinking like an attacker to defend better.

**CEH Domains:**

1. Introduction to Ethical Hacking
2. Footprinting and Reconnaissance
3. Scanning Networks
4. Enumeration
5. Vulnerability Analysis
6. System Hacking
7. Malware Threats
8. Sniffing
9. Social Engineering
10. Denial-of-Service
11. Session Hijacking
12. Evading IDS, Firewalls, Honeypots
13. Hacking Web Servers/Applications
14. SQL Injection
15. Wireless Network Hacking
16. Mobile Platform Hacking
17. IoT and OT Hacking
18. Cloud Computing
19. Cryptography

**Your Task:**

Implement a \`PenTestSimulator\` class that simulates penetration testing concepts.`,
	initialCode: `interface Target {
  ip: string;
  hostname?: string;
  ports: number[];
  services: Record<number, string>;
  vulnerabilities: string[];
}

interface ScanResult {
  target: string;
  openPorts: number[];
  services: Record<number, string>;
  osGuess?: string;
  scanTime: number;
}

interface VulnerabilityReport {
  target: string;
  vulnerabilities: {
    id: string;
    name: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    remediation: string;
  }[];
  riskScore: number;
}

type AttackVector = 'network' | 'web' | 'social' | 'physical' | 'wireless';

class PenTestSimulator {
  private targets: Map<string, Target> = new Map();
  private discoveredInfo: Map<string, any> = new Map();

  addTarget(target: Target): void {
    // TODO: Add target to simulation
  }

  performReconnaissance(targetIp: string): {
    whois?: any;
    dns?: any;
    subdomains?: string[];
  } {
    // TODO: Simulate reconnaissance phase
    return {};
  }

  performPortScan(targetIp: string, portRange?: [number, number]): ScanResult {
    // TODO: Simulate port scanning
    return {} as ScanResult;
  }

  identifyService(port: number, banner?: string): string {
    // TODO: Identify service based on port/banner
    return '';
  }

  performVulnerabilityScan(targetIp: string): VulnerabilityReport {
    // TODO: Scan for known vulnerabilities
    return {} as VulnerabilityReport;
  }

  suggestAttackVectors(target: Target): AttackVector[] {
    // TODO: Suggest attack vectors based on open ports/services
    return [];
  }

  simulateExploit(targetIp: string, vulnerabilityId: string): {
    success: boolean;
    accessLevel?: 'user' | 'admin' | 'root';
    message: string;
  } {
    // TODO: Simulate exploiting a vulnerability
    return { success: false, message: '' };
  }

  generatePenTestReport(targetIp: string): string {
    // TODO: Generate penetration test report
    return '';
  }

  calculateRiskScore(vulnerabilities: VulnerabilityReport['vulnerabilities']): number {
    // TODO: Calculate risk score based on vulnerabilities
    return 0;
  }
}

export { PenTestSimulator, Target, ScanResult, VulnerabilityReport, AttackVector };`,
	solutionCode: `interface Target {
  ip: string;
  hostname?: string;
  ports: number[];
  services: Record<number, string>;
  vulnerabilities: string[];
}

interface ScanResult {
  target: string;
  openPorts: number[];
  services: Record<number, string>;
  osGuess?: string;
  scanTime: number;
}

interface VulnerabilityReport {
  target: string;
  vulnerabilities: {
    id: string;
    name: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    remediation: string;
  }[];
  riskScore: number;
}

type AttackVector = 'network' | 'web' | 'social' | 'physical' | 'wireless';

class PenTestSimulator {
  private targets: Map<string, Target> = new Map();
  private discoveredInfo: Map<string, any> = new Map();

  private knownVulns: Record<string, { name: string; severity: 'low' | 'medium' | 'high' | 'critical'; description: string; remediation: string }> = {
    'CVE-2021-44228': { name: 'Log4Shell', severity: 'critical', description: 'RCE in Log4j', remediation: 'Update to Log4j 2.17+' },
    'CVE-2017-0144': { name: 'EternalBlue', severity: 'critical', description: 'SMB RCE', remediation: 'Apply MS17-010 patch' },
    'CVE-2014-0160': { name: 'Heartbleed', severity: 'high', description: 'OpenSSL memory leak', remediation: 'Update OpenSSL' },
    'CVE-2019-0708': { name: 'BlueKeep', severity: 'critical', description: 'RDP RCE', remediation: 'Apply security update' },
  };

  private commonServices: Record<number, string> = {
    21: 'FTP', 22: 'SSH', 23: 'Telnet', 25: 'SMTP', 53: 'DNS',
    80: 'HTTP', 110: 'POP3', 143: 'IMAP', 443: 'HTTPS', 445: 'SMB',
    3306: 'MySQL', 3389: 'RDP', 5432: 'PostgreSQL', 8080: 'HTTP-Proxy',
  };

  addTarget(target: Target): void {
    this.targets.set(target.ip, target);
  }

  performReconnaissance(targetIp: string): {
    whois?: any;
    dns?: any;
    subdomains?: string[];
  } {
    const target = this.targets.get(targetIp);
    if (!target) return {};

    const result = {
      whois: {
        registrar: 'Example Registrar',
        created: '2020-01-01',
        nameservers: ['ns1.example.com', 'ns2.example.com'],
      },
      dns: {
        A: [targetIp],
        MX: ['mail.example.com'],
        TXT: ['v=spf1 include:_spf.google.com ~all'],
      },
      subdomains: target.hostname
        ? ['www', 'mail', 'admin', 'api'].map(s => \`\${s}.\${target.hostname}\`)
        : [],
    };

    this.discoveredInfo.set(targetIp, result);
    return result;
  }

  performPortScan(targetIp: string, portRange?: [number, number]): ScanResult {
    const target = this.targets.get(targetIp);
    const startTime = Date.now();

    if (!target) {
      return {
        target: targetIp,
        openPorts: [],
        services: {},
        scanTime: Date.now() - startTime,
      };
    }

    let portsToScan = target.ports;
    if (portRange) {
      portsToScan = target.ports.filter(p => p >= portRange[0] && p <= portRange[1]);
    }

    const services: Record<number, string> = {};
    for (const port of portsToScan) {
      services[port] = target.services[port] || this.identifyService(port);
    }

    // Guess OS based on open ports
    let osGuess: string | undefined;
    if (target.ports.includes(445) && target.ports.includes(3389)) {
      osGuess = 'Windows';
    } else if (target.ports.includes(22)) {
      osGuess = 'Linux/Unix';
    }

    return {
      target: targetIp,
      openPorts: portsToScan,
      services,
      osGuess,
      scanTime: Date.now() - startTime,
    };
  }

  identifyService(port: number, banner?: string): string {
    if (banner) {
      if (banner.includes('Apache')) return 'Apache HTTP';
      if (banner.includes('nginx')) return 'nginx';
      if (banner.includes('OpenSSH')) return 'OpenSSH';
    }
    return this.commonServices[port] || 'Unknown';
  }

  performVulnerabilityScan(targetIp: string): VulnerabilityReport {
    const target = this.targets.get(targetIp);

    if (!target) {
      return { target: targetIp, vulnerabilities: [], riskScore: 0 };
    }

    const vulnerabilities = target.vulnerabilities.map(vulnId => {
      const vuln = this.knownVulns[vulnId];
      return {
        id: vulnId,
        name: vuln?.name || 'Unknown',
        severity: vuln?.severity || 'medium' as const,
        description: vuln?.description || 'No description',
        remediation: vuln?.remediation || 'Investigate and patch',
      };
    });

    return {
      target: targetIp,
      vulnerabilities,
      riskScore: this.calculateRiskScore(vulnerabilities),
    };
  }

  suggestAttackVectors(target: Target): AttackVector[] {
    const vectors: Set<AttackVector> = new Set();

    // Network-based
    if (target.ports.some(p => [21, 22, 23, 445, 3389].includes(p))) {
      vectors.add('network');
    }

    // Web-based
    if (target.ports.some(p => [80, 443, 8080, 8443].includes(p))) {
      vectors.add('web');
    }

    // Social engineering is always an option
    vectors.add('social');

    return Array.from(vectors);
  }

  simulateExploit(targetIp: string, vulnerabilityId: string): {
    success: boolean;
    accessLevel?: 'user' | 'admin' | 'root';
    message: string;
  } {
    const target = this.targets.get(targetIp);

    if (!target) {
      return { success: false, message: 'Target not found' };
    }

    if (!target.vulnerabilities.includes(vulnerabilityId)) {
      return { success: false, message: 'Vulnerability not present on target' };
    }

    const vuln = this.knownVulns[vulnerabilityId];
    if (!vuln) {
      return { success: false, message: 'Unknown vulnerability' };
    }

    // Simulate exploit based on severity
    if (vuln.severity === 'critical') {
      return { success: true, accessLevel: 'root', message: \`Exploited \${vuln.name} - gained root access\` };
    } else if (vuln.severity === 'high') {
      return { success: true, accessLevel: 'admin', message: \`Exploited \${vuln.name} - gained admin access\` };
    }

    return { success: true, accessLevel: 'user', message: \`Exploited \${vuln.name} - gained user access\` };
  }

  generatePenTestReport(targetIp: string): string {
    const scan = this.performPortScan(targetIp);
    const vulnReport = this.performVulnerabilityScan(targetIp);
    const target = this.targets.get(targetIp);

    const lines = [
      '=== Penetration Test Report ===',
      \`Target: \${targetIp}\`,
      target?.hostname ? \`Hostname: \${target.hostname}\` : '',
      '',
      '--- Port Scan Results ---',
      \`Open Ports: \${scan.openPorts.join(', ')}\`,
      \`OS Guess: \${scan.osGuess || 'Unknown'}\`,
      '',
      '--- Vulnerability Assessment ---',
      \`Risk Score: \${vulnReport.riskScore}/100\`,
      \`Vulnerabilities Found: \${vulnReport.vulnerabilities.length}\`,
      '',
    ];

    for (const vuln of vulnReport.vulnerabilities) {
      lines.push(\`[\${vuln.severity.toUpperCase()}] \${vuln.name} (\${vuln.id})\`);
      lines.push(\`  \${vuln.description}\`);
      lines.push(\`  Remediation: \${vuln.remediation}\`);
    }

    return lines.filter(l => l !== '').join('\\n');
  }

  calculateRiskScore(vulnerabilities: VulnerabilityReport['vulnerabilities']): number {
    if (vulnerabilities.length === 0) return 0;

    const severityScores = { 'low': 10, 'medium': 30, 'high': 60, 'critical': 100 };
    let totalScore = 0;

    for (const vuln of vulnerabilities) {
      totalScore += severityScores[vuln.severity];
    }

    return Math.min(100, Math.round(totalScore / vulnerabilities.length));
  }
}

export { PenTestSimulator, Target, ScanResult, VulnerabilityReport, AttackVector };`,
	hint1: `For performPortScan, filter target's ports by range if provided, map ports to services using identifyService, and guess OS based on port patterns.`,
	hint2: `For calculateRiskScore, assign severity scores (critical=100, high=60, medium=30, low=10), average them, and cap at 100.`,
	testCode: `import { PenTestSimulator } from './solution';

const testTarget = {
  ip: '192.168.1.100',
  hostname: 'test.example.com',
  ports: [22, 80, 443, 3306],
  services: { 22: 'SSH', 80: 'HTTP', 443: 'HTTPS', 3306: 'MySQL' },
  vulnerabilities: ['CVE-2014-0160'],
};

// Test1: addTarget stores target
test('Test1', () => {
  const sim = new PenTestSimulator();
  sim.addTarget(testTarget);
  const scan = sim.performPortScan('192.168.1.100');
  expect(scan.openPorts).toContain(22);
});

// Test2: performReconnaissance returns info
test('Test2', () => {
  const sim = new PenTestSimulator();
  sim.addTarget(testTarget);
  const recon = sim.performReconnaissance('192.168.1.100');
  expect(recon.subdomains?.length).toBeGreaterThan(0);
});

// Test3: performPortScan finds open ports
test('Test3', () => {
  const sim = new PenTestSimulator();
  sim.addTarget(testTarget);
  const scan = sim.performPortScan('192.168.1.100');
  expect(scan.openPorts.length).toBe(4);
  expect(scan.services[80]).toBe('HTTP');
});

// Test4: identifyService works for known ports
test('Test4', () => {
  const sim = new PenTestSimulator();
  expect(sim.identifyService(22)).toBe('SSH');
  expect(sim.identifyService(80)).toBe('HTTP');
});

// Test5: performVulnerabilityScan finds vulns
test('Test5', () => {
  const sim = new PenTestSimulator();
  sim.addTarget(testTarget);
  const report = sim.performVulnerabilityScan('192.168.1.100');
  expect(report.vulnerabilities.length).toBe(1);
  expect(report.vulnerabilities[0].name).toBe('Heartbleed');
});

// Test6: suggestAttackVectors identifies web
test('Test6', () => {
  const sim = new PenTestSimulator();
  const vectors = sim.suggestAttackVectors(testTarget);
  expect(vectors).toContain('web');
  expect(vectors).toContain('network');
});

// Test7: simulateExploit succeeds for present vuln
test('Test7', () => {
  const sim = new PenTestSimulator();
  sim.addTarget(testTarget);
  const result = sim.simulateExploit('192.168.1.100', 'CVE-2014-0160');
  expect(result.success).toBe(true);
});

// Test8: simulateExploit fails for missing vuln
test('Test8', () => {
  const sim = new PenTestSimulator();
  sim.addTarget(testTarget);
  const result = sim.simulateExploit('192.168.1.100', 'CVE-9999-9999');
  expect(result.success).toBe(false);
});

// Test9: calculateRiskScore works correctly
test('Test9', () => {
  const sim = new PenTestSimulator();
  const score = sim.calculateRiskScore([
    { id: '1', name: 'Test', severity: 'critical', description: '', remediation: '' },
  ]);
  expect(score).toBe(100);
});

// Test10: generatePenTestReport includes all sections
test('Test10', () => {
  const sim = new PenTestSimulator();
  sim.addTarget(testTarget);
  const report = sim.generatePenTestReport('192.168.1.100');
  expect(report).toContain('Port Scan');
  expect(report).toContain('Vulnerability');
});`,
	whyItMatters: `CEH teaches you to think like an attacker - essential for building secure systems.

**Penetration Testing Methodology:**

\`\`\`
1. Reconnaissance
   └── Passive: OSINT, WHOIS, DNS
   └── Active: Port scanning, banner grabbing

2. Scanning
   └── Network mapping
   └── Vulnerability scanning
   └── Service enumeration

3. Gaining Access
   └── Exploiting vulnerabilities
   └── Password attacks
   └── Social engineering

4. Maintaining Access
   └── Backdoors
   └── Privilege escalation
   └── Persistence

5. Covering Tracks
   └── Log manipulation
   └── Timestomping
   └── Anti-forensics
\`\`\`

**Career Paths:**

| Role | Focus |
|------|-------|
| Penetration Tester | Offensive security |
| Red Team | Adversary simulation |
| Bug Bounty Hunter | Finding vulns for rewards |
| Security Researcher | Discovering new vulns |`,
	order: 1,
	translations: {
		ru: {
			title: 'Практика сертификации CEH',
			description: `Практикуйте концепции CEH (Certified Ethical Hacker) через программирование.

**Обзор CEH:**

CEH фокусируется на наступательной безопасности - мыслить как атакующий для лучшей защиты.

**Домены CEH:**

1. Разведка и сбор информации
2. Сканирование сетей
3. Анализ уязвимостей
4. Взлом систем
5. Социальная инженерия
6. Веб-атаки
7. SQL-инъекции

**Ваша задача:**

Реализуйте класс \`PenTestSimulator\`.`,
			hint1: `Для performPortScan отфильтруйте порты цели по диапазону, сопоставьте порты с сервисами и угадайте ОС по паттернам портов.`,
			hint2: `Для calculateRiskScore назначьте баллы по серьёзности (critical=100, high=60, medium=30, low=10), усредните и ограничьте 100.`,
			whyItMatters: `CEH учит думать как атакующий - важно для создания безопасных систем.`
		},
		uz: {
			title: 'CEH sertifikatsiyasi amaliyoti',
			description: `CEH (Certified Ethical Hacker) tushunchalarini dasturlash orqali mashq qiling.

**CEH haqida:**

CEH hujumkor xavfsizlikka e'tibor qaratadi - yaxshiroq himoya qilish uchun tajovuzkor kabi o'ylash.

**Sizning vazifangiz:**

\`PenTestSimulator\` klassini amalga oshiring.`,
			hint1: `performPortScan uchun maqsad portlarini diapazon bo'yicha filtrlang, portlarni xizmatlarga moslashtiring va port patternlariga qarab OS ni taxmin qiling.`,
			hint2: `calculateRiskScore uchun jiddiylik ballari bering (critical=100, high=60, medium=30, low=10), o'rtacha oling va 100 bilan cheklang.`,
			whyItMatters: `CEH tajovuzkor kabi o'ylashni o'rgatadi - xavfsiz tizimlar yaratish uchun muhim.`
		}
	}
};

export default task;
