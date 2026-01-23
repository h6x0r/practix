import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'interview-security-architecture',
	title: 'Security Architecture Design',
	difficulty: 'hard',
	tags: ['security', 'interview', 'architecture', 'design', 'typescript'],
	estimatedTime: '45m',
	isPremium: true,
	youtubeUrl: '',
	description: `Practice security architecture design - critical for senior security and architect roles.

**Architecture Principles:**

1. **Defense in Depth** - Multiple layers of security controls
2. **Least Privilege** - Minimum necessary access
3. **Zero Trust** - Never trust, always verify
4. **Secure by Design** - Security built-in, not bolted-on
5. **Fail Secure** - Safe defaults when failures occur

**Your Task:**

Implement a \`SecurityArchitectureReviewer\` class that helps evaluate and design secure architectures.`,
	initialCode: `interface ArchitectureComponent {
  id: string;
  name: string;
  type: 'frontend' | 'backend' | 'database' | 'cache' | 'queue' | 'external' | 'cdn' | 'loadbalancer';
  environment: 'public' | 'dmz' | 'internal' | 'restricted';
  dataClassification: 'public' | 'internal' | 'confidential' | 'restricted';
  controls: SecurityControl[];
}

interface SecurityControl {
  id: string;
  type: 'authentication' | 'authorization' | 'encryption' | 'logging' | 'firewall' | 'waf' | 'ids' | 'dlp';
  implementation: string;
  effectiveness: 'low' | 'medium' | 'high';
}

interface DataFlow {
  source: string;
  destination: string;
  protocol: string;
  encrypted: boolean;
  authenticated: boolean;
  dataTypes: string[];
}

interface Architecture {
  name: string;
  components: ArchitectureComponent[];
  dataFlows: DataFlow[];
  trustBoundaries: { name: string; componentIds: string[] }[];
}

interface SecurityGap {
  component: string;
  issue: string;
  risk: 'low' | 'medium' | 'high' | 'critical';
  recommendation: string;
  effort: 'low' | 'medium' | 'high';
}

interface ArchitectureReview {
  overallScore: number;
  gaps: SecurityGap[];
  strengths: string[];
  complianceIssues: string[];
  recommendations: string[];
}

class SecurityArchitectureReviewer {
  private architecture: Architecture | null = null;

  // Architecture Setup
  createArchitecture(name: string): Architecture {
    // TODO: Create a new architecture
    return {} as Architecture;
  }

  addComponent(component: ArchitectureComponent): void {
    // TODO: Add a component to the architecture
  }

  addDataFlow(flow: DataFlow): void {
    // TODO: Add a data flow
  }

  addTrustBoundary(name: string, componentIds: string[]): void {
    // TODO: Add a trust boundary
  }

  // Architecture Review
  reviewArchitecture(): ArchitectureReview {
    // TODO: Perform comprehensive security review
    return {} as ArchitectureReview;
  }

  checkDefenseInDepth(): {
    score: number;
    layers: { name: string; controls: string[] }[];
    gaps: string[];
  } {
    // TODO: Verify defense in depth implementation
    return { score: 0, layers: [], gaps: [] };
  }

  checkZeroTrust(): {
    compliant: boolean;
    issues: string[];
    recommendations: string[];
  } {
    // TODO: Check zero trust compliance
    return { compliant: false, issues: [], recommendations: [] };
  }

  // Gap Analysis
  identifySecurityGaps(): SecurityGap[] {
    // TODO: Identify security gaps in the architecture
    return [];
  }

  prioritizeRemediation(gaps: SecurityGap[]): SecurityGap[] {
    // TODO: Prioritize gaps for remediation
    return [];
  }

  // Compliance Check
  checkCompliance(framework: 'pci-dss' | 'hipaa' | 'soc2' | 'gdpr'): {
    compliant: boolean;
    issues: string[];
    requirements: { id: string; status: 'met' | 'partial' | 'not_met'; gap?: string }[];
  } {
    // TODO: Check compliance against framework
    return { compliant: false, issues: [], requirements: [] };
  }

  // Design Recommendations
  suggestImprovements(): {
    component: string;
    currentState: string;
    recommendedState: string;
    rationale: string;
    priority: 'high' | 'medium' | 'low';
  }[] {
    // TODO: Suggest architecture improvements
    return [];
  }

  generateSecurityRequirements(): {
    category: string;
    requirements: string[];
  }[] {
    // TODO: Generate security requirements based on architecture
    return [];
  }

  // Interview Helpers
  getDesignQuestion(level: 'junior' | 'senior' | 'architect'): {
    question: string;
    context: string;
    expectedElements: string[];
    followUps: string[];
  } {
    // TODO: Generate architecture design interview question
    return {} as any;
  }

  evaluateDesign(proposedComponents: string[], expectedElements: string[]): {
    score: number;
    matched: string[];
    missing: string[];
    feedback: string;
  } {
    // TODO: Evaluate candidate's design proposal
    return {} as any;
  }
}

// Test your implementation
const reviewer = new SecurityArchitectureReviewer();

// Test 1: Create architecture
const arch = reviewer.createArchitecture('E-Commerce Platform');
console.log('Test 1 - Architecture created:', arch.name === 'E-Commerce Platform');

// Test 2: Add components
reviewer.addComponent({
  id: 'c1',
  name: 'Web Application',
  type: 'frontend',
  environment: 'public',
  dataClassification: 'public',
  controls: [
    { id: 'ctrl1', type: 'waf', implementation: 'AWS WAF', effectiveness: 'high' },
  ],
});
reviewer.addComponent({
  id: 'c2',
  name: 'API Server',
  type: 'backend',
  environment: 'internal',
  dataClassification: 'confidential',
  controls: [
    { id: 'ctrl2', type: 'authentication', implementation: 'OAuth2', effectiveness: 'high' },
  ],
});
console.log('Test 2 - Components added:', true);

// Test 3: Add data flow
reviewer.addDataFlow({
  source: 'c1',
  destination: 'c2',
  protocol: 'HTTPS',
  encrypted: true,
  authenticated: true,
  dataTypes: ['user credentials', 'payment data'],
});
console.log('Test 3 - Data flow added:', true);

// Test 4: Review architecture
const review = reviewer.reviewArchitecture();
console.log('Test 4 - Review complete:', 'overallScore' in review);

// Test 5: Check defense in depth
const depthCheck = reviewer.checkDefenseInDepth();
console.log('Test 5 - Depth check:', depthCheck.layers.length >= 0);

// Test 6: Check zero trust
const zeroTrust = reviewer.checkZeroTrust();
console.log('Test 6 - Zero trust check:', 'compliant' in zeroTrust);

// Test 7: Identify gaps
const gaps = reviewer.identifySecurityGaps();
console.log('Test 7 - Gaps identified:', Array.isArray(gaps));

// Test 8: Check compliance
const compliance = reviewer.checkCompliance('pci-dss');
console.log('Test 8 - Compliance check:', 'requirements' in compliance);

// Test 9: Suggest improvements
const improvements = reviewer.suggestImprovements();
console.log('Test 9 - Improvements:', Array.isArray(improvements));

// Test 10: Design question
const question = reviewer.getDesignQuestion('senior');
console.log('Test 10 - Design question:', question.expectedElements?.length > 0);`,
	solutionCode: `interface ArchitectureComponent {
  id: string;
  name: string;
  type: 'frontend' | 'backend' | 'database' | 'cache' | 'queue' | 'external' | 'cdn' | 'loadbalancer';
  environment: 'public' | 'dmz' | 'internal' | 'restricted';
  dataClassification: 'public' | 'internal' | 'confidential' | 'restricted';
  controls: SecurityControl[];
}

interface SecurityControl {
  id: string;
  type: 'authentication' | 'authorization' | 'encryption' | 'logging' | 'firewall' | 'waf' | 'ids' | 'dlp';
  implementation: string;
  effectiveness: 'low' | 'medium' | 'high';
}

interface DataFlow {
  source: string;
  destination: string;
  protocol: string;
  encrypted: boolean;
  authenticated: boolean;
  dataTypes: string[];
}

interface Architecture {
  name: string;
  components: ArchitectureComponent[];
  dataFlows: DataFlow[];
  trustBoundaries: { name: string; componentIds: string[] }[];
}

interface SecurityGap {
  component: string;
  issue: string;
  risk: 'low' | 'medium' | 'high' | 'critical';
  recommendation: string;
  effort: 'low' | 'medium' | 'high';
}

interface ArchitectureReview {
  overallScore: number;
  gaps: SecurityGap[];
  strengths: string[];
  complianceIssues: string[];
  recommendations: string[];
}

class SecurityArchitectureReviewer {
  private architecture: Architecture | null = null;

  createArchitecture(name: string): Architecture {
    this.architecture = {
      name,
      components: [],
      dataFlows: [],
      trustBoundaries: [],
    };
    return this.architecture;
  }

  addComponent(component: ArchitectureComponent): void {
    if (!this.architecture) return;
    this.architecture.components.push(component);
  }

  addDataFlow(flow: DataFlow): void {
    if (!this.architecture) return;
    this.architecture.dataFlows.push(flow);
  }

  addTrustBoundary(name: string, componentIds: string[]): void {
    if (!this.architecture) return;
    this.architecture.trustBoundaries.push({ name, componentIds });
  }

  reviewArchitecture(): ArchitectureReview {
    if (!this.architecture) {
      return { overallScore: 0, gaps: [], strengths: [], complianceIssues: [], recommendations: [] };
    }

    const gaps = this.identifySecurityGaps();
    const depthCheck = this.checkDefenseInDepth();
    const zeroTrustCheck = this.checkZeroTrust();

    const strengths: string[] = [];
    const complianceIssues: string[] = [];
    const recommendations: string[] = [];

    // Identify strengths
    const hasWaf = this.architecture.components.some(c => c.controls.some(ctrl => ctrl.type === 'waf'));
    const hasAuth = this.architecture.components.some(c => c.controls.some(ctrl => ctrl.type === 'authentication'));
    const hasEncryption = this.architecture.dataFlows.every(f => f.encrypted);

    if (hasWaf) strengths.push('Web Application Firewall deployed');
    if (hasAuth) strengths.push('Authentication controls in place');
    if (hasEncryption) strengths.push('All data flows encrypted');
    if (depthCheck.score >= 70) strengths.push('Good defense in depth');

    // Identify compliance issues
    if (!hasEncryption) complianceIssues.push('Not all data in transit is encrypted');
    if (!zeroTrustCheck.compliant) complianceIssues.push('Zero trust principles not fully implemented');

    // Generate recommendations
    if (gaps.length > 0) {
      const criticalGaps = gaps.filter(g => g.risk === 'critical');
      if (criticalGaps.length > 0) {
        recommendations.push(\`Address \${criticalGaps.length} critical security gaps immediately\`);
      }
    }

    recommendations.push(...depthCheck.gaps.map(g => \`Add security layer: \${g}\`));
    recommendations.push(...zeroTrustCheck.recommendations.slice(0, 3));

    // Calculate overall score
    let score = 100;
    score -= gaps.filter(g => g.risk === 'critical').length * 20;
    score -= gaps.filter(g => g.risk === 'high').length * 10;
    score -= gaps.filter(g => g.risk === 'medium').length * 5;
    score = Math.max(0, Math.min(100, score));

    return {
      overallScore: score,
      gaps,
      strengths,
      complianceIssues,
      recommendations,
    };
  }

  checkDefenseInDepth(): {
    score: number;
    layers: { name: string; controls: string[] }[];
    gaps: string[];
  } {
    if (!this.architecture) return { score: 0, layers: [], gaps: [] };

    const layers: { name: string; controls: string[] }[] = [];
    const gaps: string[] = [];

    // Check network layer
    const networkControls = this.architecture.components
      .flatMap(c => c.controls)
      .filter(ctrl => ['firewall', 'waf', 'ids'].includes(ctrl.type))
      .map(ctrl => ctrl.implementation);

    if (networkControls.length > 0) {
      layers.push({ name: 'Network', controls: networkControls });
    } else {
      gaps.push('Network security controls (firewall, WAF, IDS)');
    }

    // Check application layer
    const appControls = this.architecture.components
      .flatMap(c => c.controls)
      .filter(ctrl => ['authentication', 'authorization'].includes(ctrl.type))
      .map(ctrl => ctrl.implementation);

    if (appControls.length > 0) {
      layers.push({ name: 'Application', controls: appControls });
    } else {
      gaps.push('Application security controls (authentication, authorization)');
    }

    // Check data layer
    const hasEncryption = this.architecture.dataFlows.every(f => f.encrypted);
    const hasDlp = this.architecture.components.some(c => c.controls.some(ctrl => ctrl.type === 'dlp'));

    if (hasEncryption) {
      layers.push({ name: 'Data', controls: ['Encryption in transit'] });
    } else {
      gaps.push('Data encryption in transit');
    }

    if (hasDlp) {
      layers[layers.length - 1]?.controls.push('Data Loss Prevention');
    }

    // Check monitoring layer
    const hasLogging = this.architecture.components.some(c => c.controls.some(ctrl => ctrl.type === 'logging'));

    if (hasLogging) {
      layers.push({ name: 'Monitoring', controls: ['Logging'] });
    } else {
      gaps.push('Monitoring and logging');
    }

    const score = Math.round((layers.length / 4) * 100);

    return { score, layers, gaps };
  }

  checkZeroTrust(): {
    compliant: boolean;
    issues: string[];
    recommendations: string[];
  } {
    if (!this.architecture) return { compliant: false, issues: [], recommendations: [] };

    const issues: string[] = [];
    const recommendations: string[] = [];

    // Check: Verify explicitly (all data flows authenticated)
    const unauthenticatedFlows = this.architecture.dataFlows.filter(f => !f.authenticated);
    if (unauthenticatedFlows.length > 0) {
      issues.push(\`\${unauthenticatedFlows.length} unauthenticated data flows\`);
      recommendations.push('Implement authentication for all data flows');
    }

    // Check: Least privilege access
    const hasAuthz = this.architecture.components.some(c => c.controls.some(ctrl => ctrl.type === 'authorization'));
    if (!hasAuthz) {
      issues.push('No authorization controls found');
      recommendations.push('Implement role-based or attribute-based access control');
    }

    // Check: Assume breach (monitoring, segmentation)
    const hasMonitoring = this.architecture.components.some(c => c.controls.some(ctrl => ctrl.type === 'logging' || ctrl.type === 'ids'));
    if (!hasMonitoring) {
      issues.push('Insufficient monitoring for breach detection');
      recommendations.push('Implement comprehensive logging and intrusion detection');
    }

    // Check: Network segmentation
    const publicComponents = this.architecture.components.filter(c => c.environment === 'public');
    const internalComponents = this.architecture.components.filter(c => c.environment === 'internal' || c.environment === 'restricted');

    if (publicComponents.length > 0 && internalComponents.length > 0 && this.architecture.trustBoundaries.length === 0) {
      issues.push('No trust boundaries defined between public and internal zones');
      recommendations.push('Define trust boundaries and implement network segmentation');
    }

    // Check: Encryption everywhere
    const unencryptedFlows = this.architecture.dataFlows.filter(f => !f.encrypted);
    if (unencryptedFlows.length > 0) {
      issues.push(\`\${unencryptedFlows.length} unencrypted data flows\`);
      recommendations.push('Encrypt all data in transit');
    }

    return {
      compliant: issues.length === 0,
      issues,
      recommendations,
    };
  }

  identifySecurityGaps(): SecurityGap[] {
    if (!this.architecture) return [];

    const gaps: SecurityGap[] = [];

    // Check each component
    for (const component of this.architecture.components) {
      // Public components need WAF
      if (component.environment === 'public') {
        const hasWaf = component.controls.some(c => c.type === 'waf');
        if (!hasWaf) {
          gaps.push({
            component: component.name,
            issue: 'Public-facing component without WAF protection',
            risk: 'high',
            recommendation: 'Deploy Web Application Firewall',
            effort: 'medium',
          });
        }
      }

      // Components with confidential data need encryption and DLP
      if (component.dataClassification === 'confidential' || component.dataClassification === 'restricted') {
        const hasDlp = component.controls.some(c => c.type === 'dlp');
        if (!hasDlp) {
          gaps.push({
            component: component.name,
            issue: \`\${component.dataClassification} data without DLP controls\`,
            risk: component.dataClassification === 'restricted' ? 'critical' : 'high',
            recommendation: 'Implement Data Loss Prevention',
            effort: 'high',
          });
        }
      }

      // All components should have logging
      const hasLogging = component.controls.some(c => c.type === 'logging');
      if (!hasLogging) {
        gaps.push({
          component: component.name,
          issue: 'No logging controls',
          risk: 'medium',
          recommendation: 'Implement comprehensive logging',
          effort: 'low',
        });
      }

      // Backend components need authentication
      if (['backend', 'database'].includes(component.type)) {
        const hasAuth = component.controls.some(c => c.type === 'authentication');
        if (!hasAuth) {
          gaps.push({
            component: component.name,
            issue: 'Backend component without authentication',
            risk: 'critical',
            recommendation: 'Implement strong authentication',
            effort: 'medium',
          });
        }
      }
    }

    // Check data flows
    for (const flow of this.architecture.dataFlows) {
      const sensitiveData = flow.dataTypes.some(t =>
        t.toLowerCase().includes('credential') ||
        t.toLowerCase().includes('payment') ||
        t.toLowerCase().includes('personal') ||
        t.toLowerCase().includes('pii')
      );

      if (sensitiveData && !flow.encrypted) {
        gaps.push({
          component: \`Flow: \${flow.source} → \${flow.destination}\`,
          issue: 'Sensitive data transmitted unencrypted',
          risk: 'critical',
          recommendation: 'Enable TLS encryption for this data flow',
          effort: 'low',
        });
      }

      if (sensitiveData && !flow.authenticated) {
        gaps.push({
          component: \`Flow: \${flow.source} → \${flow.destination}\`,
          issue: 'Sensitive data flow without authentication',
          risk: 'high',
          recommendation: 'Implement mutual TLS or API authentication',
          effort: 'medium',
        });
      }
    }

    return gaps;
  }

  prioritizeRemediation(gaps: SecurityGap[]): SecurityGap[] {
    const riskScore = { critical: 4, high: 3, medium: 2, low: 1 };
    const effortScore = { low: 3, medium: 2, high: 1 }; // Lower effort = higher priority

    return [...gaps].sort((a, b) => {
      const scoreA = riskScore[a.risk] * 2 + effortScore[a.effort];
      const scoreB = riskScore[b.risk] * 2 + effortScore[b.effort];
      return scoreB - scoreA;
    });
  }

  checkCompliance(framework: 'pci-dss' | 'hipaa' | 'soc2' | 'gdpr'): {
    compliant: boolean;
    issues: string[];
    requirements: { id: string; status: 'met' | 'partial' | 'not_met'; gap?: string }[];
  } {
    if (!this.architecture) return { compliant: false, issues: [], requirements: [] };

    const requirements: { id: string; status: 'met' | 'partial' | 'not_met'; gap?: string }[] = [];
    const issues: string[] = [];

    const frameworkRequirements: Record<string, { id: string; check: () => boolean; gap: string }[]> = {
      'pci-dss': [
        {
          id: 'PCI-DSS 3.4',
          check: () => this.architecture!.dataFlows.every(f => f.encrypted),
          gap: 'Card data must be encrypted in transit',
        },
        {
          id: 'PCI-DSS 8.3',
          check: () => this.architecture!.components.some(c => c.controls.some(ctrl => ctrl.type === 'authentication')),
          gap: 'Secure authentication required',
        },
        {
          id: 'PCI-DSS 10.1',
          check: () => this.architecture!.components.some(c => c.controls.some(ctrl => ctrl.type === 'logging')),
          gap: 'Audit logging required',
        },
        {
          id: 'PCI-DSS 6.6',
          check: () => this.architecture!.components.some(c => c.controls.some(ctrl => ctrl.type === 'waf')),
          gap: 'WAF required for public-facing apps',
        },
      ],
      'hipaa': [
        {
          id: 'HIPAA 164.312(a)',
          check: () => this.architecture!.components.some(c => c.controls.some(ctrl => ctrl.type === 'authentication')),
          gap: 'Unique user identification required',
        },
        {
          id: 'HIPAA 164.312(e)',
          check: () => this.architecture!.dataFlows.every(f => f.encrypted),
          gap: 'PHI must be encrypted in transit',
        },
        {
          id: 'HIPAA 164.312(b)',
          check: () => this.architecture!.components.some(c => c.controls.some(ctrl => ctrl.type === 'logging')),
          gap: 'Audit controls required',
        },
      ],
      'soc2': [
        {
          id: 'SOC2 CC6.1',
          check: () => this.architecture!.components.some(c => c.controls.some(ctrl => ctrl.type === 'authentication')),
          gap: 'Logical access controls required',
        },
        {
          id: 'SOC2 CC6.7',
          check: () => this.architecture!.dataFlows.every(f => f.encrypted),
          gap: 'Data transmission must be protected',
        },
        {
          id: 'SOC2 CC7.2',
          check: () => this.architecture!.components.some(c => c.controls.some(ctrl => ctrl.type === 'ids' || ctrl.type === 'logging')),
          gap: 'Security monitoring required',
        },
      ],
      'gdpr': [
        {
          id: 'GDPR Art. 32',
          check: () => this.architecture!.dataFlows.every(f => f.encrypted),
          gap: 'Appropriate encryption required',
        },
        {
          id: 'GDPR Art. 32',
          check: () => this.architecture!.components.some(c => c.controls.some(ctrl => ctrl.type === 'authorization')),
          gap: 'Access controls required',
        },
        {
          id: 'GDPR Art. 33',
          check: () => this.architecture!.components.some(c => c.controls.some(ctrl => ctrl.type === 'logging')),
          gap: 'Breach detection capabilities required',
        },
      ],
    };

    const checks = frameworkRequirements[framework] || [];

    for (const check of checks) {
      const met = check.check();
      requirements.push({
        id: check.id,
        status: met ? 'met' : 'not_met',
        gap: met ? undefined : check.gap,
      });

      if (!met) {
        issues.push(\`\${check.id}: \${check.gap}\`);
      }
    }

    return {
      compliant: issues.length === 0,
      issues,
      requirements,
    };
  }

  suggestImprovements(): {
    component: string;
    currentState: string;
    recommendedState: string;
    rationale: string;
    priority: 'high' | 'medium' | 'low';
  }[] {
    if (!this.architecture) return [];

    const improvements: {
      component: string;
      currentState: string;
      recommendedState: string;
      rationale: string;
      priority: 'high' | 'medium' | 'low';
    }[] = [];

    for (const component of this.architecture.components) {
      // Suggest WAF for public components
      if (component.environment === 'public' && !component.controls.some(c => c.type === 'waf')) {
        improvements.push({
          component: component.name,
          currentState: 'No WAF protection',
          recommendedState: 'WAF with OWASP Core Rule Set',
          rationale: 'Protect against common web attacks',
          priority: 'high',
        });
      }

      // Suggest IDS for internal components
      if (component.environment === 'internal' && !component.controls.some(c => c.type === 'ids')) {
        improvements.push({
          component: component.name,
          currentState: 'No intrusion detection',
          recommendedState: 'Network-based IDS/IPS',
          rationale: 'Detect lateral movement and anomalies',
          priority: 'medium',
        });
      }

      // Suggest encryption for databases
      if (component.type === 'database' && component.dataClassification !== 'public') {
        improvements.push({
          component: component.name,
          currentState: 'Encryption status unknown',
          recommendedState: 'Encryption at rest with managed keys',
          rationale: 'Protect stored data',
          priority: 'high',
        });
      }
    }

    return improvements;
  }

  generateSecurityRequirements(): {
    category: string;
    requirements: string[];
  }[] {
    if (!this.architecture) return [];

    return [
      {
        category: 'Authentication',
        requirements: [
          'All APIs must require authentication',
          'Implement MFA for privileged access',
          'Use OAuth 2.0 or OpenID Connect for user authentication',
          'Session tokens must expire after 30 minutes of inactivity',
        ],
      },
      {
        category: 'Authorization',
        requirements: [
          'Implement role-based access control (RBAC)',
          'Follow principle of least privilege',
          'All access decisions must be logged',
          'Service accounts must have minimal permissions',
        ],
      },
      {
        category: 'Data Protection',
        requirements: [
          'All data in transit must use TLS 1.2 or higher',
          'Sensitive data at rest must be encrypted',
          'PII must be masked in logs',
          'Implement data retention policies',
        ],
      },
      {
        category: 'Monitoring',
        requirements: [
          'All security events must be logged',
          'Implement centralized log management',
          'Set up alerting for security anomalies',
          'Retain logs for minimum 90 days',
        ],
      },
      {
        category: 'Network Security',
        requirements: [
          'Implement network segmentation',
          'Deploy WAF for public-facing applications',
          'All internal traffic should be encrypted',
          'Implement egress filtering',
        ],
      },
    ];
  }

  getDesignQuestion(level: 'junior' | 'senior' | 'architect'): {
    question: string;
    context: string;
    expectedElements: string[];
    followUps: string[];
  } {
    const questions = {
      junior: {
        question: 'How would you secure a simple web application with a database?',
        context: 'The application handles user login and stores personal information.',
        expectedElements: [
          'HTTPS/TLS for encryption',
          'Authentication for users',
          'Input validation',
          'Password hashing',
          'Firewall',
        ],
        followUps: [
          'What hashing algorithm would you use for passwords?',
          'How would you protect against SQL injection?',
          'Where would you place the firewall?',
        ],
      },
      senior: {
        question: 'Design a secure architecture for an e-commerce platform handling payment data.',
        context: 'The platform must be PCI-DSS compliant and handle 100k transactions daily.',
        expectedElements: [
          'Network segmentation (DMZ, internal zones)',
          'WAF for public-facing components',
          'Token vault for payment data',
          'Mutual TLS between services',
          'Centralized logging and SIEM',
          'Load balancer with DDoS protection',
          'Database encryption at rest',
          'Access control and authentication',
        ],
        followUps: [
          'How would you handle key management?',
          'What would be your disaster recovery strategy?',
          'How do you ensure PCI-DSS scope is minimized?',
        ],
      },
      architect: {
        question: 'Design a zero-trust architecture for a financial services company with on-premise and multi-cloud presence.',
        context: 'The company has 10,000 employees, handles sensitive financial data, and has acquired companies with different tech stacks.',
        expectedElements: [
          'Identity-centric security (IAM federation)',
          'Micro-segmentation across environments',
          'Software-defined perimeter',
          'Continuous verification and authorization',
          'Device trust and posture assessment',
          'Encrypted east-west traffic',
          'Centralized policy management',
          'Security orchestration and automation',
          'Comprehensive visibility and analytics',
          'Data classification and DLP',
        ],
        followUps: [
          'How would you handle the transition from traditional network security?',
          'What metrics would you use to measure zero trust maturity?',
          'How do you handle legacy systems that cannot support modern authentication?',
        ],
      },
    };

    return questions[level];
  }

  evaluateDesign(proposedComponents: string[], expectedElements: string[]): {
    score: number;
    matched: string[];
    missing: string[];
    feedback: string;
  } {
    const matched: string[] = [];
    const missing: string[] = [];

    const proposedLower = proposedComponents.map(p => p.toLowerCase());

    for (const expected of expectedElements) {
      const expectedWords = expected.toLowerCase().split(/\s+/).filter(w => w.length > 3);
      const isMatched = proposedLower.some(p =>
        expectedWords.filter(word => p.includes(word)).length >= Math.ceil(expectedWords.length / 3)
      );

      if (isMatched) {
        matched.push(expected);
      } else {
        missing.push(expected);
      }
    }

    const score = Math.round((matched.length / expectedElements.length) * 100);

    let feedback: string;
    if (score >= 80) {
      feedback = 'Excellent design covering most security aspects. Consider the missing elements for a complete solution.';
    } else if (score >= 60) {
      feedback = 'Good foundation but missing some important security controls. Review defense in depth principles.';
    } else if (score >= 40) {
      feedback = 'Basic understanding shown but significant gaps in security coverage. Study security architecture patterns.';
    } else {
      feedback = 'The design needs substantial improvement. Focus on fundamental security principles and common patterns.';
    }

    return { score, matched, missing, feedback };
  }
}

// Test your implementation
const reviewer = new SecurityArchitectureReviewer();

// Test 1: Create architecture
const arch = reviewer.createArchitecture('E-Commerce Platform');
console.log('Test 1 - Architecture created:', arch.name === 'E-Commerce Platform');

// Test 2: Add components
reviewer.addComponent({
  id: 'c1',
  name: 'Web Application',
  type: 'frontend',
  environment: 'public',
  dataClassification: 'public',
  controls: [
    { id: 'ctrl1', type: 'waf', implementation: 'AWS WAF', effectiveness: 'high' },
  ],
});
reviewer.addComponent({
  id: 'c2',
  name: 'API Server',
  type: 'backend',
  environment: 'internal',
  dataClassification: 'confidential',
  controls: [
    { id: 'ctrl2', type: 'authentication', implementation: 'OAuth2', effectiveness: 'high' },
  ],
});
console.log('Test 2 - Components added:', true);

// Test 3: Add data flow
reviewer.addDataFlow({
  source: 'c1',
  destination: 'c2',
  protocol: 'HTTPS',
  encrypted: true,
  authenticated: true,
  dataTypes: ['user credentials', 'payment data'],
});
console.log('Test 3 - Data flow added:', true);

// Test 4: Review architecture
const review = reviewer.reviewArchitecture();
console.log('Test 4 - Review complete:', 'overallScore' in review);

// Test 5: Check defense in depth
const depthCheck = reviewer.checkDefenseInDepth();
console.log('Test 5 - Depth check:', depthCheck.layers.length >= 0);

// Test 6: Check zero trust
const zeroTrust = reviewer.checkZeroTrust();
console.log('Test 6 - Zero trust check:', 'compliant' in zeroTrust);

// Test 7: Identify gaps
const gaps = reviewer.identifySecurityGaps();
console.log('Test 7 - Gaps identified:', Array.isArray(gaps));

// Test 8: Check compliance
const compliance = reviewer.checkCompliance('pci-dss');
console.log('Test 8 - Compliance check:', 'requirements' in compliance);

// Test 9: Suggest improvements
const improvements = reviewer.suggestImprovements();
console.log('Test 9 - Improvements:', Array.isArray(improvements));

// Test 10: Design question
const question = reviewer.getDesignQuestion('senior');
console.log('Test 10 - Design question:', question.expectedElements?.length > 0);`,
	testCode: `import { describe, it, expect, beforeEach } from 'vitest';

interface ArchitectureComponent {
  id: string;
  name: string;
  type: 'frontend' | 'backend' | 'database' | 'cache' | 'queue' | 'external' | 'cdn' | 'loadbalancer';
  environment: 'public' | 'dmz' | 'internal' | 'restricted';
  dataClassification: 'public' | 'internal' | 'confidential' | 'restricted';
  controls: { id: string; type: string; implementation: string; effectiveness: 'low' | 'medium' | 'high' }[];
}

interface DataFlow {
  source: string;
  destination: string;
  protocol: string;
  encrypted: boolean;
  authenticated: boolean;
  dataTypes: string[];
}

interface Architecture {
  name: string;
  components: ArchitectureComponent[];
  dataFlows: DataFlow[];
  trustBoundaries: { name: string; componentIds: string[] }[];
}

interface SecurityGap {
  component: string;
  issue: string;
  risk: 'low' | 'medium' | 'high' | 'critical';
  recommendation: string;
  effort: 'low' | 'medium' | 'high';
}

class SecurityArchitectureReviewer {
  private architecture: Architecture | null = null;

  createArchitecture(name: string): Architecture {
    this.architecture = { name, components: [], dataFlows: [], trustBoundaries: [] };
    return this.architecture;
  }

  addComponent(component: ArchitectureComponent): void {
    if (!this.architecture) return;
    this.architecture.components.push(component);
  }

  addDataFlow(flow: DataFlow): void {
    if (!this.architecture) return;
    this.architecture.dataFlows.push(flow);
  }

  addTrustBoundary(name: string, componentIds: string[]): void {
    if (!this.architecture) return;
    this.architecture.trustBoundaries.push({ name, componentIds });
  }

  reviewArchitecture() {
    if (!this.architecture) return { overallScore: 0, gaps: [], strengths: [], complianceIssues: [], recommendations: [] };
    const gaps = this.identifySecurityGaps();
    let score = 100 - gaps.filter(g => g.risk === 'critical').length * 20 - gaps.filter(g => g.risk === 'high').length * 10;
    return { overallScore: Math.max(0, score), gaps, strengths: ['Components defined'], complianceIssues: [], recommendations: ['Review gaps'] };
  }

  checkDefenseInDepth() {
    if (!this.architecture) return { score: 0, layers: [], gaps: [] };
    const layers: { name: string; controls: string[] }[] = [];
    const gaps: string[] = [];

    const hasNetwork = this.architecture.components.some(c => c.controls.some(ctrl => ['firewall', 'waf', 'ids'].includes(ctrl.type)));
    if (hasNetwork) layers.push({ name: 'Network', controls: ['WAF'] });
    else gaps.push('Network controls');

    const hasApp = this.architecture.components.some(c => c.controls.some(ctrl => ['authentication', 'authorization'].includes(ctrl.type)));
    if (hasApp) layers.push({ name: 'Application', controls: ['Auth'] });
    else gaps.push('Application controls');

    return { score: Math.round((layers.length / 4) * 100), layers, gaps };
  }

  checkZeroTrust() {
    if (!this.architecture) return { compliant: false, issues: [], recommendations: [] };
    const issues: string[] = [];
    const recommendations: string[] = [];

    const unauth = this.architecture.dataFlows.filter(f => !f.authenticated);
    if (unauth.length > 0) { issues.push('Unauthenticated flows'); recommendations.push('Add authentication'); }

    const unenc = this.architecture.dataFlows.filter(f => !f.encrypted);
    if (unenc.length > 0) { issues.push('Unencrypted flows'); recommendations.push('Add encryption'); }

    return { compliant: issues.length === 0, issues, recommendations };
  }

  identifySecurityGaps(): SecurityGap[] {
    if (!this.architecture) return [];
    const gaps: SecurityGap[] = [];

    for (const c of this.architecture.components) {
      if (c.environment === 'public' && !c.controls.some(ctrl => ctrl.type === 'waf')) {
        gaps.push({ component: c.name, issue: 'No WAF', risk: 'high', recommendation: 'Add WAF', effort: 'medium' });
      }
      if (!c.controls.some(ctrl => ctrl.type === 'logging')) {
        gaps.push({ component: c.name, issue: 'No logging', risk: 'medium', recommendation: 'Add logging', effort: 'low' });
      }
    }

    for (const f of this.architecture.dataFlows) {
      if (!f.encrypted && f.dataTypes.some(t => t.toLowerCase().includes('credential'))) {
        gaps.push({ component: \`\${f.source} → \${f.destination}\`, issue: 'Unencrypted sensitive data', risk: 'critical', recommendation: 'Enable TLS', effort: 'low' });
      }
    }

    return gaps;
  }

  prioritizeRemediation(gaps: SecurityGap[]): SecurityGap[] {
    const riskScore = { critical: 4, high: 3, medium: 2, low: 1 };
    return [...gaps].sort((a, b) => riskScore[b.risk] - riskScore[a.risk]);
  }

  checkCompliance(framework: 'pci-dss' | 'hipaa' | 'soc2' | 'gdpr') {
    if (!this.architecture) return { compliant: false, issues: [], requirements: [] };
    const issues: string[] = [];
    const requirements: { id: string; status: 'met' | 'partial' | 'not_met'; gap?: string }[] = [];

    const hasEncryption = this.architecture.dataFlows.every(f => f.encrypted);
    const hasAuth = this.architecture.components.some(c => c.controls.some(ctrl => ctrl.type === 'authentication'));
    const hasLogging = this.architecture.components.some(c => c.controls.some(ctrl => ctrl.type === 'logging'));

    requirements.push({ id: \`\${framework.toUpperCase()}-Encryption\`, status: hasEncryption ? 'met' : 'not_met', gap: hasEncryption ? undefined : 'Encryption required' });
    requirements.push({ id: \`\${framework.toUpperCase()}-Auth\`, status: hasAuth ? 'met' : 'not_met', gap: hasAuth ? undefined : 'Authentication required' });
    requirements.push({ id: \`\${framework.toUpperCase()}-Logging\`, status: hasLogging ? 'met' : 'not_met', gap: hasLogging ? undefined : 'Logging required' });

    if (!hasEncryption) issues.push('Missing encryption');
    if (!hasAuth) issues.push('Missing authentication');
    if (!hasLogging) issues.push('Missing logging');

    return { compliant: issues.length === 0, issues, requirements };
  }

  suggestImprovements() {
    if (!this.architecture) return [];
    return this.architecture.components
      .filter(c => c.environment === 'public' && !c.controls.some(ctrl => ctrl.type === 'waf'))
      .map(c => ({ component: c.name, currentState: 'No WAF', recommendedState: 'Add WAF', rationale: 'Web protection', priority: 'high' as const }));
  }

  generateSecurityRequirements() {
    return [
      { category: 'Authentication', requirements: ['Require auth for all APIs', 'Use MFA'] },
      { category: 'Data Protection', requirements: ['Encrypt all data in transit', 'Encrypt data at rest'] },
    ];
  }

  getDesignQuestion(level: 'junior' | 'senior' | 'architect') {
    const questions = {
      junior: { question: 'How to secure a web app?', context: 'Simple app', expectedElements: ['HTTPS', 'Auth', 'Firewall'], followUps: ['What about passwords?'] },
      senior: { question: 'Design secure e-commerce', context: 'PCI-DSS', expectedElements: ['WAF', 'Segmentation', 'Encryption', 'Logging'], followUps: ['Key management?'] },
      architect: { question: 'Zero trust for enterprise', context: 'Multi-cloud', expectedElements: ['IAM', 'Micro-segmentation', 'SDP', 'Continuous verification'], followUps: ['Legacy systems?'] },
    };
    return questions[level];
  }

  evaluateDesign(proposed: string[], expected: string[]) {
    const matched = proposed.filter(p => expected.some(e => e.toLowerCase().includes(p.toLowerCase().split(' ')[0])));
    const missing = expected.filter(e => !proposed.some(p => e.toLowerCase().includes(p.toLowerCase().split(' ')[0])));
    const score = Math.round((matched.length / expected.length) * 100);
    return { score, matched, missing, feedback: score >= 80 ? 'Excellent' : 'Needs improvement' };
  }
}

describe('SecurityArchitectureReviewer', () => {
  let reviewer: SecurityArchitectureReviewer;

  beforeEach(() => {
    reviewer = new SecurityArchitectureReviewer();
  });

  it('should create and configure architecture', () => {
    const arch = reviewer.createArchitecture('Test App');

    expect(arch.name).toBe('Test App');
    expect(arch.components).toHaveLength(0);
  });

  it('should add components and data flows', () => {
    reviewer.createArchitecture('Test');
    reviewer.addComponent({
      id: 'c1', name: 'Web', type: 'frontend', environment: 'public',
      dataClassification: 'public', controls: [{ id: 'ctrl1', type: 'waf', implementation: 'WAF', effectiveness: 'high' }],
    });
    reviewer.addDataFlow({ source: 'c1', destination: 'c2', protocol: 'HTTPS', encrypted: true, authenticated: true, dataTypes: [] });

    const review = reviewer.reviewArchitecture();
    expect(review.overallScore).toBeGreaterThan(0);
  });

  it('should check defense in depth', () => {
    reviewer.createArchitecture('Test');
    reviewer.addComponent({
      id: 'c1', name: 'API', type: 'backend', environment: 'internal',
      dataClassification: 'internal', controls: [
        { id: 'ctrl1', type: 'authentication', implementation: 'OAuth', effectiveness: 'high' },
        { id: 'ctrl2', type: 'firewall', implementation: 'FW', effectiveness: 'high' },
      ],
    });

    const check = reviewer.checkDefenseInDepth();

    expect(check.layers.length).toBeGreaterThan(0);
  });

  it('should check zero trust compliance', () => {
    reviewer.createArchitecture('Test');
    reviewer.addDataFlow({ source: 'a', destination: 'b', protocol: 'HTTP', encrypted: false, authenticated: false, dataTypes: [] });

    const check = reviewer.checkZeroTrust();

    expect(check.compliant).toBe(false);
    expect(check.issues.length).toBeGreaterThan(0);
  });

  it('should identify security gaps', () => {
    reviewer.createArchitecture('Test');
    reviewer.addComponent({
      id: 'c1', name: 'Public Web', type: 'frontend', environment: 'public',
      dataClassification: 'public', controls: [],
    });

    const gaps = reviewer.identifySecurityGaps();

    expect(gaps.some(g => g.issue.includes('WAF') || g.issue.includes('logging'))).toBe(true);
  });

  it('should check compliance frameworks', () => {
    reviewer.createArchitecture('Test');
    reviewer.addComponent({
      id: 'c1', name: 'Server', type: 'backend', environment: 'internal',
      dataClassification: 'confidential', controls: [
        { id: 'ctrl1', type: 'authentication', implementation: 'OAuth', effectiveness: 'high' },
        { id: 'ctrl2', type: 'logging', implementation: 'SIEM', effectiveness: 'high' },
      ],
    });
    reviewer.addDataFlow({ source: 'c1', destination: 'c2', protocol: 'HTTPS', encrypted: true, authenticated: true, dataTypes: [] });

    const compliance = reviewer.checkCompliance('pci-dss');

    expect(compliance.requirements.length).toBeGreaterThan(0);
  });

  it('should provide design interview questions', () => {
    const junior = reviewer.getDesignQuestion('junior');
    const senior = reviewer.getDesignQuestion('senior');
    const architect = reviewer.getDesignQuestion('architect');

    expect(junior.expectedElements.length).toBeGreaterThan(0);
    expect(senior.expectedElements.length).toBeGreaterThan(junior.expectedElements.length);
    expect(architect.expectedElements.length).toBeGreaterThan(senior.expectedElements.length);
  });

  it('should evaluate design proposals', () => {
    const expected = ['HTTPS', 'Authentication', 'Firewall', 'Logging'];
    const good = ['HTTPS encryption', 'OAuth authentication', 'Network firewall'];
    const bad = ['Some component'];

    const goodEval = reviewer.evaluateDesign(good, expected);
    const badEval = reviewer.evaluateDesign(bad, expected);

    expect(goodEval.score).toBeGreaterThan(badEval.score);
    expect(goodEval.matched.length).toBeGreaterThan(badEval.matched.length);
  });

  it('should suggest improvements', () => {
    reviewer.createArchitecture('Test');
    reviewer.addComponent({
      id: 'c1', name: 'Public App', type: 'frontend', environment: 'public',
      dataClassification: 'public', controls: [],
    });

    const improvements = reviewer.suggestImprovements();

    expect(improvements.some(i => i.recommendedState.includes('WAF'))).toBe(true);
  });

  it('should generate security requirements', () => {
    reviewer.createArchitecture('Test');
    const requirements = reviewer.generateSecurityRequirements();

    expect(requirements.length).toBeGreaterThan(0);
    expect(requirements[0].requirements.length).toBeGreaterThan(0);
  });
});`,
	hint1:
		'For defense in depth, check for security controls at multiple layers: network (firewall, WAF), application (authentication, authorization), data (encryption), and monitoring (logging, IDS). A gap at any layer reduces the overall score.',
	hint2:
		'For compliance checks, map specific requirements to security controls: PCI-DSS requires encryption, logging, WAF; HIPAA requires encryption and access controls; GDPR requires encryption and breach detection. Check if each requirement is met by existing controls.',
	whyItMatters: `Security architecture skills are essential for senior roles and are heavily tested in architect interviews:

**Why Architecture Matters:**
- **Foundation of Security** - Good architecture prevents entire classes of vulnerabilities
- **Cost Effective** - Design decisions are cheaper to change than code
- **Compliance** - Architecture determines regulatory compliance feasibility

**Key Concepts to Master:**
- **Defense in Depth** - No single point of failure
- **Zero Trust** - Verify explicitly, least privilege, assume breach
- **Secure by Design** - Security built in from the start
- **Data Flow Analysis** - Understanding how data moves through the system

**Interview Expectations:**
- Draw and explain architecture diagrams
- Identify security controls for each component
- Justify design decisions with threat analysis
- Address compliance requirements

**Real-World Impact:**
- **Twitter (2020)**: Social engineering bypassed architecture controls - $120k in Bitcoin stolen
- **Marriott (2020)**: Third-party access led to 5.2M records exposed
- **Zoom (2020)**: Encryption architecture issues led to "Zoom bombing"

Strong architecture skills demonstrate strategic thinking and readiness for leadership roles.`,
	order: 6,
	translations: {
		ru: {
			title: 'Проектирование архитектуры безопасности',
			description: `Практикуйте проектирование архитектуры безопасности - критично для старших ролей и архитекторов.

**Принципы архитектуры:**

1. **Глубокая защита** - Множество слоёв контролей безопасности
2. **Минимум привилегий** - Только необходимый доступ
3. **Нулевое доверие** - Никогда не доверять, всегда проверять
4. **Безопасность по дизайну** - Безопасность встроена, не добавлена
5. **Безопасный сбой** - Безопасные значения по умолчанию при ошибках

**Ваша задача:**

Реализуйте класс \`SecurityArchitectureReviewer\`, который помогает оценивать и проектировать безопасные архитектуры.`,
			hint1:
				'Для глубокой защиты проверьте контроли на нескольких уровнях: сеть (firewall, WAF), приложение (аутентификация), данные (шифрование), мониторинг (логирование, IDS).',
			hint2:
				'Для проверки соответствия сопоставьте требования с контролями: PCI-DSS требует шифрование, логирование, WAF; HIPAA требует шифрование и контроль доступа.',
			whyItMatters: `Навыки архитектуры безопасности необходимы для старших ролей:

**Почему архитектура важна:**
- **Основа безопасности** - Хорошая архитектура предотвращает целые классы уязвимостей
- **Экономичность** - Решения по дизайну дешевле менять, чем код
- **Соответствие** - Архитектура определяет возможность соответствия регуляторам

**Ключевые концепции:**
- **Глубокая защита** - Нет единой точки отказа
- **Нулевое доверие** - Явная проверка, минимум привилегий, предположение о компрометации
- **Безопасность по дизайну** - Безопасность с самого начала
- **Анализ потоков данных** - Понимание движения данных через систему

Сильные навыки архитектуры демонстрируют стратегическое мышление и готовность к лидерским ролям.`,
		},
		uz: {
			title: 'Xavfsizlik arxitekturasini loyihalash',
			description: `Xavfsizlik arxitekturasini loyihalashni mashq qiling - katta va arxitektor rollari uchun muhim.

**Arxitektura tamoyillari:**

1. **Chuqur himoya** - Bir nechta xavfsizlik nazorati qatlamlari
2. **Minimal imtiyozlar** - Faqat zarur kirish
3. **Nol ishonch** - Hech qachon ishonmang, har doim tekshiring
4. **Dizayn bo'yicha xavfsizlik** - Xavfsizlik o'rnatilgan, qo'shilmagan
5. **Xavfsiz xato** - Xatolar yuz berganda xavfsiz standart qiymatlar

**Vazifangiz:**

Xavfsiz arxitekturalarni baholash va loyihalashga yordam beradigan \`SecurityArchitectureReviewer\` klassini yarating.`,
			hint1:
				"Chuqur himoya uchun bir nechta qatlamlarda nazoratlarni tekshiring: tarmoq (firewall, WAF), ilova (autentifikatsiya), ma'lumotlar (shifrlash), monitoring (logging, IDS).",
			hint2:
				"Muvofiqlik tekshiruvi uchun talablarni nazoratlar bilan moslashtiring: PCI-DSS shifrlash, logging, WAF talab qiladi; HIPAA shifrlash va kirish nazoratini talab qiladi.",
			whyItMatters: `Xavfsizlik arxitekturasi ko'nikmalari katta rollar uchun zarur:

**Nima uchun arxitektura muhim:**
- **Xavfsizlik asosi** - Yaxshi arxitektura butun zaiflik sinflarining oldini oladi
- **Tejamkorlik** - Dizayn qarorlarini o'zgartirish koddan arzonroq
- **Muvofiqlik** - Arxitektura regulyator muvofiqligining imkoniyatini belgilaydi

**Asosiy tushunchalar:**
- **Chuqur himoya** - Yagona xato nuqtasi yo'q
- **Nol ishonch** - Aniq tekshirish, minimal imtiyozlar, buzilishni taxmin qilish
- **Dizayn bo'yicha xavfsizlik** - Boshidanoq o'rnatilgan xavfsizlik
- **Ma'lumot oqimi tahlili** - Ma'lumotlarning tizim orqali harakatini tushunish

Kuchli arxitektura ko'nikmalari strategik fikrlash va yetakchilik rollariga tayyorlikni ko'rsatadi.`,
		},
	},
};

export default task;
