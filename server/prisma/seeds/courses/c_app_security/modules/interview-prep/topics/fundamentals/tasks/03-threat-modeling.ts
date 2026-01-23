import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'interview-threat-modeling',
	title: 'Threat Modeling Interview Practice',
	difficulty: 'hard',
	tags: ['security', 'interview', 'threat-modeling', 'stride', 'typescript'],
	estimatedTime: '40m',
	isPremium: true,
	youtubeUrl: '',
	description: `Practice threat modeling - a critical skill for security architecture interviews.

**Threat Modeling Frameworks:**

1. **STRIDE** - Spoofing, Tampering, Repudiation, Info Disclosure, DoS, Elevation
2. **DREAD** - Damage, Reproducibility, Exploitability, Affected Users, Discoverability
3. **PASTA** - Process for Attack Simulation and Threat Analysis
4. **Attack Trees** - Hierarchical threat representation

**Your Task:**

Implement a \`ThreatModelingExercise\` class that helps practice threat modeling methodology.`,
	initialCode: `interface SystemComponent {
  id: string;
  name: string;
  type: 'user' | 'process' | 'datastore' | 'external' | 'network';
  trustLevel: 'trusted' | 'semi-trusted' | 'untrusted';
  dataClassification?: 'public' | 'internal' | 'confidential' | 'restricted';
}

interface DataFlow {
  id: string;
  source: string;
  destination: string;
  protocol: string;
  dataType: string;
  encrypted: boolean;
  authenticated: boolean;
}

interface Threat {
  id: string;
  category: 'spoofing' | 'tampering' | 'repudiation' | 'information_disclosure' | 'denial_of_service' | 'elevation_of_privilege';
  title: string;
  description: string;
  affectedComponent: string;
  affectedFlow?: string;
  likelihood: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
  mitigations: string[];
}

interface ThreatModel {
  id: string;
  name: string;
  description: string;
  components: SystemComponent[];
  dataFlows: DataFlow[];
  threats: Threat[];
  trustBoundaries: { id: string; name: string; components: string[] }[];
}

class ThreatModelingExercise {
  private model: ThreatModel | null = null;

  // Model Creation
  createModel(name: string, description: string): ThreatModel {
    // TODO: Create a new threat model
    return {} as ThreatModel;
  }

  addComponent(component: SystemComponent): void {
    // TODO: Add a component to the model
  }

  addDataFlow(flow: DataFlow): void {
    // TODO: Add a data flow between components
  }

  addTrustBoundary(name: string, componentIds: string[]): void {
    // TODO: Add a trust boundary
  }

  // STRIDE Analysis
  analyzeSTRIDE(componentId: string): Threat[] {
    // TODO: Generate potential STRIDE threats for a component
    return [];
  }

  analyzeDataFlowThreats(flowId: string): Threat[] {
    // TODO: Analyze threats for a specific data flow
    return [];
  }

  // Risk Assessment
  calculateDREAD(threat: Threat): {
    damage: number;
    reproducibility: number;
    exploitability: number;
    affectedUsers: number;
    discoverability: number;
    totalScore: number;
  } {
    // TODO: Calculate DREAD score for a threat
    return {} as any;
  }

  prioritizeThreats(): Threat[] {
    // TODO: Return threats sorted by risk (likelihood × impact)
    return [];
  }

  // Interview Questions
  generateInterviewQuestions(): {
    question: string;
    expectedAnswer: string;
    category: string;
  }[] {
    // TODO: Generate interview questions based on the model
    return [];
  }

  evaluateThreatIdentification(identifiedThreats: string[]): {
    score: number;
    missed: string[];
    falsePositives: string[];
  } {
    // TODO: Evaluate how well threats were identified
    return { score: 0, missed: [], falsePositives: [] };
  }

  // Reporting
  generateReport(): {
    summary: string;
    componentAnalysis: { component: string; threatCount: number; highRisk: number }[];
    topThreats: Threat[];
    recommendations: string[];
  } {
    // TODO: Generate a threat modeling report
    return {} as any;
  }

  suggestMitigations(threatId: string): string[] {
    // TODO: Suggest mitigations for a specific threat
    return [];
  }
}

// Test your implementation
const exercise = new ThreatModelingExercise();

// Test 1: Create model
const model = exercise.createModel('E-Commerce App', 'Online shopping platform');
console.log('Test 1 - Model created:', model.name === 'E-Commerce App');

// Test 2: Add component
exercise.addComponent({
  id: 'c1',
  name: 'Web Browser',
  type: 'user',
  trustLevel: 'untrusted',
});
exercise.addComponent({
  id: 'c2',
  name: 'API Server',
  type: 'process',
  trustLevel: 'trusted',
  dataClassification: 'confidential',
});
console.log('Test 2 - Components added:', true);

// Test 3: Add data flow
exercise.addDataFlow({
  id: 'f1',
  source: 'c1',
  destination: 'c2',
  protocol: 'HTTPS',
  dataType: 'User credentials',
  encrypted: true,
  authenticated: false,
});
console.log('Test 3 - Data flow added:', true);

// Test 4: STRIDE analysis
const strideThreats = exercise.analyzeSTRIDE('c1');
console.log('Test 4 - STRIDE analysis:', strideThreats.length > 0);

// Test 5: Data flow threats
const flowThreats = exercise.analyzeDataFlowThreats('f1');
console.log('Test 5 - Flow threats:', flowThreats.length > 0);

// Test 6: Calculate DREAD
const dreadScore = exercise.calculateDREAD(strideThreats[0] || { id: 't1', category: 'spoofing', title: 'Test', description: '', affectedComponent: 'c1', likelihood: 'high', impact: 'high', mitigations: [] });
console.log('Test 6 - DREAD calculated:', dreadScore.totalScore >= 0);

// Test 7: Prioritize threats
const prioritized = exercise.prioritizeThreats();
console.log('Test 7 - Threats prioritized:', Array.isArray(prioritized));

// Test 8: Generate interview questions
const questions = exercise.generateInterviewQuestions();
console.log('Test 8 - Questions generated:', questions.length > 0);

// Test 9: Evaluate threat identification
const evalResult = exercise.evaluateThreatIdentification(['spoofing attack', 'SQL injection']);
console.log('Test 9 - Evaluation done:', 'score' in evalResult);

// Test 10: Generate report
const report = exercise.generateReport();
console.log('Test 10 - Report generated:', 'summary' in report);`,
	solutionCode: `interface SystemComponent {
  id: string;
  name: string;
  type: 'user' | 'process' | 'datastore' | 'external' | 'network';
  trustLevel: 'trusted' | 'semi-trusted' | 'untrusted';
  dataClassification?: 'public' | 'internal' | 'confidential' | 'restricted';
}

interface DataFlow {
  id: string;
  source: string;
  destination: string;
  protocol: string;
  dataType: string;
  encrypted: boolean;
  authenticated: boolean;
}

interface Threat {
  id: string;
  category: 'spoofing' | 'tampering' | 'repudiation' | 'information_disclosure' | 'denial_of_service' | 'elevation_of_privilege';
  title: string;
  description: string;
  affectedComponent: string;
  affectedFlow?: string;
  likelihood: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
  mitigations: string[];
}

interface ThreatModel {
  id: string;
  name: string;
  description: string;
  components: SystemComponent[];
  dataFlows: DataFlow[];
  threats: Threat[];
  trustBoundaries: { id: string; name: string; components: string[] }[];
}

class ThreatModelingExercise {
  private model: ThreatModel | null = null;

  createModel(name: string, description: string): ThreatModel {
    this.model = {
      id: \`model-\${Date.now()}\`,
      name,
      description,
      components: [],
      dataFlows: [],
      threats: [],
      trustBoundaries: [],
    };
    return this.model;
  }

  addComponent(component: SystemComponent): void {
    if (!this.model) return;
    this.model.components.push(component);
  }

  addDataFlow(flow: DataFlow): void {
    if (!this.model) return;
    this.model.dataFlows.push(flow);
  }

  addTrustBoundary(name: string, componentIds: string[]): void {
    if (!this.model) return;
    this.model.trustBoundaries.push({
      id: \`tb-\${Date.now()}\`,
      name,
      components: componentIds,
    });
  }

  analyzeSTRIDE(componentId: string): Threat[] {
    if (!this.model) return [];

    const component = this.model.components.find(c => c.id === componentId);
    if (!component) return [];

    const threats: Threat[] = [];
    let threatCounter = 1;

    // STRIDE category definitions based on component type
    const strideCategories: { category: Threat['category']; title: string; description: string; mitigations: string[] }[] = [];

    // Spoofing - applies to users and external entities
    if (component.type === 'user' || component.type === 'external') {
      strideCategories.push({
        category: 'spoofing',
        title: \`Identity Spoofing of \${component.name}\`,
        description: \`An attacker could impersonate \${component.name} to gain unauthorized access\`,
        mitigations: ['Implement strong authentication', 'Use MFA', 'Certificate-based authentication'],
      });
    }

    // Tampering - applies to data stores and processes
    if (component.type === 'datastore' || component.type === 'process') {
      strideCategories.push({
        category: 'tampering',
        title: \`Data Tampering at \${component.name}\`,
        description: \`An attacker could modify data in \${component.name}\`,
        mitigations: ['Input validation', 'Integrity checks', 'Digital signatures'],
      });
    }

    // Repudiation - applies to all components that perform actions
    if (component.type !== 'network') {
      strideCategories.push({
        category: 'repudiation',
        title: \`Repudiation at \${component.name}\`,
        description: \`Actions performed by \${component.name} could be denied\`,
        mitigations: ['Audit logging', 'Non-repudiation controls', 'Digital signatures'],
      });
    }

    // Information Disclosure - applies to components with sensitive data
    if (component.dataClassification === 'confidential' || component.dataClassification === 'restricted') {
      strideCategories.push({
        category: 'information_disclosure',
        title: \`Information Disclosure from \${component.name}\`,
        description: \`Sensitive data in \${component.name} could be exposed\`,
        mitigations: ['Encryption at rest', 'Access controls', 'Data masking'],
      });
    }

    // Denial of Service - applies to processes and networks
    if (component.type === 'process' || component.type === 'network') {
      strideCategories.push({
        category: 'denial_of_service',
        title: \`Denial of Service against \${component.name}\`,
        description: \`\${component.name} could be made unavailable\`,
        mitigations: ['Rate limiting', 'Resource quotas', 'Load balancing', 'DDoS protection'],
      });
    }

    // Elevation of Privilege - applies to processes and users
    if (component.type === 'process' || component.type === 'user') {
      strideCategories.push({
        category: 'elevation_of_privilege',
        title: \`Elevation of Privilege at \${component.name}\`,
        description: \`An attacker could gain elevated permissions through \${component.name}\`,
        mitigations: ['Principle of least privilege', 'Role-based access control', 'Input validation'],
      });
    }

    // Determine likelihood and impact based on trust level and classification
    const getLikelihood = (): Threat['likelihood'] => {
      if (component.trustLevel === 'untrusted') return 'high';
      if (component.trustLevel === 'semi-trusted') return 'medium';
      return 'low';
    };

    const getImpact = (): Threat['impact'] => {
      if (component.dataClassification === 'restricted') return 'high';
      if (component.dataClassification === 'confidential') return 'high';
      if (component.dataClassification === 'internal') return 'medium';
      return 'low';
    };

    for (const stride of strideCategories) {
      const threat: Threat = {
        id: \`t-\${componentId}-\${threatCounter++}\`,
        category: stride.category,
        title: stride.title,
        description: stride.description,
        affectedComponent: componentId,
        likelihood: getLikelihood(),
        impact: getImpact(),
        mitigations: stride.mitigations,
      };
      threats.push(threat);
      this.model.threats.push(threat);
    }

    return threats;
  }

  analyzeDataFlowThreats(flowId: string): Threat[] {
    if (!this.model) return [];

    const flow = this.model.dataFlows.find(f => f.id === flowId);
    if (!flow) return [];

    const threats: Threat[] = [];
    let threatCounter = 1;

    // Analyze based on flow properties
    if (!flow.encrypted) {
      threats.push({
        id: \`t-flow-\${flowId}-\${threatCounter++}\`,
        category: 'information_disclosure',
        title: \`Unencrypted data in transit: \${flow.dataType}\`,
        description: \`Data flowing from \${flow.source} to \${flow.destination} is not encrypted\`,
        affectedComponent: flow.source,
        affectedFlow: flowId,
        likelihood: 'high',
        impact: 'high',
        mitigations: ['Enable TLS/SSL', 'Use encrypted protocols'],
      });
    }

    if (!flow.authenticated) {
      threats.push({
        id: \`t-flow-\${flowId}-\${threatCounter++}\`,
        category: 'spoofing',
        title: \`Unauthenticated data flow: \${flow.dataType}\`,
        description: \`No authentication on flow from \${flow.source} to \${flow.destination}\`,
        affectedComponent: flow.source,
        affectedFlow: flowId,
        likelihood: 'medium',
        impact: 'high',
        mitigations: ['Implement mutual TLS', 'API authentication', 'Token-based auth'],
      });
    }

    // Check for sensitive data types
    const sensitiveTypes = ['credentials', 'password', 'token', 'key', 'pii', 'credit card'];
    if (sensitiveTypes.some(type => flow.dataType.toLowerCase().includes(type))) {
      threats.push({
        id: \`t-flow-\${flowId}-\${threatCounter++}\`,
        category: 'information_disclosure',
        title: \`Sensitive data exposure: \${flow.dataType}\`,
        description: \`Sensitive data (\${flow.dataType}) requires additional protection\`,
        affectedComponent: flow.destination,
        affectedFlow: flowId,
        likelihood: 'medium',
        impact: 'high',
        mitigations: ['End-to-end encryption', 'Data tokenization', 'Secure storage'],
      });
    }

    // Tampering threat for all data flows
    threats.push({
      id: \`t-flow-\${flowId}-\${threatCounter++}\`,
      category: 'tampering',
      title: \`Data tampering in transit: \${flow.dataType}\`,
      description: \`Data could be modified between \${flow.source} and \${flow.destination}\`,
      affectedComponent: flow.destination,
      affectedFlow: flowId,
      likelihood: flow.encrypted ? 'low' : 'medium',
      impact: 'medium',
      mitigations: ['Message signing', 'Integrity checks', 'HMAC verification'],
    });

    for (const threat of threats) {
      this.model.threats.push(threat);
    }

    return threats;
  }

  calculateDREAD(threat: Threat): {
    damage: number;
    reproducibility: number;
    exploitability: number;
    affectedUsers: number;
    discoverability: number;
    totalScore: number;
  } {
    // Map likelihood and impact to DREAD scores (1-10)
    const likelihoodScore = { low: 3, medium: 6, high: 9 };
    const impactScore = { low: 3, medium: 6, high: 9 };

    // Calculate DREAD components
    const damage = impactScore[threat.impact];
    const reproducibility = threat.likelihood === 'high' ? 9 : threat.likelihood === 'medium' ? 6 : 3;
    const exploitability = likelihoodScore[threat.likelihood];
    const affectedUsers = threat.category === 'denial_of_service' ? 9 :
                         threat.category === 'information_disclosure' ? 7 : 5;
    const discoverability = threat.likelihood === 'high' ? 8 : threat.likelihood === 'medium' ? 5 : 3;

    const totalScore = Math.round((damage + reproducibility + exploitability + affectedUsers + discoverability) / 5);

    return {
      damage,
      reproducibility,
      exploitability,
      affectedUsers,
      discoverability,
      totalScore,
    };
  }

  prioritizeThreats(): Threat[] {
    if (!this.model) return [];

    const riskScore = (t: Threat): number => {
      const likelihoodValue = { low: 1, medium: 2, high: 3 };
      const impactValue = { low: 1, medium: 2, high: 3 };
      return likelihoodValue[t.likelihood] * impactValue[t.impact];
    };

    return [...this.model.threats].sort((a, b) => riskScore(b) - riskScore(a));
  }

  generateInterviewQuestions(): {
    question: string;
    expectedAnswer: string;
    category: string;
  }[] {
    if (!this.model) return [];

    const questions: { question: string; expectedAnswer: string; category: string }[] = [];

    // General threat modeling questions
    questions.push({
      question: 'What is STRIDE and how would you use it?',
      expectedAnswer: 'STRIDE is a threat modeling framework covering Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, and Elevation of Privilege. Apply each category to identify potential threats.',
      category: 'Framework Knowledge',
    });

    // Component-specific questions
    for (const component of this.model.components.slice(0, 2)) {
      questions.push({
        question: \`What threats might affect the \${component.name} (\${component.type})?\`,
        expectedAnswer: \`Consider the component's trust level (\${component.trustLevel}) and type. Main threats include spoofing if untrusted, tampering for data stores, and DoS for processes.\`,
        category: 'Component Analysis',
      });
    }

    // Data flow questions
    for (const flow of this.model.dataFlows.slice(0, 2)) {
      questions.push({
        question: \`What security concerns exist for the data flow carrying \${flow.dataType}?\`,
        expectedAnswer: \`Check encryption (\${flow.encrypted ? 'encrypted' : 'not encrypted'}), authentication (\${flow.authenticated ? 'authenticated' : 'not authenticated'}), and data sensitivity.\`,
        category: 'Data Flow Analysis',
      });
    }

    // Trust boundary questions
    if (this.model.trustBoundaries.length > 0) {
      questions.push({
        question: 'Why are trust boundaries important in threat modeling?',
        expectedAnswer: 'Trust boundaries mark where data crosses between different trust levels. Threats are most likely at boundaries where untrusted data enters trusted zones.',
        category: 'Trust Boundaries',
      });
    }

    // Mitigation questions
    if (this.model.threats.length > 0) {
      const highRiskThreat = this.model.threats.find(t => t.impact === 'high');
      if (highRiskThreat) {
        questions.push({
          question: \`How would you mitigate: \${highRiskThreat.title}?\`,
          expectedAnswer: \`Suggested mitigations: \${highRiskThreat.mitigations.join(', ')}\`,
          category: 'Mitigation Strategies',
        });
      }
    }

    return questions;
  }

  evaluateThreatIdentification(identifiedThreats: string[]): {
    score: number;
    missed: string[];
    falsePositives: string[];
  } {
    if (!this.model) return { score: 0, missed: [], falsePositives: [] };

    const actualThreats = this.model.threats.map(t => t.title.toLowerCase());
    const identified = identifiedThreats.map(t => t.toLowerCase());

    const matched: string[] = [];
    const missed: string[] = [];
    const falsePositives: string[] = [];

    // Check which actual threats were identified
    for (const actual of actualThreats) {
      const words = actual.split(/\s+/).filter(w => w.length > 3);
      const wasIdentified = identified.some(id =>
        words.some(word => id.includes(word)) || id.includes(actual)
      );

      if (wasIdentified) {
        matched.push(actual);
      } else {
        missed.push(actual);
      }
    }

    // Check for false positives
    for (const id of identified) {
      const words = id.split(/\s+/).filter(w => w.length > 3);
      const isActual = actualThreats.some(actual =>
        words.some(word => actual.includes(word)) || actual.includes(id)
      );

      if (!isActual) {
        falsePositives.push(id);
      }
    }

    const score = actualThreats.length > 0
      ? Math.round((matched.length / actualThreats.length) * 100)
      : 0;

    return { score, missed, falsePositives };
  }

  generateReport(): {
    summary: string;
    componentAnalysis: { component: string; threatCount: number; highRisk: number }[];
    topThreats: Threat[];
    recommendations: string[];
  } {
    if (!this.model) {
      return {
        summary: 'No model created',
        componentAnalysis: [],
        topThreats: [],
        recommendations: [],
      };
    }

    const prioritized = this.prioritizeThreats();
    const highRiskCount = this.model.threats.filter(t => t.impact === 'high' && t.likelihood === 'high').length;

    // Component analysis
    const componentAnalysis = this.model.components.map(c => {
      const componentThreats = this.model!.threats.filter(t => t.affectedComponent === c.id);
      return {
        component: c.name,
        threatCount: componentThreats.length,
        highRisk: componentThreats.filter(t => t.impact === 'high').length,
      };
    });

    // Generate recommendations
    const recommendations: string[] = [];

    if (this.model.dataFlows.some(f => !f.encrypted)) {
      recommendations.push('Enable encryption for all data flows');
    }

    if (this.model.components.some(c => c.trustLevel === 'untrusted')) {
      recommendations.push('Implement strict input validation at trust boundaries');
    }

    if (highRiskCount > 0) {
      recommendations.push(\`Address \${highRiskCount} high-risk threats as priority\`);
    }

    const uniqueMitigations = new Set(this.model.threats.flatMap(t => t.mitigations));
    recommendations.push(...Array.from(uniqueMitigations).slice(0, 3));

    return {
      summary: \`Threat model "\${this.model.name}" identified \${this.model.threats.length} threats across \${this.model.components.length} components. \${highRiskCount} threats are high-risk requiring immediate attention.\`,
      componentAnalysis,
      topThreats: prioritized.slice(0, 5),
      recommendations,
    };
  }

  suggestMitigations(threatId: string): string[] {
    if (!this.model) return [];

    const threat = this.model.threats.find(t => t.id === threatId);
    if (!threat) return [];

    const additionalMitigations: Record<Threat['category'], string[]> = {
      spoofing: ['Strong password policies', 'Session management', 'Account lockout'],
      tampering: ['Code signing', 'File integrity monitoring', 'Secure configuration'],
      repudiation: ['Centralized logging', 'Log tampering protection', 'Time synchronization'],
      information_disclosure: ['Data loss prevention', 'Network segmentation', 'Secure error handling'],
      denial_of_service: ['Auto-scaling', 'Circuit breakers', 'Graceful degradation'],
      elevation_of_privilege: ['Sandbox isolation', 'Capability-based security', 'Regular permission audits'],
    };

    return [...threat.mitigations, ...additionalMitigations[threat.category]];
  }
}

// Test your implementation
const exercise = new ThreatModelingExercise();

// Test 1: Create model
const model = exercise.createModel('E-Commerce App', 'Online shopping platform');
console.log('Test 1 - Model created:', model.name === 'E-Commerce App');

// Test 2: Add component
exercise.addComponent({
  id: 'c1',
  name: 'Web Browser',
  type: 'user',
  trustLevel: 'untrusted',
});
exercise.addComponent({
  id: 'c2',
  name: 'API Server',
  type: 'process',
  trustLevel: 'trusted',
  dataClassification: 'confidential',
});
console.log('Test 2 - Components added:', true);

// Test 3: Add data flow
exercise.addDataFlow({
  id: 'f1',
  source: 'c1',
  destination: 'c2',
  protocol: 'HTTPS',
  dataType: 'User credentials',
  encrypted: true,
  authenticated: false,
});
console.log('Test 3 - Data flow added:', true);

// Test 4: STRIDE analysis
const strideThreats = exercise.analyzeSTRIDE('c1');
console.log('Test 4 - STRIDE analysis:', strideThreats.length > 0);

// Test 5: Data flow threats
const flowThreats = exercise.analyzeDataFlowThreats('f1');
console.log('Test 5 - Flow threats:', flowThreats.length > 0);

// Test 6: Calculate DREAD
const dreadScore = exercise.calculateDREAD(strideThreats[0] || { id: 't1', category: 'spoofing', title: 'Test', description: '', affectedComponent: 'c1', likelihood: 'high', impact: 'high', mitigations: [] });
console.log('Test 6 - DREAD calculated:', dreadScore.totalScore >= 0);

// Test 7: Prioritize threats
const prioritized = exercise.prioritizeThreats();
console.log('Test 7 - Threats prioritized:', Array.isArray(prioritized));

// Test 8: Generate interview questions
const questions = exercise.generateInterviewQuestions();
console.log('Test 8 - Questions generated:', questions.length > 0);

// Test 9: Evaluate threat identification
const evalResult = exercise.evaluateThreatIdentification(['spoofing attack', 'SQL injection']);
console.log('Test 9 - Evaluation done:', 'score' in evalResult);

// Test 10: Generate report
const report = exercise.generateReport();
console.log('Test 10 - Report generated:', 'summary' in report);`,
	testCode: `import { describe, it, expect, beforeEach } from 'vitest';

interface SystemComponent {
  id: string;
  name: string;
  type: 'user' | 'process' | 'datastore' | 'external' | 'network';
  trustLevel: 'trusted' | 'semi-trusted' | 'untrusted';
  dataClassification?: 'public' | 'internal' | 'confidential' | 'restricted';
}

interface DataFlow {
  id: string;
  source: string;
  destination: string;
  protocol: string;
  dataType: string;
  encrypted: boolean;
  authenticated: boolean;
}

interface Threat {
  id: string;
  category: 'spoofing' | 'tampering' | 'repudiation' | 'information_disclosure' | 'denial_of_service' | 'elevation_of_privilege';
  title: string;
  description: string;
  affectedComponent: string;
  affectedFlow?: string;
  likelihood: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
  mitigations: string[];
}

interface ThreatModel {
  id: string;
  name: string;
  description: string;
  components: SystemComponent[];
  dataFlows: DataFlow[];
  threats: Threat[];
  trustBoundaries: { id: string; name: string; components: string[] }[];
}

class ThreatModelingExercise {
  private model: ThreatModel | null = null;

  createModel(name: string, description: string): ThreatModel {
    this.model = { id: \`model-\${Date.now()}\`, name, description, components: [], dataFlows: [], threats: [], trustBoundaries: [] };
    return this.model;
  }

  addComponent(component: SystemComponent): void {
    if (!this.model) return;
    this.model.components.push(component);
  }

  addDataFlow(flow: DataFlow): void {
    if (!this.model) return;
    this.model.dataFlows.push(flow);
  }

  addTrustBoundary(name: string, componentIds: string[]): void {
    if (!this.model) return;
    this.model.trustBoundaries.push({ id: \`tb-\${Date.now()}\`, name, components: componentIds });
  }

  analyzeSTRIDE(componentId: string): Threat[] {
    if (!this.model) return [];
    const component = this.model.components.find(c => c.id === componentId);
    if (!component) return [];

    const threats: Threat[] = [];
    if (component.type === 'user' || component.type === 'external') {
      threats.push({ id: \`t-\${componentId}-1\`, category: 'spoofing', title: \`Spoofing \${component.name}\`, description: 'Identity spoofing', affectedComponent: componentId, likelihood: component.trustLevel === 'untrusted' ? 'high' : 'medium', impact: 'high', mitigations: ['Strong auth'] });
    }
    if (component.type === 'process' || component.type === 'datastore') {
      threats.push({ id: \`t-\${componentId}-2\`, category: 'tampering', title: \`Tampering \${component.name}\`, description: 'Data tampering', affectedComponent: componentId, likelihood: 'medium', impact: 'high', mitigations: ['Integrity checks'] });
    }
    threats.forEach(t => this.model?.threats.push(t));
    return threats;
  }

  analyzeDataFlowThreats(flowId: string): Threat[] {
    if (!this.model) return [];
    const flow = this.model.dataFlows.find(f => f.id === flowId);
    if (!flow) return [];

    const threats: Threat[] = [];
    if (!flow.encrypted) {
      threats.push({ id: \`t-flow-\${flowId}-1\`, category: 'information_disclosure', title: 'Unencrypted flow', description: 'Data not encrypted', affectedComponent: flow.source, affectedFlow: flowId, likelihood: 'high', impact: 'high', mitigations: ['Enable TLS'] });
    }
    if (!flow.authenticated) {
      threats.push({ id: \`t-flow-\${flowId}-2\`, category: 'spoofing', title: 'Unauthenticated flow', description: 'No auth', affectedComponent: flow.source, affectedFlow: flowId, likelihood: 'medium', impact: 'high', mitigations: ['Add auth'] });
    }
    threats.forEach(t => this.model?.threats.push(t));
    return threats;
  }

  calculateDREAD(threat: Threat): { damage: number; reproducibility: number; exploitability: number; affectedUsers: number; discoverability: number; totalScore: number } {
    const impactScore = { low: 3, medium: 6, high: 9 };
    const damage = impactScore[threat.impact];
    const reproducibility = threat.likelihood === 'high' ? 9 : 6;
    const exploitability = threat.likelihood === 'high' ? 9 : 6;
    const affectedUsers = 5;
    const discoverability = 5;
    return { damage, reproducibility, exploitability, affectedUsers, discoverability, totalScore: Math.round((damage + reproducibility + exploitability + affectedUsers + discoverability) / 5) };
  }

  prioritizeThreats(): Threat[] {
    if (!this.model) return [];
    const riskScore = (t: Threat) => ({ low: 1, medium: 2, high: 3 }[t.likelihood] * { low: 1, medium: 2, high: 3 }[t.impact]);
    return [...this.model.threats].sort((a, b) => riskScore(b) - riskScore(a));
  }

  generateInterviewQuestions(): { question: string; expectedAnswer: string; category: string }[] {
    return [
      { question: 'What is STRIDE?', expectedAnswer: 'Spoofing, Tampering, Repudiation, Info Disclosure, DoS, Elevation of Privilege', category: 'Framework' },
      { question: 'Why use trust boundaries?', expectedAnswer: 'Mark where data crosses trust levels', category: 'Concepts' },
    ];
  }

  evaluateThreatIdentification(identifiedThreats: string[]): { score: number; missed: string[]; falsePositives: string[] } {
    if (!this.model) return { score: 0, missed: [], falsePositives: [] };
    const actual = this.model.threats.map(t => t.title.toLowerCase());
    const matched = identifiedThreats.filter(id => actual.some(a => a.includes(id.toLowerCase().split(' ')[0])));
    return { score: actual.length > 0 ? Math.round((matched.length / actual.length) * 100) : 0, missed: actual.filter(a => !matched.some(m => a.includes(m.toLowerCase()))), falsePositives: [] };
  }

  generateReport(): { summary: string; componentAnalysis: { component: string; threatCount: number; highRisk: number }[]; topThreats: Threat[]; recommendations: string[] } {
    if (!this.model) return { summary: 'No model', componentAnalysis: [], topThreats: [], recommendations: [] };
    return {
      summary: \`\${this.model.threats.length} threats identified\`,
      componentAnalysis: this.model.components.map(c => ({ component: c.name, threatCount: this.model!.threats.filter(t => t.affectedComponent === c.id).length, highRisk: 0 })),
      topThreats: this.prioritizeThreats().slice(0, 5),
      recommendations: ['Enable encryption', 'Implement authentication'],
    };
  }

  suggestMitigations(threatId: string): string[] {
    if (!this.model) return [];
    const threat = this.model.threats.find(t => t.id === threatId);
    return threat ? [...threat.mitigations, 'Additional mitigation'] : [];
  }
}

describe('ThreatModelingExercise', () => {
  let exercise: ThreatModelingExercise;

  beforeEach(() => {
    exercise = new ThreatModelingExercise();
  });

  it('should create a threat model', () => {
    const model = exercise.createModel('Test App', 'Description');
    expect(model.name).toBe('Test App');
    expect(model.components).toHaveLength(0);
  });

  it('should add components and data flows', () => {
    exercise.createModel('Test', 'Test');
    exercise.addComponent({ id: 'c1', name: 'Browser', type: 'user', trustLevel: 'untrusted' });
    exercise.addComponent({ id: 'c2', name: 'Server', type: 'process', trustLevel: 'trusted' });
    exercise.addDataFlow({ id: 'f1', source: 'c1', destination: 'c2', protocol: 'HTTPS', dataType: 'Credentials', encrypted: true, authenticated: true });

    // Verify by running analysis
    const threats = exercise.analyzeSTRIDE('c1');
    expect(threats.length).toBeGreaterThan(0);
  });

  it('should analyze STRIDE threats for components', () => {
    exercise.createModel('Test', 'Test');
    exercise.addComponent({ id: 'c1', name: 'User', type: 'user', trustLevel: 'untrusted' });

    const threats = exercise.analyzeSTRIDE('c1');

    expect(threats.some(t => t.category === 'spoofing')).toBe(true);
    expect(threats[0].affectedComponent).toBe('c1');
  });

  it('should analyze data flow threats', () => {
    exercise.createModel('Test', 'Test');
    exercise.addComponent({ id: 'c1', name: 'Client', type: 'user', trustLevel: 'untrusted' });
    exercise.addComponent({ id: 'c2', name: 'Server', type: 'process', trustLevel: 'trusted' });
    exercise.addDataFlow({ id: 'f1', source: 'c1', destination: 'c2', protocol: 'HTTP', dataType: 'Data', encrypted: false, authenticated: false });

    const threats = exercise.analyzeDataFlowThreats('f1');

    expect(threats.some(t => t.category === 'information_disclosure')).toBe(true);
    expect(threats.some(t => t.category === 'spoofing')).toBe(true);
  });

  it('should calculate DREAD scores', () => {
    const threat: Threat = {
      id: 't1', category: 'information_disclosure', title: 'Data leak',
      description: 'Test', affectedComponent: 'c1', likelihood: 'high', impact: 'high', mitigations: []
    };

    const dread = exercise.calculateDREAD(threat);

    expect(dread.totalScore).toBeGreaterThan(0);
    expect(dread.damage).toBe(9); // High impact
  });

  it('should prioritize threats by risk', () => {
    exercise.createModel('Test', 'Test');
    exercise.addComponent({ id: 'c1', name: 'Component', type: 'process', trustLevel: 'trusted', dataClassification: 'confidential' });

    exercise.analyzeSTRIDE('c1');
    const prioritized = exercise.prioritizeThreats();

    expect(prioritized.length).toBeGreaterThan(0);
    // Higher risk should come first
  });

  it('should generate interview questions', () => {
    exercise.createModel('Test', 'Test');
    const questions = exercise.generateInterviewQuestions();

    expect(questions.length).toBeGreaterThan(0);
    expect(questions[0].question).toBeDefined();
    expect(questions[0].expectedAnswer).toBeDefined();
  });

  it('should evaluate threat identification', () => {
    exercise.createModel('Test', 'Test');
    exercise.addComponent({ id: 'c1', name: 'User', type: 'user', trustLevel: 'untrusted' });
    exercise.analyzeSTRIDE('c1');

    const result = exercise.evaluateThreatIdentification(['Spoofing attack']);

    expect(result.score).toBeGreaterThan(0);
  });

  it('should generate comprehensive report', () => {
    exercise.createModel('Test App', 'Test');
    exercise.addComponent({ id: 'c1', name: 'Component', type: 'process', trustLevel: 'trusted' });
    exercise.analyzeSTRIDE('c1');

    const report = exercise.generateReport();

    expect(report.summary).toContain('threat');
    expect(report.recommendations.length).toBeGreaterThan(0);
  });

  it('should suggest mitigations', () => {
    exercise.createModel('Test', 'Test');
    exercise.addComponent({ id: 'c1', name: 'User', type: 'user', trustLevel: 'untrusted' });
    const threats = exercise.analyzeSTRIDE('c1');

    const mitigations = exercise.suggestMitigations(threats[0].id);

    expect(mitigations.length).toBeGreaterThan(0);
  });
});`,
	hint1:
		'For STRIDE analysis, match threat categories to component types: Spoofing applies to users/external entities, Tampering to data stores, DoS to processes/networks. Consider trust level for likelihood.',
	hint2:
		'For data flow analysis, always check encryption and authentication status. Unencrypted flows with sensitive data (credentials, PII) should be flagged as high-risk information disclosure threats.',
	whyItMatters: `Threat modeling is a cornerstone skill for security architects and senior security engineers:

**Why Threat Modeling Matters:**
- **Proactive Security** - Find issues before attackers do
- **Cost Effective** - Fixing design issues is 10-100x cheaper than fixing deployed systems
- **Systematic Approach** - Ensures comprehensive coverage of threats

**Interview Expectations:**
- **STRIDE Knowledge** - Can you systematically identify threat categories?
- **Prioritization** - Can you focus on the most important risks?
- **Mitigation Strategies** - Do you know how to address identified threats?

**Real-World Application:**
- **Microsoft SDL** - Uses threat modeling for all products
- **OWASP** - Recommends threat modeling as critical practice
- **Financial Services** - Required for regulatory compliance (PCI-DSS, SOC2)

**Common Interview Questions:**
- "Walk me through how you would threat model this system"
- "What STRIDE category does this vulnerability fall under?"
- "How would you prioritize these threats?"

Demonstrating strong threat modeling skills shows architectural thinking and security maturity.`,
	order: 3,
	translations: {
		ru: {
			title: 'Моделирование угроз на собеседовании',
			description: `Практикуйте моделирование угроз - критический навык для собеседований по архитектуре безопасности.

**Фреймворки моделирования угроз:**

1. **STRIDE** - Спуфинг, Тампинг, Отказ, Раскрытие информации, DoS, Повышение привилегий
2. **DREAD** - Ущерб, Воспроизводимость, Эксплуатируемость, Затронутые пользователи, Обнаруживаемость
3. **PASTA** - Процесс симуляции атак и анализа угроз
4. **Деревья атак** - Иерархическое представление угроз

**Ваша задача:**

Реализуйте класс \`ThreatModelingExercise\`, помогающий практиковать методологию моделирования угроз.`,
			hint1:
				'Для STRIDE анализа сопоставьте категории угроз с типами компонентов: Спуфинг для пользователей/внешних сущностей, Тампинг для хранилищ данных, DoS для процессов/сетей.',
			hint2:
				'Для анализа потоков данных всегда проверяйте статус шифрования и аутентификации. Незашифрованные потоки с чувствительными данными должны помечаться как высокорисковые.',
			whyItMatters: `Моделирование угроз - краеугольный навык для архитекторов безопасности:

**Почему важно:**
- **Проактивная безопасность** - Находите проблемы раньше атакующих
- **Экономичность** - Исправление проблем дизайна в 10-100 раз дешевле
- **Систематический подход** - Обеспечивает полное покрытие угроз

**Ожидания на собеседовании:**
- **Знание STRIDE** - Можете ли вы систематически идентифицировать категории угроз?
- **Приоритизация** - Можете ли вы сфокусироваться на важнейших рисках?
- **Стратегии смягчения** - Знаете ли вы, как устранить угрозы?`,
		},
		uz: {
			title: 'Intervyuda tahdid modellash',
			description: `Tahdid modellashni mashq qiling - xavfsizlik arxitekturasi intervyulari uchun muhim ko'nikma.

**Tahdid modellash frameworklari:**

1. **STRIDE** - Spoofing, Tampering, Repudiation, Info Disclosure, DoS, Elevation of Privilege
2. **DREAD** - Zarar, Qayta ishlab chiqarish, Ekspluatatsiya, Ta'sirlangan foydalanuvchilar, Kashf qilish
3. **PASTA** - Hujum simulyatsiyasi va tahdid tahlili jarayoni
4. **Hujum daraxtlari** - Tahdidlarning ierarxik ko'rinishi

**Vazifangiz:**

Tahdid modellash metodologiyasini mashq qilishga yordam beradigan \`ThreatModelingExercise\` klassini yarating.`,
			hint1:
				"STRIDE tahlili uchun tahdid kategoriyalarini komponent turlariga moslashtiring: Spoofing foydalanuvchilar/tashqi sub'ektlar uchun, Tampering ma'lumotlar ombori uchun, DoS jarayonlar/tarmoqlar uchun.",
			hint2:
				"Ma'lumot oqimi tahlili uchun har doim shifrlash va autentifikatsiya holatini tekshiring. Sezgir ma'lumotli shifrlanmagan oqimlar yuqori xavfli deb belgilanishi kerak.",
			whyItMatters: `Tahdid modellash xavfsizlik arxitektorlari uchun asosiy ko'nikma:

**Nima uchun muhim:**
- **Proaktiv xavfsizlik** - Muammolarni hujumchilardan oldin toping
- **Tejamkorlik** - Dizayn muammolarini tuzatish 10-100 marta arzonroq
- **Tizimli yondashuv** - Tahdidlarni to'liq qamrab olishni ta'minlaydi

**Intervyu kutishlari:**
- **STRIDE bilimi** - Tahdid kategoriyalarini tizimli ravishda aniqlay olasizmi?
- **Ustuvorlashtirish** - Eng muhim xavflarga e'tibor qarata olasizmi?
- **Yumshatish strategiyalari** - Aniqlangan tahdidlarni qanday bartaraf etishni bilasizmi?`,
		},
	},
};

export default task;
