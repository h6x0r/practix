import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-threat-modeling',
	title: 'Threat Modeling with STRIDE',
	difficulty: 'medium',
	tags: ['security', 'fundamentals', 'threat-modeling', 'stride', 'typescript'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn threat modeling using the STRIDE methodology - a systematic approach to identifying security threats.

**What is Threat Modeling?**

Threat modeling is the process of identifying potential threats to your system and determining countermeasures. It's best done during the design phase but valuable at any stage.

**STRIDE Categories:**

| Threat | Description | Property Violated |
|--------|-------------|-------------------|
| **S**poofing | Impersonating a user or system | Authentication |
| **T**ampering | Modifying data or code | Integrity |
| **R**epudiation | Denying actions performed | Non-repudiation |
| **I**nformation Disclosure | Exposing data to unauthorized users | Confidentiality |
| **D**enial of Service | Making system unavailable | Availability |
| **E**levation of Privilege | Gaining unauthorized access | Authorization |

**Your Task:**

Implement a \`ThreatModeler\` class that:

1. Identifies threats based on system components
2. Categorizes threats using STRIDE
3. Calculates risk scores
4. Suggests mitigations

**Example Usage:**

\`\`\`typescript
const modeler = new ThreatModeler();

modeler.addComponent('LoginForm', ['user_input', 'authentication']);
modeler.addComponent('Database', ['data_store', 'sensitive_data']);

const threats = modeler.identifyThreats('LoginForm');
// [{ type: 'Spoofing', description: '...', risk: 'high' }, ...]
\`\`\``,
	initialCode: `type StrideCategory = 'Spoofing' | 'Tampering' | 'Repudiation' | 'InformationDisclosure' | 'DenialOfService' | 'ElevationOfPrivilege';

interface Threat {
  id: string;
  category: StrideCategory;
  description: string;
  component: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  mitigation: string;
}

interface SystemComponent {
  name: string;
  characteristics: string[];
  dataFlow?: string[];
}

class ThreatModeler {
  private components: Map<string, SystemComponent> = new Map();
  private threats: Threat[] = [];

  addComponent(name: string, characteristics: string[]): void {
    // TODO: Add a system component
  }

  identifyThreats(componentName: string): Threat[] {
    // TODO: Identify threats for a component based on its characteristics
    // Map characteristics to STRIDE categories
    return [];
  }

  getAllThreats(): Threat[] {
    // TODO: Get all identified threats across all components
    return [];
  }

  getThreatsByCategory(category: StrideCategory): Threat[] {
    // TODO: Filter threats by STRIDE category
    return [];
  }

  calculateRiskScore(threat: Threat): number {
    // TODO: Calculate numerical risk score (1-10)
    return 0;
  }

  getMitigations(category: StrideCategory): string[] {
    // TODO: Get recommended mitigations for a threat category
    return [];
  }

  generateReport(): string {
    // TODO: Generate a threat model report
    return '';
  }

  analyzeDataFlow(from: string, to: string): Threat[] {
    // TODO: Analyze threats in data flow between components
    return [];
  }
}

export { ThreatModeler, Threat, StrideCategory, SystemComponent };`,
	solutionCode: `type StrideCategory = 'Spoofing' | 'Tampering' | 'Repudiation' | 'InformationDisclosure' | 'DenialOfService' | 'ElevationOfPrivilege';

interface Threat {
  id: string;
  category: StrideCategory;
  description: string;
  component: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  mitigation: string;
}

interface SystemComponent {
  name: string;
  characteristics: string[];
  dataFlow?: string[];
}

class ThreatModeler {
  private components: Map<string, SystemComponent> = new Map();
  private threats: Threat[] = [];
  private threatCounter = 0;

  private readonly CHARACTERISTIC_THREATS: Record<string, { category: StrideCategory; description: string; risk: 'low' | 'medium' | 'high' | 'critical' }[]> = {
    'user_input': [
      { category: 'Tampering', description: 'Malicious input could modify system behavior', risk: 'high' },
      { category: 'InformationDisclosure', description: 'Input errors may reveal system information', risk: 'medium' },
    ],
    'authentication': [
      { category: 'Spoofing', description: 'Attacker could impersonate legitimate user', risk: 'critical' },
      { category: 'Repudiation', description: 'User may deny performing actions', risk: 'medium' },
    ],
    'data_store': [
      { category: 'Tampering', description: 'Stored data could be modified by attacker', risk: 'high' },
      { category: 'InformationDisclosure', description: 'Sensitive data could be exposed', risk: 'critical' },
    ],
    'sensitive_data': [
      { category: 'InformationDisclosure', description: 'Sensitive information could leak', risk: 'critical' },
    ],
    'network': [
      { category: 'DenialOfService', description: 'Network could be flooded with requests', risk: 'high' },
      { category: 'Spoofing', description: 'Network identity could be spoofed', risk: 'medium' },
    ],
    'api_endpoint': [
      { category: 'DenialOfService', description: 'API could be overwhelmed', risk: 'medium' },
      { category: 'ElevationOfPrivilege', description: 'API could allow unauthorized access', risk: 'high' },
    ],
    'admin_function': [
      { category: 'ElevationOfPrivilege', description: 'Attacker could gain admin access', risk: 'critical' },
    ],
  };

  private readonly MITIGATIONS: Record<StrideCategory, string[]> = {
    'Spoofing': ['Implement strong authentication', 'Use MFA', 'Validate certificates'],
    'Tampering': ['Use digital signatures', 'Implement integrity checks', 'Use secure channels'],
    'Repudiation': ['Implement audit logging', 'Use digital signatures', 'Require acknowledgments'],
    'InformationDisclosure': ['Encrypt sensitive data', 'Implement access controls', 'Use secure protocols'],
    'DenialOfService': ['Implement rate limiting', 'Use load balancers', 'Add redundancy'],
    'ElevationOfPrivilege': ['Follow least privilege', 'Validate permissions', 'Segment networks'],
  };

  addComponent(name: string, characteristics: string[]): void {
    this.components.set(name, { name, characteristics });
    this.identifyThreats(name);
  }

  identifyThreats(componentName: string): Threat[] {
    const component = this.components.get(componentName);
    if (!component) return [];

    const componentThreats: Threat[] = [];

    for (const char of component.characteristics) {
      const threatTemplates = this.CHARACTERISTIC_THREATS[char] || [];

      for (const template of threatTemplates) {
        const threat: Threat = {
          id: \`T-\${++this.threatCounter}\`,
          category: template.category,
          description: \`\${template.description} in \${componentName}\`,
          component: componentName,
          riskLevel: template.risk,
          mitigation: this.MITIGATIONS[template.category][0],
        };

        this.threats.push(threat);
        componentThreats.push(threat);
      }
    }

    return componentThreats;
  }

  getAllThreats(): Threat[] {
    return [...this.threats];
  }

  getThreatsByCategory(category: StrideCategory): Threat[] {
    return this.threats.filter(t => t.category === category);
  }

  calculateRiskScore(threat: Threat): number {
    const riskScores: Record<string, number> = {
      'low': 2,
      'medium': 5,
      'high': 8,
      'critical': 10,
    };
    return riskScores[threat.riskLevel] || 0;
  }

  getMitigations(category: StrideCategory): string[] {
    return this.MITIGATIONS[category] || [];
  }

  generateReport(): string {
    const lines: string[] = [
      '# Threat Model Report',
      '',
      \`## Components Analyzed: \${this.components.size}\`,
      \`## Total Threats Identified: \${this.threats.length}\`,
      '',
      '## Threats by Category:',
    ];

    const categories: StrideCategory[] = ['Spoofing', 'Tampering', 'Repudiation', 'InformationDisclosure', 'DenialOfService', 'ElevationOfPrivilege'];

    for (const cat of categories) {
      const catThreats = this.getThreatsByCategory(cat);
      lines.push(\`- \${cat}: \${catThreats.length} threats\`);
    }

    lines.push('', '## Critical Threats:');
    const critical = this.threats.filter(t => t.riskLevel === 'critical');
    for (const t of critical) {
      lines.push(\`- [\${t.id}] \${t.description}\`);
      lines.push(\`  Mitigation: \${t.mitigation}\`);
    }

    return lines.join('\\n');
  }

  analyzeDataFlow(from: string, to: string): Threat[] {
    const flowThreats: Threat[] = [];

    flowThreats.push({
      id: \`T-\${++this.threatCounter}\`,
      category: 'Tampering',
      description: \`Data in transit from \${from} to \${to} could be modified\`,
      component: \`\${from}->\${to}\`,
      riskLevel: 'high',
      mitigation: 'Use TLS encryption for data in transit',
    });

    flowThreats.push({
      id: \`T-\${++this.threatCounter}\`,
      category: 'InformationDisclosure',
      description: \`Data flow from \${from} to \${to} could be intercepted\`,
      component: \`\${from}->\${to}\`,
      riskLevel: 'high',
      mitigation: 'Encrypt sensitive data before transmission',
    });

    this.threats.push(...flowThreats);
    return flowThreats;
  }
}

export { ThreatModeler, Threat, StrideCategory, SystemComponent };`,
	hint1: `For identifyThreats, loop through the component's characteristics and look up corresponding threats in CHARACTERISTIC_THREATS. Create Threat objects with unique IDs.`,
	hint2: `For calculateRiskScore, map risk levels to numerical values: low=2, medium=5, high=8, critical=10.`,
	testCode: `import { ThreatModeler } from './solution';

// Test1: addComponent adds component
test('Test1', () => {
  const modeler = new ThreatModeler();
  modeler.addComponent('Login', ['authentication']);
  const threats = modeler.getAllThreats();
  expect(threats.length).toBeGreaterThan(0);
});

// Test2: identifyThreats finds Spoofing for authentication
test('Test2', () => {
  const modeler = new ThreatModeler();
  modeler.addComponent('Login', ['authentication']);
  const threats = modeler.getThreatsByCategory('Spoofing');
  expect(threats.length).toBeGreaterThan(0);
});

// Test3: identifyThreats finds Tampering for user_input
test('Test3', () => {
  const modeler = new ThreatModeler();
  modeler.addComponent('Form', ['user_input']);
  const threats = modeler.getThreatsByCategory('Tampering');
  expect(threats.length).toBeGreaterThan(0);
});

// Test4: calculateRiskScore returns correct values
test('Test4', () => {
  const modeler = new ThreatModeler();
  modeler.addComponent('DB', ['sensitive_data']);
  const threats = modeler.getAllThreats();
  const score = modeler.calculateRiskScore(threats[0]);
  expect(score).toBeGreaterThanOrEqual(1);
  expect(score).toBeLessThanOrEqual(10);
});

// Test5: getMitigations returns array
test('Test5', () => {
  const modeler = new ThreatModeler();
  const mitigations = modeler.getMitigations('Spoofing');
  expect(mitigations.length).toBeGreaterThan(0);
  expect(mitigations.some(m => m.toLowerCase().includes('auth'))).toBe(true);
});

// Test6: getThreatsByCategory filters correctly
test('Test6', () => {
  const modeler = new ThreatModeler();
  modeler.addComponent('API', ['api_endpoint', 'authentication']);
  const dos = modeler.getThreatsByCategory('DenialOfService');
  expect(dos.every(t => t.category === 'DenialOfService')).toBe(true);
});

// Test7: generateReport returns string
test('Test7', () => {
  const modeler = new ThreatModeler();
  modeler.addComponent('App', ['user_input', 'data_store']);
  const report = modeler.generateReport();
  expect(report).toContain('Threat Model Report');
  expect(report).toContain('Threats');
});

// Test8: analyzeDataFlow creates threats
test('Test8', () => {
  const modeler = new ThreatModeler();
  const threats = modeler.analyzeDataFlow('Frontend', 'Backend');
  expect(threats.length).toBeGreaterThan(0);
  expect(threats.some(t => t.category === 'Tampering')).toBe(true);
});

// Test9: Threat IDs are unique
test('Test9', () => {
  const modeler = new ThreatModeler();
  modeler.addComponent('C1', ['authentication']);
  modeler.addComponent('C2', ['user_input']);
  const threats = modeler.getAllThreats();
  const ids = threats.map(t => t.id);
  expect(new Set(ids).size).toBe(ids.length);
});

// Test10: Critical threats for sensitive_data
test('Test10', () => {
  const modeler = new ThreatModeler();
  modeler.addComponent('DB', ['sensitive_data']);
  const threats = modeler.getAllThreats();
  expect(threats.some(t => t.riskLevel === 'critical')).toBe(true);
});`,
	whyItMatters: `Threat modeling catches security issues before they become vulnerabilities.

**Microsoft's SDL Requirement:**

Microsoft requires threat modeling for all products. It has identified thousands of potential vulnerabilities before code was written.

**STRIDE in Practice:**

| Your System Has | Look For These Threats |
|-----------------|----------------------|
| User login | Spoofing, Repudiation |
| Data storage | Tampering, Info Disclosure |
| Web API | DoS, Elevation of Privilege |
| Payment processing | All six categories |

**Real Example: E-commerce App**

\`\`\`
Component: Shopping Cart
├─ Spoofing: User could access another's cart
│  → Mitigation: Session binding, user validation
├─ Tampering: Price could be modified client-side
│  → Mitigation: Server-side price validation
├─ Repudiation: User denies placing order
│  → Mitigation: Audit logs, confirmation emails
├─ Info Disclosure: Cart contents leaked
│  → Mitigation: HTTPS, access controls
├─ DoS: Cart spam exhausts inventory
│  → Mitigation: Rate limiting, cart timeouts
└─ Elevation: Regular user gets admin prices
   → Mitigation: Role-based pricing validation
\`\`\``,
	order: 3,
	translations: {
		ru: {
			title: 'Моделирование угроз с STRIDE',
			description: `Изучите моделирование угроз с методологией STRIDE - систематический подход к выявлению угроз безопасности.

**Что такое моделирование угроз?**

Моделирование угроз - процесс выявления потенциальных угроз системе и определения контрмер.

**Категории STRIDE:**

- **S**poofing - Подмена личности
- **T**ampering - Подделка данных
- **R**epudiation - Отказ от действий
- **I**nformation Disclosure - Раскрытие информации
- **D**enial of Service - Отказ в обслуживании
- **E**levation of Privilege - Повышение привилегий

**Ваша задача:**

Реализуйте класс \`ThreatModeler\`.`,
			hint1: `Для identifyThreats пройдите по характеристикам компонента и найдите соответствующие угрозы в CHARACTERISTIC_THREATS.`,
			hint2: `Для calculateRiskScore сопоставьте уровни риска с числовыми значениями: low=2, medium=5, high=8, critical=10.`,
			whyItMatters: `Моделирование угроз выявляет проблемы безопасности до того, как они станут уязвимостями.`
		},
		uz: {
			title: 'STRIDE bilan tahdid modellashtirish',
			description: `STRIDE metodologiyasi bilan tahdid modellashtirishni o'rganing - xavfsizlik tahdidlarini aniqlashning tizimli yondashuvi.

**Tahdid modellashtirish nima?**

Tahdid modellashtirish - tizimingizga potensial tahdidlarni aniqlash va qarshi choralarni belgilash jarayoni.

**Sizning vazifangiz:**

\`ThreatModeler\` klassini amalga oshiring.`,
			hint1: `identifyThreats uchun komponent xususiyatlari bo'ylab yuring va CHARACTERISTIC_THREATS da tegishli tahdidlarni toping.`,
			hint2: `calculateRiskScore uchun risk darajalarini raqamli qiymatlarga moslashtiring: low=2, medium=5, high=8, critical=10.`,
			whyItMatters: `Tahdid modellashtirish xavfsizlik muammolarini zaiflikka aylanishidan oldin aniqlaydi.`
		}
	}
};

export default task;
