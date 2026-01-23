import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-cia-triad',
	title: 'CIA Triad: The Foundation of Security',
	difficulty: 'easy',
	tags: ['security', 'cia-triad', 'fundamentals', 'typescript'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn the CIA Triad - the three fundamental principles of information security.

**What is the CIA Triad?**

The CIA Triad consists of three core principles:

1. **Confidentiality** - Protecting data from unauthorized access
2. **Integrity** - Ensuring data hasn't been tampered with
3. **Availability** - Ensuring authorized users can access data when needed

**Your Task:**

Implement a \`SecurityAnalyzer\` class that evaluates security measures based on the CIA Triad. The class should:

1. Track which CIA principles are implemented
2. Calculate a security score (0-100)
3. Identify missing security measures

**Example Usage:**

\`\`\`typescript
const analyzer = new SecurityAnalyzer();

// Add security measures
analyzer.addMeasure('encryption', 'confidentiality');
analyzer.addMeasure('access-control', 'confidentiality');
analyzer.addMeasure('checksums', 'integrity');
analyzer.addMeasure('backup', 'availability');

// Check coverage
console.log(analyzer.getScore()); // 67 (2/3 + 1/3 + 1/3 principles covered)
console.log(analyzer.getMissingPrinciples()); // []
console.log(analyzer.getRecommendations());
\`\`\`

**Requirements:**

1. \`addMeasure(name: string, principle: CIAPrinciple)\` - Add a security measure
2. \`getScore(): number\` - Return 0-100 based on coverage
3. \`getMissingPrinciples(): CIAPrinciple[]\` - Return uncovered principles
4. \`getRecommendations(): string[]\` - Return security recommendations

**Scoring Logic:**
- Each principle (C, I, A) contributes 33.33% to the score
- Having at least one measure per principle = full score for that principle
- Bonus: Multiple measures per principle add 5% each (max 10% bonus)`,
	initialCode: `type CIAPrinciple = 'confidentiality' | 'integrity' | 'availability';

interface SecurityMeasure {
  name: string;
  principle: CIAPrinciple;
}

class SecurityAnalyzer {
  private measures: SecurityMeasure[] = [];

  addMeasure(name: string, principle: CIAPrinciple): void {
    // TODO: Add the security measure to the list
  }

  getScore(): number {
    // TODO: Calculate security score (0-100)
    // Each principle contributes 33.33%
    // Bonus 5% for multiple measures per principle (max 10% total bonus)
    return 0;
  }

  getMissingPrinciples(): CIAPrinciple[] {
    // TODO: Return principles with no measures
    return [];
  }

  getRecommendations(): string[] {
    // TODO: Return security recommendations based on missing principles
    return [];
  }

  getMeasuresByPrinciple(principle: CIAPrinciple): SecurityMeasure[] {
    // TODO: Return all measures for a specific principle
    return [];
  }
}

export { SecurityAnalyzer, CIAPrinciple, SecurityMeasure };`,
	solutionCode: `type CIAPrinciple = 'confidentiality' | 'integrity' | 'availability';

interface SecurityMeasure {
  name: string;
  principle: CIAPrinciple;
}

class SecurityAnalyzer {
  private measures: SecurityMeasure[] = [];

  // Add a new security measure to track
  addMeasure(name: string, principle: CIAPrinciple): void {
    this.measures.push({ name, principle });  // Store measure with its principle
  }

  // Calculate security score based on CIA coverage
  getScore(): number {
    const principles: CIAPrinciple[] = ['confidentiality', 'integrity', 'availability'];
    let score = 0;  // Start with 0 score
    let bonusPoints = 0;  // Track bonus for multiple measures

    for (const principle of principles) {
      const count = this.getMeasuresByPrinciple(principle).length;  // Count measures per principle

      if (count >= 1) {
        score += 33.33;  // Base score for having at least one measure
      }

      // Bonus points for multiple measures (5% each, max 10% total)
      if (count >= 2) {
        bonusPoints += Math.min((count - 1) * 5, 10);  // Cap at 10%
      }
    }

    return Math.min(100, Math.round(score + bonusPoints));  // Cap at 100
  }

  // Find principles without any security measures
  getMissingPrinciples(): CIAPrinciple[] {
    const principles: CIAPrinciple[] = ['confidentiality', 'integrity', 'availability'];
    return principles.filter(p => this.getMeasuresByPrinciple(p).length === 0);  // Filter uncovered
  }

  // Generate recommendations based on missing principles
  getRecommendations(): string[] {
    const recommendations: string[] = [];
    const missing = this.getMissingPrinciples();

    if (missing.includes('confidentiality')) {
      recommendations.push('Add encryption for data at rest and in transit');  // C recommendation
      recommendations.push('Implement access control mechanisms');
    }

    if (missing.includes('integrity')) {
      recommendations.push('Add checksums or digital signatures for data validation');  // I recommendation
      recommendations.push('Implement input validation and sanitization');
    }

    if (missing.includes('availability')) {
      recommendations.push('Set up backup and disaster recovery procedures');  // A recommendation
      recommendations.push('Implement redundancy and load balancing');
    }

    return recommendations;
  }

  // Get all measures for a specific principle
  getMeasuresByPrinciple(principle: CIAPrinciple): SecurityMeasure[] {
    return this.measures.filter(m => m.principle === principle);  // Filter by principle
  }
}

export { SecurityAnalyzer, CIAPrinciple, SecurityMeasure };`,
	hint1: `Start by implementing addMeasure() - just push the measure object to the array. Then getMeasuresByPrinciple() filters measures by the given principle.`,
	hint2: `For getScore(), loop through all three principles. If getMeasuresByPrinciple returns at least 1 measure, add 33.33. For bonus, add 5 for each extra measure (cap at 10 total bonus).`,
	testCode: `import { SecurityAnalyzer } from './solution';

// Test1: SecurityAnalyzer can be instantiated
test('Test1', () => {
  const analyzer = new SecurityAnalyzer();
  expect(analyzer).toBeDefined();
});

// Test2: addMeasure adds measures correctly
test('Test2', () => {
  const analyzer = new SecurityAnalyzer();
  analyzer.addMeasure('encryption', 'confidentiality');
  expect(analyzer.getMeasuresByPrinciple('confidentiality').length).toBe(1);
});

// Test3: getScore returns 0 for no measures
test('Test3', () => {
  const analyzer = new SecurityAnalyzer();
  expect(analyzer.getScore()).toBe(0);
});

// Test4: getScore returns ~33 for one principle covered
test('Test4', () => {
  const analyzer = new SecurityAnalyzer();
  analyzer.addMeasure('encryption', 'confidentiality');
  expect(analyzer.getScore()).toBe(33);
});

// Test5: getScore returns ~67 for two principles covered
test('Test5', () => {
  const analyzer = new SecurityAnalyzer();
  analyzer.addMeasure('encryption', 'confidentiality');
  analyzer.addMeasure('checksums', 'integrity');
  expect(analyzer.getScore()).toBe(67);
});

// Test6: getScore returns 100 for all principles covered
test('Test6', () => {
  const analyzer = new SecurityAnalyzer();
  analyzer.addMeasure('encryption', 'confidentiality');
  analyzer.addMeasure('checksums', 'integrity');
  analyzer.addMeasure('backup', 'availability');
  expect(analyzer.getScore()).toBe(100);
});

// Test7: getMissingPrinciples returns all when empty
test('Test7', () => {
  const analyzer = new SecurityAnalyzer();
  const missing = analyzer.getMissingPrinciples();
  expect(missing).toContain('confidentiality');
  expect(missing).toContain('integrity');
  expect(missing).toContain('availability');
});

// Test8: getMissingPrinciples returns uncovered principles
test('Test8', () => {
  const analyzer = new SecurityAnalyzer();
  analyzer.addMeasure('encryption', 'confidentiality');
  const missing = analyzer.getMissingPrinciples();
  expect(missing).not.toContain('confidentiality');
  expect(missing).toContain('integrity');
});

// Test9: getRecommendations returns suggestions for missing
test('Test9', () => {
  const analyzer = new SecurityAnalyzer();
  const recs = analyzer.getRecommendations();
  expect(recs.length).toBeGreaterThan(0);
});

// Test10: Bonus score for multiple measures
test('Test10', () => {
  const analyzer = new SecurityAnalyzer();
  analyzer.addMeasure('encryption', 'confidentiality');
  analyzer.addMeasure('access-control', 'confidentiality');
  analyzer.addMeasure('checksums', 'integrity');
  analyzer.addMeasure('backup', 'availability');
  expect(analyzer.getScore()).toBeGreaterThanOrEqual(100);
});`,
	whyItMatters: `The CIA Triad is the cornerstone of all security practices. Understanding it helps you design secure systems from the ground up.

**Real-World Impact:**

**1. Data Breaches (Confidentiality Failure)**
\`\`\`
2023 Examples:
- T-Mobile breach: 37 million customer records exposed
- MOVEit breach: 60+ million affected globally
- Cost: Average $4.45 million per breach (IBM 2023)

Root cause: Missing encryption, weak access controls
\`\`\`

**2. Data Tampering (Integrity Failure)**
\`\`\`
SolarWinds Attack (2020):
- Attackers modified build system
- Malicious code distributed to 18,000+ organizations
- Detected months later

Prevention: Code signing, integrity monitoring, secure CI/CD
\`\`\`

**3. Service Outages (Availability Failure)**
\`\`\`
AWS S3 Outage (2017):
- Typo in command took down major websites
- 4+ hours of downtime
- Billions in lost revenue

Prevention: Redundancy, circuit breakers, DR planning
\`\`\`

**Security Design Questions:**

When designing any system, ask:
1. **Confidentiality**: Who can see this data? How is it protected?
2. **Integrity**: How do we know data hasn't been modified?
3. **Availability**: What happens if this service goes down?

**Common Security Measures by Principle:**

| Confidentiality | Integrity | Availability |
|-----------------|-----------|--------------|
| Encryption (AES-256) | Checksums (SHA-256) | Load balancing |
| Access control (RBAC) | Digital signatures | Redundancy |
| Data masking | Input validation | Backups |
| TLS/SSL | Audit logs | CDN |
| VPN | Version control | Failover |`,
	order: 0,
	translations: {
		ru: {
			title: 'Триада CIA: Основа безопасности',
			description: `Изучите триаду CIA - три фундаментальных принципа информационной безопасности.

**Что такое триада CIA?**

Триада CIA состоит из трёх основных принципов:

1. **Конфиденциальность (Confidentiality)** - Защита данных от несанкционированного доступа
2. **Целостность (Integrity)** - Обеспечение того, что данные не были изменены
3. **Доступность (Availability)** - Обеспечение доступа авторизованным пользователям

**Ваша задача:**

Реализуйте класс \`SecurityAnalyzer\`, который оценивает меры безопасности на основе триады CIA.

**Требования:**

1. \`addMeasure(name: string, principle: CIAPrinciple)\` - Добавить меру безопасности
2. \`getScore(): number\` - Вернуть оценку 0-100 на основе покрытия
3. \`getMissingPrinciples(): CIAPrinciple[]\` - Вернуть непокрытые принципы
4. \`getRecommendations(): string[]\` - Вернуть рекомендации по безопасности`,
			hint1: `Начните с реализации addMeasure() - просто добавьте объект меры в массив. Затем getMeasuresByPrinciple() фильтрует меры по заданному принципу.`,
			hint2: `Для getScore() пройдитесь по всем трём принципам. Если getMeasuresByPrinciple возвращает хотя бы 1 меру, добавьте 33.33.`,
			whyItMatters: `Триада CIA - краеугольный камень всех практик безопасности. Понимание её помогает проектировать безопасные системы с самого начала.`
		},
		uz: {
			title: 'CIA Triad: Xavfsizlik asosi',
			description: `CIA Triad - axborot xavfsizligining uchta asosiy prinsipini o'rganing.

**CIA Triad nima?**

CIA Triad uchta asosiy prinsipdan iborat:

1. **Maxfiylik (Confidentiality)** - Ma'lumotlarni ruxsatsiz kirishdan himoya qilish
2. **Butunlik (Integrity)** - Ma'lumotlar o'zgartirilmaganligini ta'minlash
3. **Mavjudlik (Availability)** - Vakolatli foydalanuvchilar uchun kirish imkoniyatini ta'minlash

**Sizning vazifangiz:**

CIA Triad asosida xavfsizlik choralarini baholaydigan \`SecurityAnalyzer\` klassini amalga oshiring.`,
			hint1: `addMeasure() ni amalga oshirishdan boshlang - shunchaki chora obyektini massivga qo'shing.`,
			hint2: `getScore() uchun barcha uchta prinsip bo'yicha aylaning. Agar getMeasuresByPrinciple kamida 1 chora qaytarsa, 33.33 qo'shing.`,
			whyItMatters: `CIA Triad barcha xavfsizlik amaliyotlarining asosi hisoblanadi.`
		}
	}
};

export default task;
