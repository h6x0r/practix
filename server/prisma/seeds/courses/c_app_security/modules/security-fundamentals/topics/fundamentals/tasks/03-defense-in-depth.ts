import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-defense-in-depth',
	title: 'Defense in Depth: Layered Security',
	difficulty: 'easy',
	tags: ['security', 'fundamentals', 'defense-in-depth', 'typescript'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn the Defense in Depth principle - multiple layers of security controls.

**What is Defense in Depth?**

Defense in Depth is a security strategy that uses multiple layers of security controls. If one layer fails, others continue to provide protection - like castle walls, moats, and guards.

**Security Layers:**

1. **Physical** - Locks, cameras, access cards
2. **Network** - Firewalls, IDS/IPS, segmentation
3. **Host** - OS hardening, antivirus, patches
4. **Application** - Input validation, authentication, encryption
5. **Data** - Encryption at rest, access controls, backups

**Your Task:**

Implement a \`SecurityLayerAnalyzer\` class that:

1. Evaluates security controls at each layer
2. Identifies gaps in the security posture
3. Calculates overall security score
4. Recommends improvements

**Example Usage:**

\`\`\`typescript
const analyzer = new SecurityLayerAnalyzer();

analyzer.addControl('network', { name: 'Firewall', effectiveness: 0.8 });
analyzer.addControl('application', { name: 'Input Validation', effectiveness: 0.9 });

const analysis = analyzer.analyze();
// { score: 0.72, gaps: ['No host-level controls'], recommendations: [...] }
\`\`\``,
	initialCode: `interface SecurityControl {
  name: string;
  effectiveness: number; // 0-1
  description?: string;
}

type SecurityLayer = 'physical' | 'network' | 'host' | 'application' | 'data';

interface LayerAnalysis {
  layer: SecurityLayer;
  controls: SecurityControl[];
  score: number;
  hasControls: boolean;
}

interface FullAnalysis {
  layers: LayerAnalysis[];
  overallScore: number;
  gaps: string[];
  recommendations: string[];
}

class SecurityLayerAnalyzer {
  private controls: Map<SecurityLayer, SecurityControl[]> = new Map();
  private readonly LAYERS: SecurityLayer[] = ['physical', 'network', 'host', 'application', 'data'];

  addControl(layer: SecurityLayer, control: SecurityControl): void {
    // TODO: Add a security control to a layer
  }

  removeControl(layer: SecurityLayer, controlName: string): boolean {
    // TODO: Remove a control by name
    return false;
  }

  getLayerScore(layer: SecurityLayer): number {
    // TODO: Calculate score for a single layer
    // Average effectiveness of controls, or 0 if no controls
    return 0;
  }

  analyze(): FullAnalysis {
    // TODO: Perform full security analysis
    // Calculate scores, identify gaps, make recommendations
    return {
      layers: [],
      overallScore: 0,
      gaps: [],
      recommendations: [],
    };
  }

  getRecommendations(layer: SecurityLayer): string[] {
    // TODO: Get specific recommendations for a layer
    return [];
  }

  hasMinimumControls(): boolean {
    // TODO: Check if at least 3 layers have controls
    return false;
  }
}

export { SecurityLayerAnalyzer, SecurityControl, SecurityLayer, FullAnalysis };`,
	solutionCode: `interface SecurityControl {
  name: string;
  effectiveness: number; // 0-1
  description?: string;
}

type SecurityLayer = 'physical' | 'network' | 'host' | 'application' | 'data';

interface LayerAnalysis {
  layer: SecurityLayer;
  controls: SecurityControl[];
  score: number;
  hasControls: boolean;
}

interface FullAnalysis {
  layers: LayerAnalysis[];
  overallScore: number;
  gaps: string[];
  recommendations: string[];
}

class SecurityLayerAnalyzer {
  private controls: Map<SecurityLayer, SecurityControl[]> = new Map();
  private readonly LAYERS: SecurityLayer[] = ['physical', 'network', 'host', 'application', 'data'];

  private readonly LAYER_RECOMMENDATIONS: Record<SecurityLayer, string[]> = {
    physical: ['Add access card systems', 'Install security cameras', 'Use locked server rooms'],
    network: ['Configure firewall rules', 'Implement network segmentation', 'Deploy IDS/IPS'],
    host: ['Enable automatic OS updates', 'Install endpoint protection', 'Harden OS configurations'],
    application: ['Implement input validation', 'Add authentication/authorization', 'Use secure coding practices'],
    data: ['Encrypt data at rest', 'Implement access controls', 'Set up regular backups'],
  };

  addControl(layer: SecurityLayer, control: SecurityControl): void {
    const layerControls = this.controls.get(layer) || [];
    layerControls.push(control);
    this.controls.set(layer, layerControls);
  }

  removeControl(layer: SecurityLayer, controlName: string): boolean {
    const layerControls = this.controls.get(layer);
    if (!layerControls) return false;

    const index = layerControls.findIndex(c => c.name === controlName);
    if (index === -1) return false;

    layerControls.splice(index, 1);
    return true;
  }

  getLayerScore(layer: SecurityLayer): number {
    const layerControls = this.controls.get(layer) || [];
    if (layerControls.length === 0) return 0;

    const totalEffectiveness = layerControls.reduce((sum, c) => sum + c.effectiveness, 0);
    return totalEffectiveness / layerControls.length;
  }

  analyze(): FullAnalysis {
    const layers: LayerAnalysis[] = [];
    const gaps: string[] = [];
    const recommendations: string[] = [];

    for (const layer of this.LAYERS) {
      const layerControls = this.controls.get(layer) || [];
      const score = this.getLayerScore(layer);

      layers.push({
        layer,
        controls: layerControls,
        score,
        hasControls: layerControls.length > 0,
      });

      if (layerControls.length === 0) {
        gaps.push(\`No controls at \${layer} layer\`);
        recommendations.push(...this.getRecommendations(layer).slice(0, 1));
      } else if (score < 0.5) {
        gaps.push(\`Weak controls at \${layer} layer (score: \${(score * 100).toFixed(0)}%)\`);
        recommendations.push(\`Improve \${layer} controls effectiveness\`);
      }
    }

    const layersWithControls = layers.filter(l => l.hasControls);
    const overallScore = layersWithControls.length > 0
      ? layersWithControls.reduce((sum, l) => sum + l.score, 0) / this.LAYERS.length
      : 0;

    if (!this.hasMinimumControls()) {
      recommendations.unshift('Add controls to at least 3 security layers');
    }

    return { layers, overallScore, gaps, recommendations };
  }

  getRecommendations(layer: SecurityLayer): string[] {
    return this.LAYER_RECOMMENDATIONS[layer] || [];
  }

  hasMinimumControls(): boolean {
    let layersWithControls = 0;
    for (const layer of this.LAYERS) {
      if ((this.controls.get(layer) || []).length > 0) {
        layersWithControls++;
      }
    }
    return layersWithControls >= 3;
  }
}

export { SecurityLayerAnalyzer, SecurityControl, SecurityLayer, FullAnalysis };`,
	hint1: `For addControl, get the existing array from the Map (or create empty), push the new control, and set it back in the Map.`,
	hint2: `For analyze, iterate through all LAYERS, calculate scores, identify gaps (layers with no controls or low scores), and collect recommendations.`,
	testCode: `import { SecurityLayerAnalyzer } from './solution';

// Test1: addControl adds to layer
test('Test1', () => {
  const analyzer = new SecurityLayerAnalyzer();
  analyzer.addControl('network', { name: 'Firewall', effectiveness: 0.8 });
  expect(analyzer.getLayerScore('network')).toBeCloseTo(0.8);
});

// Test2: Empty layer has score 0
test('Test2', () => {
  const analyzer = new SecurityLayerAnalyzer();
  expect(analyzer.getLayerScore('physical')).toBe(0);
});

// Test3: Multiple controls averaged
test('Test3', () => {
  const analyzer = new SecurityLayerAnalyzer();
  analyzer.addControl('application', { name: 'Auth', effectiveness: 0.8 });
  analyzer.addControl('application', { name: 'Validation', effectiveness: 0.6 });
  expect(analyzer.getLayerScore('application')).toBeCloseTo(0.7);
});

// Test4: removeControl works
test('Test4', () => {
  const analyzer = new SecurityLayerAnalyzer();
  analyzer.addControl('network', { name: 'Firewall', effectiveness: 0.8 });
  expect(analyzer.removeControl('network', 'Firewall')).toBe(true);
  expect(analyzer.getLayerScore('network')).toBe(0);
});

// Test5: analyze identifies gaps
test('Test5', () => {
  const analyzer = new SecurityLayerAnalyzer();
  analyzer.addControl('network', { name: 'Firewall', effectiveness: 0.8 });
  const analysis = analyzer.analyze();
  expect(analysis.gaps.length).toBeGreaterThan(0);
  expect(analysis.gaps.some(g => g.includes('physical'))).toBe(true);
});

// Test6: hasMinimumControls returns false with < 3 layers
test('Test6', () => {
  const analyzer = new SecurityLayerAnalyzer();
  analyzer.addControl('network', { name: 'Firewall', effectiveness: 0.8 });
  analyzer.addControl('application', { name: 'Auth', effectiveness: 0.9 });
  expect(analyzer.hasMinimumControls()).toBe(false);
});

// Test7: hasMinimumControls returns true with >= 3 layers
test('Test7', () => {
  const analyzer = new SecurityLayerAnalyzer();
  analyzer.addControl('network', { name: 'Firewall', effectiveness: 0.8 });
  analyzer.addControl('application', { name: 'Auth', effectiveness: 0.9 });
  analyzer.addControl('data', { name: 'Encryption', effectiveness: 0.95 });
  expect(analyzer.hasMinimumControls()).toBe(true);
});

// Test8: getRecommendations returns array
test('Test8', () => {
  const analyzer = new SecurityLayerAnalyzer();
  const recs = analyzer.getRecommendations('network');
  expect(recs.length).toBeGreaterThan(0);
  expect(recs.some(r => r.toLowerCase().includes('firewall'))).toBe(true);
});

// Test9: Overall score calculation
test('Test9', () => {
  const analyzer = new SecurityLayerAnalyzer();
  analyzer.addControl('network', { name: 'Firewall', effectiveness: 1.0 });
  analyzer.addControl('application', { name: 'Auth', effectiveness: 1.0 });
  const analysis = analyzer.analyze();
  expect(analysis.overallScore).toBeCloseTo(0.4); // 2/5 layers
});

// Test10: removeControl returns false for non-existent
test('Test10', () => {
  const analyzer = new SecurityLayerAnalyzer();
  expect(analyzer.removeControl('network', 'NonExistent')).toBe(false);
});`,
	whyItMatters: `Defense in Depth has prevented countless breaches by ensuring no single point of failure.

**Real-World Example: Target Breach (2013)**

\`\`\`
What Failed:
- Network: Vendor access not segmented
- Host: Malware not detected
- Application: POS systems unpatched
- Data: Card data not encrypted in transit

What Could Have Helped:
Layer 1: Network segmentation → Vendor isolated
Layer 2: Better endpoint detection → Malware caught
Layer 3: Application whitelisting → Unauthorized code blocked
Layer 4: End-to-end encryption → Data useless if stolen

Result: 40 million cards stolen, $162M+ costs
\`\`\`

**Defense in Depth Architecture:**

\`\`\`
                    [Internet]
                        │
                   ┌────┴────┐
                   │ Firewall │  ← Network Layer
                   └────┬────┘
                        │
                   ┌────┴────┐
                   │  WAF    │  ← Application Layer
                   └────┬────┘
                        │
                   ┌────┴────┐
                   │ Load    │
                   │ Balancer│
                   └────┬────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
   │ App 1   │    │ App 2   │    │ App 3   │ ← Host Layer
   │(hardened)│   │(hardened)│   │(hardened)│
   └────┬────┘    └────┬────┘    └────┬────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                   ┌────┴────┐
                   │Database │  ← Data Layer
                   │(encrypted)│
                   └─────────┘
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Эшелонированная защита: Многоуровневая безопасность',
			description: `Изучите принцип Defense in Depth - множественные уровни защиты.

**Что такое Defense in Depth?**

Эшелонированная защита - стратегия безопасности с несколькими уровнями контроля. Если один уровень падает, другие продолжают защищать.

**Ваша задача:**

Реализуйте класс \`SecurityLayerAnalyzer\`:

1. Оценка контролей безопасности на каждом уровне
2. Выявление пробелов в защите
3. Расчёт общего балла безопасности
4. Рекомендации по улучшению`,
			hint1: `Для addControl получите существующий массив из Map (или создайте пустой), добавьте контроль, сохраните обратно.`,
			hint2: `Для analyze пройдите по всем LAYERS, вычислите баллы, выявите пробелы, соберите рекомендации.`,
			whyItMatters: `Эшелонированная защита предотвратила бесчисленные взломы, исключая единую точку отказа.`
		},
		uz: {
			title: 'Chuqur himoya: Ko\'p qatlamli xavfsizlik',
			description: `Defense in Depth printsipini o'rganing - bir necha xavfsizlik nazorati qatlamlari.

**Defense in Depth nima?**

Chuqur himoya - bir necha xavfsizlik nazorati qatlamlarini ishlatadigan xavfsizlik strategiyasi.

**Sizning vazifangiz:**

\`SecurityLayerAnalyzer\` klassini amalga oshiring:

1. Har bir qatlamda xavfsizlik nazoratlarini baholash
2. Xavfsizlik holatidagi bo'shliqlarni aniqlash
3. Umumiy xavfsizlik ballini hisoblash
4. Yaxshilashlarni tavsiya qilish`,
			hint1: `addControl uchun Map dan mavjud massivni oling (yoki bo'sh yarating), nazoratni qo'shing, qaytadan saqlang.`,
			hint2: `analyze uchun barcha LAYERS bo'ylab yuring, ballarni hisoblang, bo'shliqlarni aniqlang, tavsiyalarni to'plang.`,
			whyItMatters: `Chuqur himoya yagona nosozlik nuqtasini yo'q qilish orqali son-sanoqsiz buzilishlarni oldini oldi.`
		}
	}
};

export default task;
