import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-rag-prompting',
	title: 'RAG Prompting',
	difficulty: 'hard',
	tags: ['rag', 'prompting', 'retrieval'],
	estimatedTime: '20m',
	isPremium: true,
	order: 4,
	description: `# RAG Prompting

Create prompts for Retrieval-Augmented Generation systems.

## What is RAG?

RAG combines:
1. **Retrieval**: Find relevant documents
2. **Augmentation**: Add context to prompt
3. **Generation**: LLM generates answer

## Prompt Structure

\`\`\`
[System instruction]
[Retrieved context]
[User question]
[Answer guidelines]
\`\`\`

## Example

\`\`\`python
prompt = """
Answer based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""
\`\`\``,

	initialCode: `def create_rag_prompt(context: str, question: str,
                       system_prompt: str = None) -> str:
    """Create a RAG prompt with retrieved context."""
    # Your code here
    pass

def format_context_chunks(chunks: list, max_tokens: int = 2000) -> str:
    """Format multiple retrieved chunks into context."""
    # Your code here
    pass

def create_citation_prompt(context: str, question: str) -> str:
    """Create a RAG prompt that includes citations."""
    # Your code here
    pass

def create_multi_doc_prompt(documents: list, question: str) -> str:
    """Create prompt for multi-document RAG."""
    # Your code here
    pass

def create_conversational_rag_prompt(context: str, question: str,
                                      history: list) -> str:
    """Create RAG prompt with conversation history."""
    # Your code here
    pass
`,

	solutionCode: `def create_rag_prompt(context: str, question: str,
                       system_prompt: str = None) -> str:
    """Create a RAG prompt with retrieved context."""
    if system_prompt is None:
        system_prompt = """Answer the question based ONLY on the provided context.
If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question."
Do not make up information."""

    return f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""

def format_context_chunks(chunks: list, max_tokens: int = 2000) -> str:
    """Format multiple retrieved chunks into context."""
    context_parts = []
    current_length = 0

    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk.get("text", str(chunk))
        source = chunk.get("source", f"Document {i}")
        score = chunk.get("score", None)

        formatted = f"[{source}]\\n{chunk_text}"

        # Rough token estimate (4 chars per token)
        chunk_tokens = len(formatted) // 4

        if current_length + chunk_tokens > max_tokens:
            break

        context_parts.append(formatted)
        current_length += chunk_tokens

    return "\\n\\n---\\n\\n".join(context_parts)

def create_citation_prompt(context: str, question: str) -> str:
    """Create a RAG prompt that includes citations."""
    return f"""Answer the question using the provided context. Include citations in [Source] format.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- Cite sources using [Source] format after each claim
- If information is not in the context, say so

Answer with citations:"""

def create_multi_doc_prompt(documents: list, question: str) -> str:
    """Create prompt for multi-document RAG."""
    doc_sections = []
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", f"Document {i}")
        content = doc.get("content", str(doc))
        doc_sections.append(f"### {title}\\n{content}")

    all_docs = "\\n\\n".join(doc_sections)

    return f"""You have access to multiple documents. Use them to answer the question.

Documents:
{all_docs}

Question: {question}

Synthesize information from relevant documents to provide a comprehensive answer:"""

def create_conversational_rag_prompt(context: str, question: str,
                                      history: list) -> str:
    """Create RAG prompt with conversation history."""
    history_text = ""
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        history_text += f"{role.capitalize()}: {content}\\n"

    return f"""Use the context and conversation history to answer the question.

Context:
{context}

Conversation History:
{history_text}

Current Question: {question}

Answer:"""
`,

	testCode: `import unittest

class TestRAGPrompting(unittest.TestCase):
    def test_create_rag_prompt_basic(self):
        result = create_rag_prompt("Python is a language", "What is Python?")
        self.assertIn("Python is a language", result)
        self.assertIn("What is Python?", result)

    def test_create_rag_prompt_custom_system(self):
        result = create_rag_prompt("context", "question", system_prompt="Be brief")
        self.assertIn("Be brief", result)

    def test_format_context_chunks(self):
        chunks = [
            {"text": "First chunk", "source": "Doc1"},
            {"text": "Second chunk", "source": "Doc2"}
        ]
        result = format_context_chunks(chunks)
        self.assertIn("First chunk", result)
        self.assertIn("[Doc1]", result)

    def test_create_citation_prompt(self):
        result = create_citation_prompt("Some context", "What?")
        self.assertIn("citation", result.lower())
        self.assertIn("[Source]", result)

    def test_create_conversational_rag_prompt(self):
        history = [{"role": "user", "content": "Hello"}]
        result = create_conversational_rag_prompt("context", "question", history)
        self.assertIn("Hello", result)
        self.assertIn("Conversation History", result)

    def test_rag_prompt_returns_string(self):
        result = create_rag_prompt("ctx", "q")
        self.assertIsInstance(result, str)

    def test_format_chunks_includes_separator(self):
        chunks = [{"text": "A", "source": "S1"}, {"text": "B", "source": "S2"}]
        result = format_context_chunks(chunks)
        self.assertIn("---", result)

    def test_multi_doc_prompt(self):
        docs = [{"title": "Title1", "content": "Content1"}]
        result = create_multi_doc_prompt(docs, "What?")
        self.assertIn("Title1", result)
        self.assertIn("Content1", result)

    def test_multi_doc_includes_documents_header(self):
        docs = [{"title": "T", "content": "C"}]
        result = create_multi_doc_prompt(docs, "Q")
        self.assertIn("Documents:", result)

    def test_citation_prompt_includes_instructions(self):
        result = create_citation_prompt("context", "question")
        self.assertIn("Instructions:", result)
`,

	hint1: 'Always instruct the model to answer only from provided context',
	hint2: 'Format sources clearly so citations are accurate',

	whyItMatters: `RAG is the foundation of modern LLM applications:

- **Grounded responses**: Reduces hallucinations
- **Up-to-date**: Use current information
- **Domain-specific**: Custom knowledge bases
- **Verifiable**: Cite sources for fact-checking

Used in ChatGPT plugins, enterprise search, and documentation bots.`,

	translations: {
		ru: {
			title: 'RAG промптинг',
			description: `# RAG промптинг

Создавайте промпты для систем Retrieval-Augmented Generation.

## Что такое RAG?

RAG комбинирует:
1. **Retrieval**: Поиск релевантных документов
2. **Augmentation**: Добавление контекста в промпт
3. **Generation**: LLM генерирует ответ

## Пример

\`\`\`python
prompt = """
Answer based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""
\`\`\``,
			hint1: 'Всегда инструктируйте модель отвечать только из предоставленного контекста',
			hint2: 'Форматируйте источники четко для точных цитат',
			whyItMatters: `RAG - основа современных LLM приложений:

- **Обоснованные ответы**: Уменьшает галлюцинации
- **Актуальность**: Использование текущей информации
- **Доменная специфика**: Кастомные базы знаний
- **Проверяемость**: Цитирование источников для фактчекинга`,
		},
		uz: {
			title: 'RAG prompting',
			description: `# RAG prompting

Retrieval-Augmented Generation tizimlari uchun promptlar yarating.

## RAG nima?

RAG birlashtiradi:
1. **Retrieval**: Tegishli hujjatlarni topish
2. **Augmentation**: Promptga kontekst qo'shish
3. **Generation**: LLM javob yaratadi

## Misol

\`\`\`python
prompt = """
Answer based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""
\`\`\``,
			hint1: "Modelga faqat taqdim etilgan kontekstdan javob berishni har doim ko'rsatma bering",
			hint2: "Sitatlar aniq bo'lishi uchun manbalarni aniq formatlang",
			whyItMatters: `RAG zamonaviy LLM ilovalarining asosi:

- **Asoslangan javoblar**: Gallyutsinatsiyalarni kamaytiradi
- **Dolzarb**: Joriy ma'lumotlardan foydalanish
- **Sohaga xos**: Maxsus bilim bazalari
- **Tekshirilishi mumkin**: Fakt-checking uchun manbalarni keltirish`,
		},
	},
};

export default task;
