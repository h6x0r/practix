import React, { memo } from "react";

interface CodeBlockProps {
  language?: string;
  code: string;
}

const CodeBlock = memo(({ language, code }: CodeBlockProps) => {
  const highlightCode = (text: string): React.ReactNode[] => {
    const lines = text.split("\n");
    return lines.map((line, i) => {
      const trimmed = line.trim();

      if (trimmed.startsWith("//")) {
        return (
          <div key={i}>
            <span className="text-green-600 dark:text-green-400">{line}</span>
          </div>
        );
      }

      const commentIndex = line.indexOf("//");
      if (commentIndex > 0) {
        const codePart = line.slice(0, commentIndex);
        const commentPart = line.slice(commentIndex);
        return (
          <div key={i}>
            <span className="text-gray-800 dark:text-gray-200">{codePart}</span>
            <span className="text-green-600 dark:text-green-400">
              {commentPart}
            </span>
          </div>
        );
      }

      return (
        <div key={i} className="text-gray-800 dark:text-gray-200">
          {line}
        </div>
      );
    });
  };

  return (
    <div className="my-4 rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 shadow-sm">
      {language && (
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-[#1e1e1e] border-b border-gray-200 dark:border-gray-700">
          <span className="w-3 h-3 rounded-full bg-red-500"></span>
          <span className="w-3 h-3 rounded-full bg-yellow-500"></span>
          <span className="w-3 h-3 rounded-full bg-green-500"></span>
          <span className="ml-2 text-xs text-gray-500 dark:text-gray-400 font-mono">
            {language}
          </span>
        </div>
      )}
      <pre className="p-4 bg-gray-50 dark:bg-[#0d1117] overflow-x-auto font-mono text-xs leading-5">
        {highlightCode(code)}
      </pre>
    </div>
  );
});

// Render inline markdown: bold, inline code
const renderInline = (text: string): React.ReactNode => {
  const parts: React.ReactNode[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // Inline code `code`
    const codeMatch = remaining.match(/^`([^`]+)`/);
    if (codeMatch) {
      parts.push(
        <code
          key={key++}
          className="bg-gray-100 dark:bg-dark-bg px-1.5 py-0.5 rounded text-xs font-mono text-brand-600 dark:text-brand-400 border border-gray-200 dark:border-dark-border"
        >
          {codeMatch[1]}
        </code>,
      );
      remaining = remaining.slice(codeMatch[0].length);
      continue;
    }

    // Bold **text**
    const boldMatch = remaining.match(/^\*\*([^*]+)\*\*/);
    if (boldMatch) {
      parts.push(
        <strong
          key={key++}
          className="font-semibold text-gray-900 dark:text-white"
        >
          {boldMatch[1]}
        </strong>,
      );
      remaining = remaining.slice(boldMatch[0].length);
      continue;
    }

    // Regular text until next special char
    const textMatch = remaining.match(/^[^`*]+/);
    if (textMatch) {
      parts.push(<span key={key++}>{textMatch[0]}</span>);
      remaining = remaining.slice(textMatch[0].length);
      continue;
    }

    // Single special char
    parts.push(<span key={key++}>{remaining[0]}</span>);
    remaining = remaining.slice(1);
  }

  return <>{parts}</>;
};

export const DescriptionRenderer = memo(({ text }: { text: string }) => {
  const elements: React.ReactNode[] = [];
  let key = 0;

  // First, extract code blocks and replace with placeholders
  const codeBlocks: { language: string; code: string }[] = [];
  let processedText = text.replace(
    /```(\w*)\n([\s\S]*?)```/g,
    (_, lang, code) => {
      codeBlocks.push({ language: lang || "code", code: code.trim() });
      return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
    },
  );

  // Split by lines and process each
  const lines = processedText.split("\n");

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Empty line = paragraph break
    if (line.trim() === "") {
      elements.push(<div key={key++} className="h-3" />);
      continue;
    }

    // Code block placeholder
    const codeBlockMatch = line.match(/^__CODE_BLOCK_(\d+)__$/);
    if (codeBlockMatch) {
      const idx = parseInt(codeBlockMatch[1]);
      const block = codeBlocks[idx];
      elements.push(
        <div key={key++}>
          <CodeBlock language={block.language} code={block.code} />
        </div>,
      );
      continue;
    }

    // Markdown headers: # H1, ## H2, ### H3, #### H4
    // Skip first H1 if it's at the beginning (duplicates task title in header)
    const headerMatch = line.match(/^(#{1,4})\s+(.+)$/);
    if (headerMatch) {
      const level = headerMatch[1].length;
      const content = headerMatch[2];

      // Skip first H1 at the very start of description (title is already in header)
      if (level === 1 && elements.length === 0) {
        continue;
      }

      const headerStyles: Record<number, string> = {
        1: "font-bold text-gray-900 dark:text-white text-lg mt-4 mb-3",
        2: "font-bold text-gray-900 dark:text-white text-base mt-4 mb-2",
        3: "font-semibold text-gray-900 dark:text-white text-sm mt-4 mb-2",
        4: "font-semibold text-gray-800 dark:text-gray-200 text-sm mt-3 mb-1.5",
      };
      const Tag = `h${level}` as "h1" | "h2" | "h3" | "h4";
      elements.push(
        <Tag key={key++} className={headerStyles[level]}>
          {content}
        </Tag>,
      );
      continue;
    }

    // Header **Text:**
    const boldHeaderMatch = line.match(/^\*\*([^*]+):\*\*$/);
    if (boldHeaderMatch) {
      elements.push(
        <h4
          key={key++}
          className="font-bold text-gray-900 dark:text-white mt-6 mb-2 text-sm"
        >
          {boldHeaderMatch[1]}:
        </h4>,
      );
      continue;
    }

    // Numbered list: 1. text
    const numberedMatch = line.match(/^(\d+)\. (.+)$/);
    if (numberedMatch) {
      elements.push(
        <div key={key++} className="flex gap-2 ml-2 my-1.5">
          <span className="text-gray-500 dark:text-gray-400 font-medium text-sm flex-shrink-0 leading-7">
            {numberedMatch[1]}.
          </span>
          <span className="flex-1 leading-7">
            {renderInline(numberedMatch[2])}
          </span>
        </div>,
      );
      continue;
    }

    // Bullet list: - text
    const bulletMatch = line.match(/^- (.+)$/);
    if (bulletMatch) {
      elements.push(
        <div key={key++} className="flex gap-2 ml-4 my-1">
          <span className="w-1.5 h-1.5 rounded-full bg-gray-400 flex-shrink-0 mt-[11px]"></span>
          <span className="flex-1 leading-7">
            {renderInline(bulletMatch[1])}
          </span>
        </div>,
      );
      continue;
    }

    // Regular paragraph with inline formatting
    elements.push(
      <p key={key++} className="my-1">
        {renderInline(line)}
      </p>,
    );
  }

  return (
    <div className="text-sm leading-7 text-gray-700 dark:text-gray-300">
      {elements}
    </div>
  );
});
