import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Copy, Check, FileCode } from 'lucide-react';

/**
 * MessageContent Component
 * Renders markdown content with code blocks and copy functionality.
 * Optimized for stability.
 */

// Custom code block component with copy button
const CodeBlock = ({ language, value }) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        await navigator.clipboard.writeText(value);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const normalizedLang = language || 'text';

    return (
        <div className="code-block-container">
            <div className="code-block-header">
                <div className="code-block-language">
                    <FileCode size={14} />
                    <span>{normalizedLang}</span>
                </div>
                <button
                    onClick={handleCopy}
                    className="copy-button"
                    title="Copy code"
                >
                    {copied ? (
                        <>
                            <Check size={14} />
                            <span>Copied!</span>
                        </>
                    ) : (
                        <>
                            <Copy size={14} />
                            <span>Copy</span>
                        </>
                    )}
                </button>
            </div>
            <pre className="simple-code-block">
                <code>{value}</code>
            </pre>
        </div>
    );
};

// Inline code component
const InlineCode = ({ children }) => {
    return (
        <code className="inline-code">
            {children}
        </code>
    );
};

const MessageContent = ({ content }) => {
    if (!content) return null;

    return (
        <div className="message-content">
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                    // Code blocks
                    code({ node, inline, className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || '');
                        const value = String(children).replace(/\n$/, '');

                        if (!inline && (match || value.includes('\n'))) {
                            return (
                                <CodeBlock
                                    language={match ? match[1] : 'text'}
                                    value={value}
                                />
                            );
                        }

                        return <InlineCode {...props}>{children}</InlineCode>;
                    },
                    // Links
                    a({ node, children, href, ...props }) {
                        return (
                            <a
                                href={href}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="message-link"
                                {...props}
                            >
                                {children}
                            </a>
                        );
                    },
                    // Headings
                    h1: ({ children }) => <h1 className="message-h1">{children}</h1>,
                    h2: ({ children }) => <h2 className="message-h2">{children}</h2>,
                    h3: ({ children }) => <h3 className="message-h3">{children}</h3>,
                    // Lists
                    ul: ({ children }) => <ul className="message-ul">{children}</ul>,
                    ol: ({ children }) => <ol className="message-ol">{children}</ol>,
                    li: ({ children }) => <li className="message-li">{children}</li>,
                    // Blockquote
                    blockquote: ({ children }) => (
                        <blockquote className="message-blockquote">{children}</blockquote>
                    ),
                    // Paragraphs
                    p: ({ children }) => <p className="message-p">{children}</p>,
                }}
            >
                {content}
            </ReactMarkdown>
        </div>
    );
};

export default MessageContent;
