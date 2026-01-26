import React, { useState } from 'react';
import { Send } from 'lucide-react';

const ChatInput = ({ onSendMessage, isLoading }) => {
    const [question, setQuestion] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!question.trim() || isLoading) return;

        onSendMessage(question);
        setQuestion('');
    };

    return (
        <form onSubmit={handleSubmit} className="input-form">
            <div className="input-container">
                <div className="input-inner">
                    <input
                        type="text"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        placeholder="Ask about the code, architecture, RAG pipeline..."
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        className="send-button"
                        disabled={isLoading || !question.trim()}
                    >
                        <Send size={22} />
                    </button>
                </div>
            </div>
        </form>
    );
};

export default ChatInput;
