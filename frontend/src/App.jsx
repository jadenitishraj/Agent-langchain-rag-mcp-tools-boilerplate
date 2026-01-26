import { useState, useRef, useEffect } from 'react'
import { Bot, User, Sparkles, Zap } from 'lucide-react'
import './index.css'
import MessageContent from './components/MessageContent'
import ChatInput from './components/ChatInput'

function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [streamingMessage, setStreamingMessage] = useState('')

  // Refs for smart scrolling
  const chatContainerRef = useRef(null)
  const isAtBottomRef = useRef(true)
  const lastScrollTime = useRef(0)

  // Check if user is at bottom on scroll
  const handleScroll = () => {
    if (chatContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current
      // If within 50px of bottom, consider it "at bottom"
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50
      isAtBottomRef.current = isAtBottom
    }
  }

  // Stable scroll function
  const scrollToBottom = (force = false) => {
    if (chatContainerRef.current) {
      // Only scroll if we are already at bottom OR if forced
      if (isAtBottomRef.current || force) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
      }
    }
  }

  // Scroll when new messages are added (always force scroll for new message start)
  useEffect(() => {
    scrollToBottom(true)
  }, [messages])

  // Scroll during streaming (throttled, and only if user is at bottom)
  useEffect(() => {
    const now = Date.now()
    if (now - lastScrollTime.current > 100) {
      scrollToBottom(false) // Don't force, respect user scroll
      lastScrollTime.current = now
    }
  }, [streamingMessage])

  // Convert messages to history format for API
  const getHistory = () => {
    return messages.map(msg => ({
      role: msg.type === 'user' ? 'user' : 'assistant',
      content: msg.content
    }))
  }

  // Handle User Sending Message
  const handleSendMessage = async (userQuestion) => {
    if (!userQuestion.trim() || isLoading) return

    const newMessages = [...messages, { type: 'user', content: userQuestion }]
    setMessages(newMessages)
    setIsLoading(true)
    setStreamingMessage('')

    // Force scroll to bottom when sending
    setTimeout(() => scrollToBottom(true), 0)

    // Get history BEFORE adding the new user message (since we send it as "question")
    const history = messages.map(msg => ({
      role: msg.type === 'user' ? 'user' : 'assistant',
      content: msg.content
    }))

    try {
      const response = await fetch('http://127.0.0.1:8000/agent/ask/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userQuestion,
          history: history  // Send conversation history
        }),
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let fullMessage = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))

              if (data.type === 'token') {
                fullMessage += data.content
                setStreamingMessage(fullMessage)
              } else if (data.type === 'error') {
                fullMessage = data.content
                setStreamingMessage(fullMessage)
              } else if (data.type === 'replace') {
                fullMessage = data.content
                setStreamingMessage(fullMessage)
              } else if (data.type === 'done') {
                setMessages([...newMessages, { type: 'agent', content: fullMessage }])
                setStreamingMessage('')
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error) {
      setMessages([...newMessages, { type: 'error', content: 'Unable to connect. Please try again.' }])
      setStreamingMessage('')
    } finally {
      setIsLoading(false)
    }
  }

  const clearChat = () => {
    setMessages([])
    setStreamingMessage('')
  }

  return (
    <div className="app-container">
      {/* Animated background orbs */}
      <div className="orb orb-1"></div>
      <div className="orb orb-2"></div>
      <div className="orb orb-3"></div>

      <div className="main-wrapper">

        {/* Header */}
        <header className="header">
          <div className="logo-container">
            <div className="logo-icon">
              <Zap size={24} color="white" />
            </div>
            <div className="logo-text">
              <h1>AgentForge</h1>
              <div className="status-badge">
                <span className="status-dot"></span>
                Online
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="clear-button"
              >
                Clear Chat
              </button>
            )}
            <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.5)' }}>
              {messages.length > 0 ? `${messages.length} messages` : 'RAG + Memory'}
            </div>
          </div>
        </header>

        {/* Chat Area */}
        <div
          className="chat-container"
          ref={chatContainerRef}
          onScroll={handleScroll}
        >
          <div className="messages-area">
            {messages.length === 0 && !streamingMessage && (
              <div className="empty-state">
                <div className="empty-icon">
                  <Sparkles size={40} color="#a78bfa" />
                </div>
                <h2>Explore the Codebase</h2>
                <p>
                  Ask me anything about this <span className="highlight">LangGraph + RAG + MCP</span> boilerplate.
                  I can explain code, architecture, and how everything works together.
                </p>
                <p style={{ marginTop: '12px', fontSize: '13px', color: 'rgba(255,255,255,0.4)' }}>
                  💡 I remember our conversation, so feel free to ask follow-up questions!
                </p>
              </div>
            )}

            {messages.map((msg, index) => (
              <div
                key={index}
                className={`message-row ${msg.type === 'user' ? 'user' : ''}`}
              >
                <div className={`avatar ${msg.type === 'user' ? 'user' : 'agent'}`}>
                  {msg.type === 'user' ? <User size={20} color="white" /> : <Bot size={20} color="white" />}
                </div>
                <div className={`message-bubble ${msg.type}`}>
                  <MessageContent content={msg.content} />
                </div>
              </div>
            ))}

            {/* Streaming message */}
            {streamingMessage && (
              <div className="message-row">
                <div className="avatar agent">
                  <Bot size={20} color="white" />
                </div>
                <div className="message-bubble agent streaming">
                  <div style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                    {streamingMessage}
                    <span className="cursor">▋</span>
                  </div>
                </div>
              </div>
            )}

            {/* Loading dots */}
            {isLoading && !streamingMessage && (
              <div className="message-row">
                <div className="avatar agent">
                  <Bot size={20} color="white" />
                </div>
                <div className="loading-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Input Area (Isolated Component) */}
        <ChatInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />

      </div>
    </div>
  )
}

export default App
