import { Component, StrictMode, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// A render throw used to unmount the whole tree → blank white window with no clue what happened.
// This keeps the app alive: shows the error + a reload, and logs it so it's diagnosable.
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) { return { error } }
  componentDidCatch(error: Error) { console.error('[app] render crashed:', error) }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <div style={{ padding: 24, fontFamily: 'monospace', color: '#ddd', background: '#201d1d', height: '100vh', overflow: 'auto' }}>
        <div style={{ fontSize: 14, marginBottom: 8, color: '#e0a85e' }}>Something broke in the UI</div>
        <div style={{ fontSize: 13, marginBottom: 10, color: '#ffb4b4' }}>{this.state.error?.name}: {this.state.error?.message}</div>
        <pre style={{ whiteSpace: 'pre-wrap', fontSize: 11, color: '#c88' }}>{String(this.state.error?.stack || this.state.error)}</pre>
        <button onClick={() => this.setState({ error: null })} style={{ marginTop: 12, marginRight: 8 }}>Dismiss</button>
        <button onClick={() => window.location.reload()} style={{ marginTop: 12 }}>Reload</button>
      </div>
    )
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
