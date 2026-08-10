import React, { useState, useEffect } from 'react'
import ShowcaseConsole from './components/ShowcaseConsole'
import DiagnosticDashboard from './components/DiagnosticDashboard'

export default function App() {
  const getInitialView = () => {
    if (window.location.hash === '#dashboard') {
      return 'dashboard'
    }
    return 'showcase'
  }

  const [activeView, setActiveView] = useState(getInitialView)

  useEffect(() => {
    const handleHashChange = () => {
      if (window.location.hash === '#dashboard') {
        setActiveView('dashboard')
      } else {
        setActiveView('showcase')
      }
    }

    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  const handleOpenDashboard = () => {
    setActiveView('dashboard')
    window.location.hash = 'dashboard'
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleBackToShowcase = () => {
    setActiveView('showcase')
    window.location.hash = 'showcase'
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen bg-[#05070B] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
      {activeView === 'showcase' ? (
        <ShowcaseConsole onOpenDashboard={handleOpenDashboard} />
      ) : (
        <DiagnosticDashboard onBackToShowcase={handleBackToShowcase} />
      )}
    </div>
  )
}
