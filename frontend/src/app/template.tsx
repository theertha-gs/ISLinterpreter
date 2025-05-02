"use client"

import { useEffect, useState } from "react"
import LoadingPage from "./loading"

export default function Template({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Add a small delay to prevent flash of loading state for fast transitions
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 200)

    return () => clearTimeout(timer)
  }, [])

  // Show loading state while transitioning between pages
  if (isLoading) {
    return <LoadingPage />
  }

  return children
}
