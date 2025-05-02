"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-context"
import LoadingPage from "./loading"

export default function HomePage() {
  const router = useRouter()
  const { user, loading } = useAuth()
  const [isRedirecting, setIsRedirecting] = useState(false)

  useEffect(() => {
    console.log("Root page mounted")
    console.log("Auth loading:", loading)
    console.log("User state:", user ? "Authenticated" : "Not authenticated")
    console.log("localStorage isLoggedIn:", localStorage.getItem("isLoggedIn"))

    if (!loading) {
      const isLoggedIn = localStorage.getItem("isLoggedIn") === "true"
      setIsRedirecting(true)
      
      if (isLoggedIn) {
        console.log("User is logged in, redirecting to welcome page")
        window.location.href = "/welcome"
      } else {
        console.log("User not logged in, redirecting to login page")
        window.location.href = "/login"
      }
    }
  }, [loading, user])

  // Show loading spinner during initialization and redirection
  if (loading || isRedirecting) {
    return <LoadingPage />
  }

  // This should never render as we always redirect
  return null
}
