"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Book, Camera, Settings, LogOut } from "lucide-react"
import { useAuth } from "@/lib/auth-context"
import LoadingPage from "../loading"

export default function WelcomePage() {
  const router = useRouter()
  const { user, logout } = useAuth()
  const [isNavigating, setIsNavigating] = useState(false)

  useEffect(() => {
    console.log("Welcome page mounted")
    console.log("User state:", user ? "Authenticated" : "Not authenticated")
    console.log("localStorage isLoggedIn:", localStorage.getItem("isLoggedIn"))

    const isLoggedIn = localStorage.getItem("isLoggedIn") === "true"
    if (!isLoggedIn) {
      console.log("User not logged in, redirecting to login")
      window.location.href = "/login"
    }
  }, [user])

  // Handle navigation with loading state
  const handleNavigation = (path: string) => {
    setIsNavigating(true)
    router.push(path)
  }

  const handleLogout = async () => {
    setIsNavigating(true)
    try {
      await logout()
      window.location.href = "/login"
    } catch (error) {
      console.error("Logout failed:", error)
      setIsNavigating(false)
    }
  }

  // Show loading states
  if (isNavigating || !user) {
    return <LoadingPage />
  }

  return (
    <main className="container mx-auto min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center pb-8">
          <CardTitle className="text-4xl font-bold bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
            Welcome to ISL Translator
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-6">
          <Button 
            size="lg" 
            variant="outline"
            className="w-full flex items-center gap-2 p-6 text-lg hover:scale-105 transition-transform"
            onClick={() => handleNavigation("/guide")}
          >
            <Book className="w-5 h-5" />
            Go to Guide
          </Button>
          
          <Button 
            size="lg"
            className="w-full flex items-center gap-2 p-6 text-lg hover:scale-105 transition-transform"
            onClick={() => handleNavigation("/camera")}
          >
            <Camera className="w-5 h-5" />
            Start Camera
          </Button>
          
          <Button 
            size="lg"
            variant="secondary"
            className="w-full flex items-center gap-2 p-6 text-lg hover:scale-105 transition-transform"
            onClick={() => handleNavigation("/custom-gestures")}
          >
            <Settings className="w-5 h-5" />
            Customize Gestures
          </Button>

          <Button 
            size="lg"
            variant="destructive"
            className="w-full flex items-center gap-2 p-6 text-lg hover:scale-105 transition-transform mt-4"
            onClick={handleLogout}
          >
            <LogOut className="w-5 h-5" />
            Logout
          </Button>
        </CardContent>
      </Card>
    </main>
  )
}
