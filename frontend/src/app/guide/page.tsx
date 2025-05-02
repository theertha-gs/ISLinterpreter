"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { ArrowLeft, Hand, Camera, Lightbulb } from "lucide-react"
import { useAuth } from "@/lib/auth-context"
import LoadingPage from "../loading"

export default function GuidePage() {
  const router = useRouter()
  const { user } = useAuth()
  const [isNavigating, setIsNavigating] = useState(false)

  useEffect(() => {
    console.log("Guide page mounted")
    console.log("User state:", user ? "Authenticated" : "Not authenticated")
    
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

  // Show loading states
  if (isNavigating || !user) {
    return <LoadingPage />
  }

  return (
    <main className="container mx-auto py-6 px-4">
      <Button 
        variant="ghost" 
        className="mb-6 hover:bg-muted/50 transition-colors"
        onClick={() => handleNavigation("/welcome")}
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to Welcome
      </Button>

      <Card className="border border-muted">
        <CardHeader>
          <CardTitle className="text-3xl font-bold bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
            How to Use ISL Translator
          </CardTitle>
          <CardDescription>
            Follow these steps to get started with sign language translation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-8">
          {/* Step 1 */}
          <div className="flex items-start gap-4 group cursor-default hover:bg-muted/50 p-4 rounded-lg transition-colors">
            <Camera className="w-8 h-8 text-primary shrink-0 mt-1 group-hover:scale-110 transition-transform" />
            <div>
              <h3 className="text-lg font-semibold mb-1">Position Your Camera</h3>
              <p className="text-muted-foreground">
                Ensure your camera has a clear view of your hands. Good lighting and a plain background work best.
              </p>
            </div>
          </div>

          {/* Step 2 */}
          <div className="flex items-start gap-4 group cursor-default hover:bg-muted/50 p-4 rounded-lg transition-colors">
            <Hand className="w-8 h-8 text-primary shrink-0 mt-1 group-hover:scale-110 transition-transform" />
            <div>
              <h3 className="text-lg font-semibold mb-1">Make Signs</h3>
              <p className="text-muted-foreground">
                Form hand signs clearly and hold them steady for a moment. Keep your hands within the camera frame.
              </p>
            </div>
          </div>

          {/* Step 3 */}
          <div className="flex items-start gap-4 group cursor-default hover:bg-muted/50 p-4 rounded-lg transition-colors">
            <Lightbulb className="w-8 h-8 text-primary shrink-0 mt-1 group-hover:scale-110 transition-transform" />
            <div>
              <h3 className="text-lg font-semibold mb-1">View Translation</h3>
              <p className="text-muted-foreground">
                The system will detect your signs and display the translation in real-time. Practice with simple signs first.
              </p>
            </div>
          </div>

          <div className="pt-4">
            <Button
              size="lg"
              className="w-full hover:scale-[1.02] transition-transform"
              onClick={() => handleNavigation("/camera")}
            >
              <Camera className="mr-2 h-5 w-5" />
              Start Using Camera
            </Button>
          </div>
        </CardContent>
      </Card>
    </main>
  )
}
