"use client"

import { useEffect, useState } from "react"
import type { Gesture } from "@/types/gesture"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { ArrowLeft, Plus, Library, Loader2, Trash2 } from "lucide-react"
import { useAuth } from "@/lib/auth-context"
import LoadingPage from "../loading"
import { toast } from "sonner"

export default function CustomGesturesPage() {
  const router = useRouter()
  const { user } = useAuth()
  const [isNavigating, setIsNavigating] = useState(false)
  const [gestures, setGestures] = useState<Gesture[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isDeleting, setIsDeleting] = useState<string | null>(null)

  // Fetch gestures
  const fetchGestures = async () => {
    try {
      const response = await fetch("/api/custom-gestures")
      if (!response.ok) throw new Error("Failed to fetch gestures")
      const data = await response.json()
      setGestures(data.gestures || [])
    } catch (error) {
      console.error("Error fetching gestures:", error)
      setGestures([])
    } finally {
      setIsLoading(false)
    }
  }

  // Handle gesture deletion
  const handleDeleteGesture = async (id: string) => {
    try {
      setIsDeleting(id)
      const response = await fetch(`/api/custom-gestures?id=${id}`, {
        method: 'DELETE',
      })

      if (!response.ok) {
        throw new Error("Failed to delete gesture")
      }

      // Remove gesture from local state
      setGestures(prev => prev.filter(g => g.id !== id))
      toast.success("Gesture deleted successfully")
    } catch (error) {
      console.error("Error deleting gesture:", error)
      toast.error("Failed to delete gesture")
    } finally {
      setIsDeleting(null)
    }
  }

  useEffect(() => {
    fetchGestures()
    console.log("Custom gestures page mounted")
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
  if (isNavigating || !user || isLoading) {
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
            Custom Gestures
          </CardTitle>
          <CardDescription>
            Create and manage your own custom sign language gestures
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Empty State or Gesture List */}
          {gestures.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Library className="w-16 h-16 text-primary/20 mb-4" />
              <h3 className="text-xl font-semibold mb-2">No Custom Gestures</h3>
              <p className="text-muted-foreground max-w-md mb-6">
                You haven't created any custom gestures yet. Click the button below to create your first gesture.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {gestures.map((gesture) => (
                <Card key={gesture.id} className="flex flex-col relative group">
                  <CardHeader>
                    <CardTitle className="text-lg">{gesture.name}</CardTitle>
                    <CardDescription>
                      Created {new Date(gesture.createdAt).toLocaleDateString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="relative">
                    <img 
                      src={gesture.firstFramePath} 
                      alt={`First frame of ${gesture.name}`}
                      className="w-full h-48 object-cover rounded-md"
                    />
                    <Button
                      variant="destructive"
                      size="sm"
                      className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={() => handleDeleteGesture(gesture.id)}
                      disabled={isDeleting === gesture.id}
                    >
                      {isDeleting === gesture.id ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin mr-2" />
                          Deleting...
                        </>
                      ) : (
                        <>
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
          
          {/* Add Custom Gesture Button */}
          <Button 
            size="lg"
            className="w-full"
            onClick={() => handleNavigation("/custom-gesture")}
          >
            <Plus className="mr-2 h-5 w-5" />
            Add Custom Gesture
          </Button>
        </CardContent>
      </Card>
    </main>
  )
}
