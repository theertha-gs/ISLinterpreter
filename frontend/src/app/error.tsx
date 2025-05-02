"use client"

import { useEffect } from "react"
import { Button } from "@/components/ui/button"
import { AlertCircle } from "lucide-react"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error("Application error:", error)
  }, [error])

  const handleRetry = () => {
    // Clear any stored auth state that might be causing issues
    localStorage.removeItem("isLoggedIn")
    reset()
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-b from-background to-muted/20">
      <Card className="max-w-md w-full">
        <CardHeader className="text-center pb-2">
          <AlertCircle className="h-12 w-12 text-destructive mx-auto mb-2" />
          <CardTitle className="text-2xl text-destructive">Something went wrong</CardTitle>
        </CardHeader>
        <CardContent className="text-center">
          <p className="text-muted-foreground">
            {error.message || "An unexpected error occurred. Please try again."}
          </p>
        </CardContent>
        <CardFooter className="flex justify-center gap-4">
          <Button
            variant="outline"
            onClick={() => window.location.reload()}
          >
            Refresh Page
          </Button>
          <Button
            onClick={handleRetry}
          >
            Try Again
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
