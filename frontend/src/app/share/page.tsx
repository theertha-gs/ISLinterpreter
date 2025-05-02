"use client"

import { useEffect, useState } from "react"
import { useSearchParams } from "next/navigation"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Hand, ArrowLeft } from "lucide-react"
import Link from "next/link"

export default function SharePage() {
  const searchParams = useSearchParams()
  const [gesture, setGesture] = useState<string>("")
  
  useEffect(() => {
    const sharedGesture = searchParams.get("gesture")
    if (sharedGesture) {
      setGesture(decodeURIComponent(sharedGesture))
    }
  }, [searchParams])

  if (!gesture) {
    return (
      <main className="container mx-auto py-4 px-4 min-h-screen flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle>Invalid Share Link</CardTitle>
            <CardDescription>No gesture was found in the shared link.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex justify-center">
              <Link href="/">
                <Button variant="outline">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Return Home
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </main>
    )
  }

  return (
    <main className="container mx-auto py-4 px-4 min-h-screen flex items-center justify-center">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Hand className="h-5 w-5" />
            <CardTitle>Shared Sign Language Gesture</CardTitle>
          </div>
          <CardDescription>This gesture was shared from the ISL Translator</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center">
            <div className="bg-muted/30 rounded-lg p-6 mb-4">
              <p className="text-4xl font-bold text-primary">{gesture}</p>
            </div>
            <Link href="/">
              <Button variant="outline" size="sm">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Try the Translator
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    </main>
  )
}
