"use client"

import { useEffect, useState, useRef } from "react"
import { useParams } from "next/navigation"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Hand, ArrowLeft } from "lucide-react"
import Link from "next/link"
import { Badge } from "@/components/ui/badge"

export default function ShareSessionPage() {
  const params = useParams()
  const sessionId = params.sessionId as string
  const wsRef = useRef<WebSocket | null>(null)
  const [prediction, setPrediction] = useState<string>("")
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected")
  const [error, setError] = useState<string | null>(null)
  
  useEffect(() => {
    const connectWebSocket = () => {
      setWsStatus("connecting")
      
      if (wsRef.current && 
          (wsRef.current.readyState === WebSocket.OPEN || 
           wsRef.current.readyState === WebSocket.CONNECTING)) {
        wsRef.current.close()
      }

      try {
        const ws = new WebSocket(`ws://localhost:8000/ws/view/${sessionId}`)
        
        const connectionTimeout = setTimeout(() => {
          if (ws.readyState !== WebSocket.OPEN) {
            ws.close()
            setError("Connection timeout. The sharing session might have ended.")
            setWsStatus("disconnected")
          }
        }, 5000)

        ws.onopen = () => {
          clearTimeout(connectionTimeout)
          setWsStatus("connected")
          setError(null)
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            if (data.error) {
              setError(data.error)
              return
            }
            
            if (data.prediction && data.prediction !== "No hand detected") {
              setPrediction(data.prediction)
            } else {
              setPrediction("")
            }
          } catch (error) {
            console.error("Error parsing WebSocket message:", error)
          }
        }

        ws.onerror = () => {
          setError("Failed to connect to the sharing session.")
          setWsStatus("disconnected")
          clearTimeout(connectionTimeout)
        }

        ws.onclose = (event) => {
          setWsStatus("disconnected")
          if (!error && event.code !== 1000) {
            setError("The sharing session has ended.")
          }
          setPrediction("")
        }

        wsRef.current = ws
      } catch (error) {
        setError("Failed to connect to the sharing session.")
        setWsStatus("disconnected")
      }
    }

    connectWebSocket()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [sessionId])

  return (
    <main className="container mx-auto py-4 px-4 min-h-screen flex items-center justify-center">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex justify-between items-center mb-2">
            <div className="flex items-center gap-2">
              <Hand className="h-5 w-5" />
              <CardTitle>Live Sign Language</CardTitle>
            </div>
            <Badge
              variant={
                wsStatus === "connected" ? "success" : wsStatus === "connecting" ? "outline" : "destructive"
              }
            >
              {wsStatus === "connected"
                ? "Connected"
                : wsStatus === "connecting"
                  ? "Connecting..."
                  : "Disconnected"}
            </Badge>
          </div>
          <CardDescription>Watching real-time sign language translation</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center">
            {error ? (
              <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-4">
                {error}
              </div>
            ) : (
              <>
                <div className="bg-muted/30 rounded-lg p-6 mb-4 min-h-[100px] flex items-center justify-center">
                  {prediction ? (
                    <p className="text-4xl font-bold text-primary">{prediction}</p>
                  ) : (
                    <p className="text-muted-foreground">Waiting for gestures...</p>
                  )}
                </div>
              </>
            )}
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
