"use client"

import { useEffect, useRef, useState } from "react"
import { Camera, Check, Loader2, Pause, Play, WifiOff, Plus, LogOut, Share2, Copy } from "lucide-react"
import { useRouter } from "next/navigation"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/components/ui/use-toast"

export default function CameraPage() {
  const router = useRouter()
  const { toast } = useToast()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const [isStreaming, setIsStreaming] = useState(false)
  const [prediction, setPrediction] = useState<string>("")
  const [accuracy, setAccuracy] = useState<number>(0)
  const [isLoading, setIsLoading] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected")
  const [handDetected, setHandDetected] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string>("")

  // Check authentication on component mount
  useEffect(() => {
    const isLoggedIn = localStorage.getItem("isLoggedIn") === "true"
    
    if (!isLoggedIn) {
      router.push("/login")
    }
  }, [router])

  // Create a new session
  const createSession = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/create-session")
      const data = await response.json()
      setSessionId(data.session_id)
      return data.session_id
    } catch (error) {
      console.error("Failed to create session:", error)
      return null
    }
  }

  // Connect to WebSocket server
  const connectWebSocket = async () => {
    setConnectionError(null)
    setWsStatus("connecting")

    if (wsRef.current && 
        (wsRef.current.readyState === WebSocket.OPEN || 
         wsRef.current.readyState === WebSocket.CONNECTING)) {
      wsRef.current.close()
    }

    // Create a new session
    const newSessionId = await createSession()
    if (!newSessionId) {
      setConnectionError("Failed to create sharing session.")
      setWsStatus("disconnected")
      return
    }

    try {
      const ws = new WebSocket(`ws://localhost:8000/ws/broadcast/${newSessionId}`)
      
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          ws.close()
          setConnectionError("Connection timeout. Please check if the backend server is running.")
          setWsStatus("disconnected")
        }
      }, 5000)

      ws.onopen = () => {
        clearTimeout(connectionTimeout)
        setWsStatus("connected")
        setConnectionError(null)
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.error) {
            if (data.error === "No hand detected") {
              setHandDetected(false)
              setPrediction("")
              setAccuracy(0)
            }
            return
          }
          
          setHandDetected(!!data.prediction)
          
          if (data.prediction) {
            setPrediction(data.prediction)
            if (data.probabilities && data.prediction in data.probabilities) {
              setAccuracy(Math.round(data.probabilities[data.prediction] * 100))
            } else if (data.confidence) {
              setAccuracy(Math.round(data.confidence * 100))
            }
          } else {
            setPrediction("")
            setAccuracy(0)
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error)
        }
      }

      ws.onerror = () => {
        setConnectionError("Failed to connect to the server. Please check if the backend is running.")
        setWsStatus("disconnected")
        clearTimeout(connectionTimeout)
      }

      ws.onclose = (event) => {
        setWsStatus("disconnected")
        if (!connectionError && event.code !== 1000) {
          setConnectionError(`Connection closed (Code: ${event.code}). ${event.reason || "Please try reconnecting."}`)
        }
        setPrediction("")
        setAccuracy(0)
        setHandDetected(false)
        clearTimeout(connectionTimeout)
      }

      wsRef.current = ws
    } catch (error) {
      setConnectionError("Failed to create WebSocket connection. Please try again.")
      setWsStatus("disconnected")
    }
  }

  // Disconnect WebSocket
  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
      setWsStatus("disconnected")
      setPrediction("")
      setAccuracy(0)
      setHandDetected(false)
      setConnectionError(null)
      setSessionId("")
    }
  }

  // Start or stop the camera stream
  const toggleCamera = async () => {
    if (isStreaming) {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream
        stream.getTracks().forEach((track) => track.stop())
        videoRef.current.srcObject = null
        setIsStreaming(false)
        setPrediction("")
        setAccuracy(0)
        setHandDetected(false)
        disconnectWebSocket()
      }
    } else {
      setIsLoading(true)
      setCameraError(null)

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          setIsStreaming(true)
          await connectWebSocket()
        }
      } catch (err) {
        console.error("Error accessing camera:", err)
        setCameraError("Could not access camera. Please ensure you've granted camera permissions.")
      } finally {
        setIsLoading(false)
      }
    }
  }

  // Handle frame capture and sending
  const captureAndSendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext("2d")

    if (!context) return

    try {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      const base64Image = canvas.toDataURL("image/jpeg", 0.7).split(",")[1]

      wsRef.current.send(
        JSON.stringify({
          frame: base64Image,
          timestamp: Date.now(),
        })
      )
    } catch (error) {
      console.error("Error capturing or sending frame:", error)
    }
  }

  // Set up frame capture interval
  useEffect(() => {
    let frameInterval: NodeJS.Timeout | null = null
    const frameRate = 5

    if (isStreaming && wsStatus === "connected") {
      frameInterval = setInterval(() => {
        captureAndSendFrame()
      }, 1000 / frameRate)
    }

    return () => {
      if (frameInterval) {
        clearInterval(frameInterval)
      }
    }
  }, [isStreaming, wsStatus])

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream
        stream.getTracks().forEach((track) => track.stop())
      }
      disconnectWebSocket()
    }
  }, [])

  return (
    <main className="container mx-auto py-4 px-4 relative min-h-screen">
      <div className="flex flex-col gap-6">
        {/* Camera Section */}
        <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="flex items-center gap-2">
                  <Camera className="h-5 w-5" />
                  Camera Feed
                </CardTitle>
                <div className="flex items-center gap-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => router.push("/welcome")}
                  >
                    Back to Welcome
                  </Button>
                  <Badge
                    variant={
                      wsStatus === "connected" ? "success" : wsStatus === "connecting" ? "outline" : "destructive"
                    }
                  >
                    {wsStatus === "connected"
                      ? "Connected to Model"
                      : wsStatus === "connecting"
                        ? "Connecting..."
                        : "Disconnected"}
                  </Badge>
                </div>
              </div>
              <CardDescription>Position your hands in the frame to translate sign language</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col items-center">
              <div className="relative w-full h-[45vh] bg-muted rounded-lg overflow-hidden mb-4">
                {cameraError ? (
                  <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-muted-foreground">
                    {cameraError}
                  </div>
                ) : !isStreaming ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Camera className="h-16 w-16 text-muted-foreground opacity-20" />
                  </div>
                ) : null}
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-contain ${!isStreaming ? "hidden" : ""}`}
                />
                <canvas ref={canvasRef} className="hidden" />
              </div>

              {connectionError && (
                <div className="w-full p-3 mb-3 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
                  {connectionError}
                </div>
              )}

              <div className="flex flex-col sm:flex-row gap-3 w-full">
                <Button
                  onClick={toggleCamera}
                  disabled={isLoading}
                  variant={isStreaming ? "destructive" : "default"}
                  className="w-full"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Initializing Camera
                    </>
                  ) : isStreaming ? (
                    <>
                      <Pause className="mr-2 h-4 w-4" />
                      Stop Camera
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Start Camera
                    </>
                  )}
                </Button>

                {isStreaming && wsStatus !== "connected" && (
                  <Button onClick={connectWebSocket} variant="outline" className="w-full">
                    <WifiOff className="mr-2 h-4 w-4" />
                    Reconnect to Model
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

        {/* Prediction Section */}
        <Card className="flex flex-col">
            <CardHeader className="pb-0">
              <div className="flex justify-between items-center">
                <CardTitle>Detected Gesture</CardTitle>
              </div>
              <CardDescription>Real-time gesture recognition</CardDescription>
            </CardHeader>
            
            <CardContent className="pt-2">
              {isStreaming ? (
                <div className="text-center">
                  <h3 className="text-sm font-medium mb-2">Detected Sign</h3>
                  <div className="bg-muted/30 rounded-lg p-3 inline-block min-w-[100px] mb-3">
                    {handDetected ? (
                      prediction ? (
                        <p className="text-2xl font-bold text-primary">{prediction}</p>
                      ) : (
                        <p className="text-muted-foreground text-sm">Recognizing...</p>
                      )
                    ) : (
                      <p className="text-muted-foreground text-sm">No hand</p>
                    )}
                  </div>
                  {wsStatus === "connected" && (
                    <div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="gap-2"
                        onClick={async () => {
                          const shareUrl = `${window.location.origin}/share/${sessionId}`
                          await navigator.clipboard.writeText(shareUrl)
                          toast({
                            title: "Share link copied",
                            description: "Share this link with others to let them watch your sign language translations in real-time",
                          })
                        }}
                      >
                        <Share2 className="h-4 w-4" />
                        Copy Share Link
                      </Button>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-[120px] text-center text-muted-foreground">
                  <Camera className="h-10 w-10 mb-3 opacity-20" />
                  <p className="text-sm">Start the camera to begin translating sign language</p>
                </div>
              )}
            </CardContent>
          </Card>
      </div>
    </main>
  )
}
