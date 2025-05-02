"use client"

import { useEffect, useRef, useState } from "react"
import { Camera, ArrowLeft, Save, Loader2, Play, Square } from "lucide-react"
import { useRouter } from "next/navigation"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"

export default function CustomGesturePage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const router = useRouter()

  const [isStreaming, setIsStreaming] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [gestureName, setGestureName] = useState("")
  const [recordedFrames, setRecordedFrames] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [recordingTime, setRecordingTime] = useState(0)
  
  // Start or stop the camera stream
  const toggleCamera = async () => {
    if (isStreaming) {
      // Stop the stream
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
        setIsStreaming(false);
      }
    } else {
      // Start the stream
      setIsLoading(true);
      setCameraError(null);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsStreaming(true);
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
        setCameraError("Could not access camera. Please ensure you've granted camera permissions.");
      } finally {
        setIsLoading(false);
      }
    }
  }

  // Capture a frame from the video
  const captureFrame = (): string | null => {
    if (!videoRef.current || !canvasRef.current) {
      return null;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    if (!context) return null;

    try {
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw current video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert canvas to base64 image (JPEG format with quality 0.7 for reduced size)
      const base64Image = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
      return base64Image;
    } catch (error) {
      console.error("Error capturing frame:", error);
      return null;
    }
  }

  // Start recording frames
  const startRecording = () => {
    if (!isStreaming) return;
    
    setRecordedFrames([]);
    setIsRecording(true);
    setRecordingTime(0);
    setError(null);
    setSuccess(null);
  }

  // Stop recording frames
  const stopRecording = () => {
    setIsRecording(false);
  }

  // Save the recorded gesture
  const saveGesture = async () => {
    if (recordedFrames.length === 0) {
      setError("No frames recorded. Please record your gesture first.");
      return;
    }

    if (!gestureName.trim()) {
      setError("Please enter a name for your gesture.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/custom-gestures", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: gestureName.trim(),
          frames: recordedFrames,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "Failed to save gesture");
      }

      setSuccess(`Gesture "${gestureName}" saved successfully!`);
      setGestureName("");
      setRecordedFrames([]);
    } catch (err) {
      console.error("Error saving gesture:", err);
      setError(err instanceof Error ? err.message : "Failed to save gesture");
    } finally {
      setIsLoading(false);
    }
  }

  // Go back to the main page
  const goBack = () => {
    router.push("/");
  }

  // Capture frames while recording is active
  useEffect(() => {
    if (!isRecording) return;

    const intervalId = setInterval(() => {
      const frame = captureFrame();
      if (frame) {
        setRecordedFrames((prev) => [...prev, frame]);
        setRecordingTime((prev) => prev + 100);
      }
    }, 100); // Capture 10 frames per second

    // Stop recording after 3 seconds
    const timeoutId = setTimeout(() => {
      stopRecording();
    }, 3000);

    return () => {
      clearInterval(intervalId);
      clearTimeout(timeoutId);
    };
  }, [isRecording]);

  // Clean up camera stream on unmount
  useEffect(() => {
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <main className="container mx-auto py-6 px-4">
      <div className="mb-4 flex items-center gap-2">
        <Button variant="outline" size="icon" onClick={goBack}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h1 className="text-2xl font-bold">Custom Gesture</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Camera Section */}
        <div className="lg:col-span-2">
          <Card className="h-full relative z-10">
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="flex items-center gap-2">
                  <Camera className="h-5 w-5" />
                  Record Custom Gesture
                </CardTitle>
                {isRecording && (
                  <Badge variant="destructive" className="animate-pulse">
                    Recording ({(recordingTime / 1000).toFixed(1)}s)
                  </Badge>
                )}
              </div>
              <CardDescription>
                Position your hands in the frame and record your custom gesture
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col items-center">
              <div className="relative w-full aspect-video bg-muted rounded-lg overflow-hidden mb-4">
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
                  className={`w-full h-full object-cover ${!isStreaming ? "hidden" : ""}`}
                />
                {/* Hidden canvas used for frame capture */}
                <canvas ref={canvasRef} className="hidden" />
              </div>

              <div className="flex flex-col space-y-4 w-full">
                <div className="flex flex-col sm:flex-row gap-3">
                  <Button
                    onClick={toggleCamera}
                    disabled={isLoading || isRecording}
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
                        <Square className="mr-2 h-4 w-4" />
                        Stop Camera
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Start Camera
                      </>
                    )}
                  </Button>

                  {isStreaming && !isRecording && (
                    <Button
                      onClick={startRecording}
                      variant="default"
                      className="w-full sm:w-auto bg-red-500 hover:bg-red-600"
                    >
                      <div className="h-2 w-2 rounded-full bg-white mr-2" />
                      Record
                    </Button>
                  )}

                  {isStreaming && isRecording && (
                    <Button
                      onClick={stopRecording}
                      variant="default"
                      className="w-full"
                    >
                      <Square className="mr-2 h-4 w-4" />
                      Stop Recording
                    </Button>
                  )}
                </div>

                {recordedFrames.length > 0 && (
                  <div className="bg-muted p-4 rounded-md">
                    <p className="text-sm text-muted-foreground mb-2">
                      {recordedFrames.length} frames recorded
                    </p>
                    <div className="flex flex-col space-y-3">
                      <div>
                        <Label htmlFor="gesture-name">Gesture Name</Label>
                        <Input
                          id="gesture-name"
                          placeholder="Enter gesture name"
                          value={gestureName}
                          onChange={(e) => setGestureName(e.target.value)}
                        />
                      </div>
                      <Button
                        onClick={saveGesture}
                        disabled={isLoading || !gestureName.trim()}
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Saving...
                          </>
                        ) : (
                          <>
                            <Save className="mr-2 h-4 w-4" />
                            Save Custom Gesture
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                )}

                {error && (
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {success && (
                  <Alert className="bg-green-50 border-green-200 text-green-800">
                    <AlertDescription>{success}</AlertDescription>
                  </Alert>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Info Section */}
        <div>
          <Card className="h-full">
            <CardHeader>
              <CardTitle>Instructions</CardTitle>
              <CardDescription>How to create a custom gesture</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium">1. Position your hand</h3>
                <p className="text-sm text-muted-foreground">
                  Make sure your hand is clearly visible in the camera frame.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium">2. Record your gesture</h3>
                <p className="text-sm text-muted-foreground">
                  Click the "Record Gesture" button and hold your gesture for 3 seconds.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium">3. Name your gesture</h3>
                <p className="text-sm text-muted-foreground">
                  Give your gesture a descriptive name so you can easily identify it later.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium">4. Save your gesture</h3>
                <p className="text-sm text-muted-foreground">
                  Click the "Save Custom Gesture" button to store your gesture in the database.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  )
} 