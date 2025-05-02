"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Eye, EyeOff, LogIn } from "lucide-react"
import { useAuth } from "@/lib/auth-context"
import { FirebaseError } from "firebase/app"
import { SignupForm } from "./signup"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function LoginPage() {
  const router = useRouter()
  const { login, user } = useAuth()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    // Debug logs
    console.log("Login page mounted")
    console.log("Current user state:", user ? "Logged in" : "Not logged in")

    if (user) {
      console.log("User is authenticated, setting localStorage")
      localStorage.setItem("isLoggedIn", "true")
      console.log("Redirecting to welcome page...")
      // Use window.location for hard redirect
      window.location.href = "/welcome"
    }
  }, [user])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    console.log("Login attempt started")
    
    if (!email.trim() || !password.trim()) {
      setError("Please enter both email and password")
      return
    }
    
    setIsLoading(true)
    setError(null)
    
    try {
      console.log("Attempting Firebase login...")
      await login(email, password)
      
      console.log("Login successful, setting localStorage")
      localStorage.setItem("isLoggedIn", "true")
      
      console.log("Redirecting to welcome page...")
      // Use window.location for hard redirect
      window.location.href = "/welcome"
    } catch (err) {
      console.error("Login error:", err)
      
      if (err instanceof FirebaseError) {
        switch (err.code) {
          case 'auth/invalid-email':
            setError("Invalid email format")
            break
          case 'auth/user-not-found':
          case 'auth/wrong-password':
            setError("Invalid email or password")
            break
          case 'auth/too-many-requests':
            setError("Too many failed login attempts. Please try again later")
            break
          default:
            setError(`Failed to login: ${err.code}`)
        }
      } else {
        setError("An unexpected error occurred")
      }
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-md">
        <Tabs defaultValue="login" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-4">
            <TabsTrigger value="login">Login</TabsTrigger>
            <TabsTrigger value="signup">Sign Up</TabsTrigger>
          </TabsList>
          
          <TabsContent value="login">
            <Card className="border-2">
              <CardHeader className="space-y-1 text-center">
                <CardTitle className="text-3xl font-bold">ISL Translator</CardTitle>
                <CardDescription>
                  Enter your credentials to sign in
                </CardDescription>
              </CardHeader>
              
              <form onSubmit={handleLogin}>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input
                      id="email"
                      type="email"
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="password">Password</Label>
                    <div className="relative">
                      <Input
                        id="password"
                        type={showPassword ? "text" : "password"}
                        placeholder="Enter your password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="absolute right-0 top-0 h-full px-3"
                        onClick={() => setShowPassword(!showPassword)}
                      >
                        {showPassword ? (
                          <EyeOff className="h-4 w-4" />
                        ) : (
                          <Eye className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                  
                  {error && (
                    <Alert variant="destructive">
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}
                </CardContent>
                
                <CardFooter>
                  <Button 
                    type="submit" 
                    className="w-full" 
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <div className="flex items-center gap-2">
                        <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                        <span>Logging in...</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <LogIn className="h-4 w-4" />
                        <span>Sign In</span>
                      </div>
                    )}
                  </Button>
                </CardFooter>
              </form>
            </Card>
          </TabsContent>
          
          <TabsContent value="signup">
            <SignupForm />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}
