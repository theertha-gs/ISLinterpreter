"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Eye, EyeOff, UserPlus } from "lucide-react"
import { useAuth } from "@/lib/auth-context"
import { FirebaseError } from "firebase/app"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"

export function SignupForm() {
  const { signup } = useAuth()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault()
    console.log("Signup attempt started")
    
    if (!email.trim() || !password.trim() || !confirmPassword.trim()) {
      setError("All fields are required")
      return
    }
    
    if (password !== confirmPassword) {
      setError("Passwords do not match")
      return
    }
    
    if (password.length < 6) {
      setError("Password must be at least 6 characters long")
      return
    }
    
    setIsLoading(true)
    setError(null)
    
    try {
      console.log("Starting Firebase signup...")
      await signup(email, password)
      
      console.log("Signup successful, setting localStorage")
      localStorage.setItem("isLoggedIn", "true")
      
      console.log("Redirecting to welcome page...")
      // Use window.location for hard redirect
      window.location.href = "/welcome"
    } catch (err) {
      console.error("Signup error:", err)
      
      if (err instanceof FirebaseError) {
        switch (err.code) {
          case 'auth/email-already-in-use':
            setError("Email is already in use")
            break
          case 'auth/invalid-email':
            setError("Invalid email format")
            break
          case 'auth/weak-password':
            setError("Password is too weak")
            break
          default:
            setError(`Failed to create account: ${err.code}`)
        }
      } else {
        setError("An unexpected error occurred")
      }
      setIsLoading(false)
    }
  }

  return (
    <Card className="border-2">
      <CardHeader className="space-y-1 text-center">
        <CardTitle className="text-3xl font-bold">Create Account</CardTitle>
        <CardDescription>
          Sign up for a new account to get started
        </CardDescription>
      </CardHeader>
      
      <form onSubmit={handleSignup}>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="signup-email">Email</Label>
            <Input
              id="signup-email"
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="signup-password">Password</Label>
            <div className="relative">
              <Input
                id="signup-password"
                type={showPassword ? "text" : "password"}
                placeholder="Create a password"
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
          
          <div className="space-y-2">
            <Label htmlFor="confirm-password">Confirm Password</Label>
            <Input
              id="confirm-password"
              type={showPassword ? "text" : "password"}
              placeholder="Confirm your password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
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
                <span>Creating account...</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <UserPlus className="h-4 w-4" />
                <span>Sign Up</span>
              </div>
            )}
          </Button>
        </CardFooter>
      </form>
    </Card>
  )
}
