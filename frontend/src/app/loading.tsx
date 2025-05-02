"use client"

export default function LoadingPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-background to-muted/20">
      <div className="relative">
        <div className="absolute -inset-2 blur-lg bg-primary/20 rounded-full animate-pulse"></div>
        <div className="relative animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
      <p className="mt-4 text-muted-foreground animate-pulse">
        Loading ISL Translator...
      </p>
    </div>
  )
}
