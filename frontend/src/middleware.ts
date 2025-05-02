import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// Track recent redirects to prevent loops
const recentRedirects = new Map<string, { timestamp: number, count: number }>();
const REDIRECT_EXPIRY = 5000; // 5 seconds
const MAX_REDIRECTS = 3;

export function middleware(request: NextRequest) {
  // Get the path of the request
  const path = request.nextUrl.pathname
  
  // Define public paths that don't require authentication
  const isPublicPath = path === '/login'
  
  // Get the authentication status from cookies
  const hasAuthToken = request.cookies.has('auth-token')
  const authTokenValue = request.cookies.get('auth-token')?.value
  
  // Create a unique key for this path
  const redirectKey = path;
  
  // Check if we've had too many redirects recently
  const now = Date.now();
  const recentRedirect = recentRedirects.get(redirectKey);
  
  // Only apply redirect limiting on the same path - not across paths
  if (recentRedirect && now - recentRedirect.timestamp < REDIRECT_EXPIRY) {
    if (recentRedirect.count >= MAX_REDIRECTS) {
      console.log(`[Middleware] Too many redirects for ${path}, allowing request`);
      // Just let the request through to break the redirect loop
      return NextResponse.next();
    }
    recentRedirect.count++;
  } else {
    // Reset or initialize the redirect count
    recentRedirects.set(redirectKey, { timestamp: now, count: 1 });
  }
  
  // Clean up old entries
  for (const [key, value] of recentRedirects.entries()) {
    if (now - value.timestamp > REDIRECT_EXPIRY) {
      recentRedirects.delete(key);
    }
  }
  
  console.log(`[Middleware] Path: ${path}, Public path: ${isPublicPath}, Has auth token: ${hasAuthToken}`);
  
  // Redirect to login if not authenticated and not on a public path
  if (!hasAuthToken && !isPublicPath) {
    console.log('[Middleware] Not authenticated, redirecting to login');
    // Create the URL for redirection
    const loginUrl = new URL('/login', request.url)
    return NextResponse.redirect(loginUrl)
  }
  
  // Redirect to dashboard if already authenticated and trying to access login
  if (hasAuthToken && isPublicPath) {
    console.log('[Middleware] Already authenticated, redirecting to home');
    // Create the URL for redirection
    const homeUrl = new URL('/', request.url)
    return NextResponse.redirect(homeUrl)
  }
  
  // Allow the request to proceed as normal
  console.log('[Middleware] Request allowed to proceed');
  return NextResponse.next()
}

// Configure the middleware to run on specific paths
export const config = {
  matcher: [
    // Match all routes except for api routes, static files, and other special Next.js paths
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
} 