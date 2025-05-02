"use client";
import { useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { LogOut } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

export default function NavBar() {
  const [open, setOpen] = useState(false);
  const router = useRouter();
  const pathname = usePathname();
  const { user, logout } = useAuth();
  const isLoginPage = pathname === "/login";

  const handleLogout = async () => {
    try {
      console.log("Logging out user");
      await logout();
      
      // Use window.location for a hard refresh to ensure proper cookie clearing
      console.log("Redirecting to login page");
      window.location.href = "/login";
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  return (
    <nav className="w-full">
      <div className="flex justify-between items-center px-8 py-2">
        <a href="/" className="text-2xl font-bold">
          ISL
        </a>
        <div className="flex items-center gap-3">
          {!isLoginPage && (
            <Dialog open={open} onOpenChange={setOpen}>
              <DialogTrigger asChild>
                <Button variant="default">
                  Guide
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-md">
                <DialogHeader>
                  <DialogTitle>Guide Information</DialogTitle>
                </DialogHeader>
                <div className="flex justify-center p-6">
                  <img 
                    src="/guide.jpeg" 
                    alt="Guide" 
                    className="rounded-md max-w-full h-auto"
                  />
                </div>
              </DialogContent>
            </Dialog>
          )}
          
          {user && (
            <Button 
              variant="destructive" 
              onClick={handleLogout}
              className="flex items-center gap-2"
            >
              <LogOut className="h-4 w-4" />
              <span>Logout</span>
            </Button>
          )}
        </div>
      </div>
      <div className="border-b border-secondary"></div>
    </nav>
  );
}