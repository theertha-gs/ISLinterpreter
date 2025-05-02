import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { v4 as uuidv4 } from "uuid";

// Directory to store custom gestures
const CUSTOM_GESTURES_DIR = path.join(process.cwd(), "public", "custom-gestures");

// Ensure the directory exists
if (!fs.existsSync(CUSTOM_GESTURES_DIR)) {
  fs.mkdirSync(CUSTOM_GESTURES_DIR, { recursive: true });
}

// Database file path
const DB_FILE = path.join(CUSTOM_GESTURES_DIR, "gestures.json");

// Initialize database if it doesn't exist
if (!fs.existsSync(DB_FILE)) {
  fs.writeFileSync(DB_FILE, JSON.stringify({ gestures: [] }));
}

// Type definitions
interface Gesture {
  id: string;
  name: string;
  createdAt: string;
  frameCount: number;
  firstFramePath: string;
}

interface GestureData {
  name: string;
  frames: string[]; // Base64 encoded frames
}

// GET handler to retrieve all custom gestures
export async function GET() {
  try {
    // Read the database
    const data = JSON.parse(fs.readFileSync(DB_FILE, "utf-8"));
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error retrieving gestures:", error);
    return NextResponse.json(
      { error: "Failed to retrieve gestures" },
      { status: 500 }
    );
  }
}

// POST handler to save a new custom gesture
export async function POST(request: NextRequest) {
  try {
    const data: GestureData = await request.json();
    const { name, frames } = data;

    // Validate input
    if (!name || !frames || frames.length === 0) {
      return NextResponse.json(
        { error: "Name and frames are required" },
        { status: 400 }
      );
    }

    // Generate a unique ID for the gesture
    const gestureId = uuidv4();
    const gestureDir = path.join(CUSTOM_GESTURES_DIR, gestureId);
    
    // Create a directory for this gesture
    fs.mkdirSync(gestureDir, { recursive: true });

    // Save frames as images
    for (let i = 0; i < frames.length; i++) {
      const base64Data = frames[i];
      const buffer = Buffer.from(base64Data, "base64");
      const framePath = path.join(gestureDir, `frame_${i}.jpg`);
      fs.writeFileSync(framePath, buffer);
    }

    // Create gesture metadata
    const gesture: Gesture = {
      id: gestureId,
      name,
      createdAt: new Date().toISOString(),
      frameCount: frames.length,
      firstFramePath: `/custom-gestures/${gestureId}/frame_0.jpg`,
    };

    // Update the database
    const dbData = JSON.parse(fs.readFileSync(DB_FILE, "utf-8"));
    dbData.gestures.push(gesture);
    fs.writeFileSync(DB_FILE, JSON.stringify(dbData, null, 2));

    return NextResponse.json({ 
      message: "Gesture saved successfully", 
      gesture 
    });
  } catch (error) {
    console.error("Error saving gesture:", error);
    return NextResponse.json(
      { error: "Failed to save gesture" },
      { status: 500 }
    );
  }
}

// DELETE handler to remove a custom gesture
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get("id");

    if (!id) {
      return NextResponse.json(
        { error: "Gesture ID is required" },
        { status: 400 }
      );
    }

    // Read the database
    const dbData = JSON.parse(fs.readFileSync(DB_FILE, "utf-8"));
    
    // Find the gesture
    const gestureIndex = dbData.gestures.findIndex((g: Gesture) => g.id === id);
    
    if (gestureIndex === -1) {
      return NextResponse.json(
        { error: "Gesture not found" },
        { status: 404 }
      );
    }

    // Remove the gesture from the database
    dbData.gestures.splice(gestureIndex, 1);
    fs.writeFileSync(DB_FILE, JSON.stringify(dbData, null, 2));

    // Remove the gesture directory
    const gestureDir = path.join(CUSTOM_GESTURES_DIR, id);
    if (fs.existsSync(gestureDir)) {
      fs.rmSync(gestureDir, { recursive: true, force: true });
    }

    return NextResponse.json({ message: "Gesture deleted successfully" });
  } catch (error) {
    console.error("Error deleting gesture:", error);
    return NextResponse.json(
      { error: "Failed to delete gesture" },
      { status: 500 }
    );
  }
} 