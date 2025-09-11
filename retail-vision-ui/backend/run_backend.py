#!/usr/bin/env python3
"""
Retail Vision Backend Runner
Simple script to start the FastAPI backend server
"""

import uvicorn
import sys


def main():
    print("🚀 Starting Retail Vision Backend...")
    print("📍 Backend will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
