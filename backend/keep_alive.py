import httpx
import asyncio
from datetime import datetime

async def keep_alive():
    """Ping health endpoint every 14 minutes"""
    url = "https://your-app-name.onrender.com/health"
    
    while True:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10)
                print(f"[{datetime.now()}] Keep-alive ping: {response.status_code}")
        except Exception as e:
            print(f"[{datetime.now()}] Keep-alive failed: {e}")
        
        await asyncio.sleep(840)  # 14 minutes

if __name__ == "__main__":
    asyncio.run(keep_alive())