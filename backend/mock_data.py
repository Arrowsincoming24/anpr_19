import asyncio
import uuid
import random
from datetime import datetime, timezone

from .database import save_detection, get_stats

async def inject_mock_data_if_empty():
    try:
        stats = await get_stats()
        if stats["total"] > 0:
            return  # db already has data
            
        print("Injecting mock data for testing...")
        states = ["MH", "KA", "DL", "UP", "HR"]
        makes = ["Maruti Suzuki", "Hyundai", "Tata", "Mahindra", "Honda"]
        
        # We need a dummy crop_image (a simple 1x1 black pixel or gray square)
        # using a small valid base64 transparent GIF instead of large SVG to keep db small
        dummy_b64 = "data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
        
        for i in range(25):
            uid = str(uuid.uuid4())[:8]
            state = random.choice(states)
            plate = f"{state}{random.randint(10,99)}AB{random.randint(1000,9999)}"
            make = random.choice(makes)
            
            await save_detection(
                uuid=uid,
                plate_text=plate,
                confidence=random.uniform(0.75, 0.99),
                source="upload:mock_feed.jpg",
                total_ms=random.uniform(40.0, 250.0),
                crop_image=dummy_b64,
                state=state,
                city="Pune" if state == "MH" else "Bangalore" if state == "KA" else "Delhi",
                series_age="Unknown",
                owner=f"Demo Owner {i}",
                make=make,
                speed=random.uniform(30, 80) if i % 4 == 0 else 0.0,
                environment="Daylight",
                camera_id=f"CAM-0{random.randint(1,4)}",
            )
        print("Mock data injected successfully.")
    except Exception as e:
        print(f"Failed to inject mock data: {e}")
