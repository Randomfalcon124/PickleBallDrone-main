import asyncio
import subprocess
from mavsdk import System

async def get_wsl_ip():
    try:
        # Run a WSL command to get the IP address of WSL's eth0
        result = subprocess.run(["wsl", "hostname", "-I"], capture_output=True, text=True)
        ip = result.stdout.strip().split()[0]
        print(f"Detected WSL IP: {ip}")
        return ip
    except Exception as e:
        print(f"Error getting WSL IP: {e}")
        return None

async def run():
    ip = await get_wsl_ip()
    if not ip:
        print("Could not detect WSL IP. Make sure WSL is installed and PX4 is running.")
        return

    drone = System()
    await drone.connect(system_address=f"udp://{ip}:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break

    # Example: print health info
    async for health in drone.telemetry.health():
        print(f"Drone health: {health}")
        break

asyncio.run(run())
