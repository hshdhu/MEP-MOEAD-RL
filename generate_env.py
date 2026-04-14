"""
Generate Environment - Create an environment and save it to file.
"""
from utils.config_loader import load_config
from pathlib import Path

def main():
    print("=" * 60)
    print("Environment Generator")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Generate environment
    print("\nGenerating environment...")
    env = config.get_environment()
    
    print(f"\nEnvironment created:")
    print(f"   Size: {env.width}x{env.height}")
    print(f"   Sensors: {len(env.sensors)}")
    print(f"   Obstacles: {len(env.obstacles)}")
    
    # Determine save folder based on sensor count
    num_sensors = len(env.sensors)
    if num_sensors <= 50:
        folder_name = "50 sensors"
    elif num_sensors <= 100:
        folder_name = "100 sensors"
    elif num_sensors <= 150:
        folder_name = "150 sensors"
    else:
        folder_name = "200 sensors"
    
    # Create data folder if it doesn't exist
    save_folder = Path("data") / folder_name
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"env_{timestamp}.json"
    
    # Save environment
    env.save("data", f"{folder_name}/{filename}", path=None)
    
    print(f"\nEnvironment saved to: data/{folder_name}/{filename}")
    print(f"\nTo use this environment, run:")
    print(f"   python run_ppo.py \"data/{folder_name}/{filename}\"")

if __name__ == "__main__":
    main()
