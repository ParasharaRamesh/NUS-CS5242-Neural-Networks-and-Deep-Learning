import pyautogui
import time
import random
import threading
import keyboard

# Disable fail-safe
pyautogui.FAILSAFE = False

# Global variable to signal the script to stop
stop_script = False

def move_mouse_randomly():
    while not stop_script:
        try:
            # Get the screen dimensions
            screen_width, screen_height = pyautogui.size()

            # Get the current mouse position
            current_x, current_y = pyautogui.position()

            # Generate random offsets for a slightly larger step
            x_offset = random.uniform(-50, 50)
            y_offset = random.uniform(-50, 50)

            # Calculate the new position
            new_x = max(0, min(screen_width - 1, current_x + x_offset))
            new_y = max(0, min(screen_height - 1, current_y + y_offset))

            # Check if the new coordinates are close to the corners; adjust if needed
            if new_x < 10:
                new_x = 10
            elif new_x > screen_width - 10:
                new_x = screen_width - 10

            if new_y < 10:
                new_y = 10
            elif new_y > screen_height - 10:
                new_y = screen_height - 10

            # Move the mouse to the new coordinates
            pyautogui.moveTo(new_x, new_y, duration=0.1)

            # Sleep for a random duration (e.g., 0.1 to 0.35 seconds)
            time.sleep(random.uniform(0.1, 0.35))

        except pyautogui.FailSafeException:
            # Handle the fail-safe as needed, or just ignore it
            print("FailSafeException: Ignored.")
            pass

def stop_script_func():
    global stop_script
    stop_script = True

if __name__ == "__main__":
    # Start the mouse movement in a separate thread
    mouse_thread = threading.Thread(target=move_mouse_randomly)
    mouse_thread.start()

    # Check for a key press to stop the script
    print("Press any key to stop the script.")
    keyboard.wait("esc")  # Waits for any key press, you can customize this

    # Signal the script to stop
    stop_script_func()

    # Wait for the mouse thread to finish
    mouse_thread.join()

    print("Script stopped.")
