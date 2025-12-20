import serial
import serial.tools.list_ports
import time

def find_arduino():
    """
    Scans all serial ports and tries to detect an Arduino Nano.
    Returns the port device string or None if not found.
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description or "1A86" in port.hwid or "2341" in port.hwid:
            print(f"Arduino Nano found on {port.device}")
            return port.device
    return None

def send_motor_command(angle, power):
    """
    Sends a command string like '90,150' (angle=90°, power=150) to the Arduino Nano.
    """
    port = find_arduino()
    if port is None:
        print("No Arduino Nano detected.")
        return False

    try:
        ser = serial.Serial(port, 115200, timeout=2)  # Baud updated to 115200
        time.sleep(2)  # Wait for Arduino reset
        command = f"{angle},{power}\n"
        ser.write(command.encode())
        print(f"Sent: {command.strip()}")

        # Read any response from Arduino
        time.sleep(0.1)
        while ser.in_waiting > 0:
            print("Arduino says:", ser.readline().decode().strip())

        ser.close()
        return True
    except Exception as e:
        print(f"Error communicating with Arduino: {e}")
        return False

if __name__ == "__main__":
    # Example commands
    send_motor_command(0, 15)   # Move forward at power 150
    time.sleep(2)
    send_motor_command(270, 15)  # Move right at power 150
    time.sleep(2)
    send_motor_command(0, 0)     # Stop


