import pydobot # type: ignore
from serial.tools import list_ports # type: ignore
import time

# --- 1. ค้นหาและเชื่อมต่อกับ Dobot ---
print("Searching for Dobot...")
available_ports = list_ports.comports()
if not available_ports:
    print("Error: No serial ports found. Please connect the Dobot.")
    exit()

port = available_ports[0].device
device = pydobot.Dobot(port=port, verbose=False)
print(f"Connected to Dobot on port: {port}")
time.sleep(1)

# --- 2. ลูปสำหรับอ่านและบันทึกค่าพิกัด ---
try:
    print("\n--- Dobot Position Recorder ---")
    print("You can now manually move the arm to any position.")
    
    while True:
        # รอให้ผู้ใช้กด Enter
        input(">> Move the arm to your desired position and press Enter to get coordinates... ")
        
        # อ่านค่าพิกัดปัจจุบันจากแขนกล
        # pose() จะคืนค่า (x, y, z, r, j1, j2, j3, j4)
        current_pose = device.pose()
        
        # แสดงผลค่าพิกัด x, y, z, r (จัดรูปแบบทศนิยม 2 ตำแหน่งเพื่อให้อ่านง่าย)
        x, y, z, r = current_pose[0], current_pose[1], current_pose[2], current_pose[3]
        print(f"Position Captured: (x, y, z) = ({x:.2f}, {y:.2f}, {z:.2f}) | r = {r:.2f}")
        print("-" * 35)

        # ถามว่าจะบันทึกตำแหน่งต่อไปหรือไม่
        again = input("Record another position? (y/n): ").lower()
        if again != 'y':
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # --- 3. ปิดการเชื่อมต่อ ---
    print("Closing connection to Dobot.")
    device.close()