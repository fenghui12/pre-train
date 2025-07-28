import tkinter as tk
import time

print(f"[{time.time()}] Tkinter application start.")

root = tk.Tk()
root.title("Test Tkinter Speed")
root.geometry("300x200")

label = tk.Label(root, text="Hello, Tkinter!")
label.pack(pady=50)

print(f"[{time.time()}] Tkinter window created. Starting mainloop.")
root.mainloop()
print(f"[{time.time()}] Tkinter application exit.")