import math
import tkinter as tk
from tkinter import messagebox


root = tk.Tk()
root.title("Лабораторна робота 1")
root.geometry("1100x700")
root.resizable(False, False)


running = False
after_id = None
current_time = 0.0

x0 = 0.0
y0 = 0.0
speed = 1.0
angle = 30.0
vx = 0.0
vy = 0.0
time_limit = 10.0
dt = 0.2

scale = 30.0
max_value = 10.0

fields = {}


def create_input(parent, text, key, value, row):
    label = tk.Label(parent, text=text, font=("Arial", 11), anchor="w")
    label.grid(row=row, column=0, sticky="w", pady=4)

    entry = tk.Entry(parent, width=12, font=("Arial", 11))
    entry.grid(row=row, column=1, padx=8, pady=4)
    entry.insert(0, str(value))

    fields[key] = entry


def update_velocity():
    global vx, vy

    angle_in_radians = math.radians(angle)
    vx = speed * math.cos(angle_in_radians)
    vy = speed * math.sin(angle_in_radians)


def update_scale():
    global scale, max_value

    x_end = x0 + vx * time_limit
    y_end = y0 + vy * time_limit

    max_value = max(abs(x0), abs(y0), abs(x_end), abs(y_end), 1.0)
    scale = 220 / max_value


def read_values():
    global x0, y0, speed, angle, time_limit

    try:
        x0 = float(fields["x0"].get())
        y0 = float(fields["y0"].get())
        speed = float(fields["speed"].get())
        angle = float(fields["angle"].get())
        time_limit = float(fields["time"].get())
    except ValueError:
        messagebox.showerror("Input error", "Enter only numbers.")
        return False

    if time_limit < 0:
        messagebox.showerror("Input error", "Time must be 0 or bigger.")
        return False

    update_velocity()
    update_scale()
    return True


def to_canvas(x, y):
    center_x = 350
    center_y = 280

    canvas_x = center_x + x * scale
    canvas_y = center_y - y * scale

    return canvas_x, canvas_y


def draw_plane():
    canvas.delete("plane")

    left_x, center_y = to_canvas(-max_value, 0)
    right_x, _ = to_canvas(max_value, 0)
    center_x, top_y = to_canvas(0, max_value)
    _, bottom_y = to_canvas(0, -max_value)

    canvas.create_line(left_x, center_y, right_x, center_y, width=2, fill="firebrick", arrow=tk.LAST, tags="plane")
    canvas.create_line(center_x, bottom_y, center_x, top_y, width=2, fill="seagreen", arrow=tk.LAST, tags="plane")

    canvas.create_text(right_x - 10, center_y - 15, text="X", fill="firebrick", font=("Arial", 12, "bold"), tags="plane")
    canvas.create_text(center_x + 15, top_y + 10, text="Y", fill="seagreen", font=("Arial", 12, "bold"), tags="plane")
    canvas.create_text(center_x - 10, center_y + 15, text="0", fill="black", font=("Arial", 10), tags="plane")

    x_end = x0 + vx * time_limit
    y_end = y0 + vy * time_limit
    start_x, start_y = to_canvas(x0, y0)
    finish_x, finish_y = to_canvas(x_end, y_end)

    canvas.create_line(start_x, start_y, finish_x, finish_y, width=2, fill="gray50", dash=(6, 4), tags="plane")
    canvas.create_text(start_x, start_y - 15, text="старт", font=("Arial", 10), tags="plane")
    canvas.create_text(finish_x, finish_y - 15, text="фініш", font=("Arial", 10), tags="plane")


def show_ball(x, y):
    canvas_x, canvas_y = to_canvas(x, y)

    canvas.delete("ball")
    canvas.create_oval(
        canvas_x - 12,
        canvas_y - 12,
        canvas_x + 12,
        canvas_y + 12,
        fill="orange",
        outline="black",
        width=2,
        tags="ball",
    )


def stop_animation():
    global running, after_id

    running = False

    if after_id is not None:
        root.after_cancel(after_id)
        after_id = None


def start_animation():
    global running, current_time

    stop_animation()

    if not read_values():
        return

    running = True
    current_time = 0.0

    draw_plane()
    animate()


def animate():
    global current_time, running, after_id

    if not running:
        return

    if current_time > time_limit + 0.000001:
        running = False
        after_id = None
        return

    x = x0 + vx * current_time
    y = y0 + vy * current_time

    show_ball(x, y)

    current_time = current_time + dt
    after_id = root.after(100, animate)


def reset_all():
    global current_time

    stop_animation()

    defaults = {
        "x0": "0",
        "y0": "0",
        "speed": "1",
        "angle": "30",
        "time": "10",
    }

    for key in defaults:
        fields[key].delete(0, tk.END)
        fields[key].insert(0, defaults[key])

    current_time = 0.0
    read_values()
    draw_plane()
    show_ball(x0, y0)


left_frame = tk.Frame(root, padx=15, pady=15)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

right_frame = tk.Frame(root, padx=15, pady=15)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

title_label = tk.Label(left_frame, text="Модель руху", font=("Arial", 16, "bold"))
title_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

create_input(left_frame, "x0", "x0", 0, 1)
create_input(left_frame, "y0", "y0", 0, 2)
create_input(left_frame, "швидкість v", "speed", 1, 3)
create_input(left_frame, "кут a", "angle", 30, 4)
create_input(left_frame, "час t", "time", 10, 5)

start_button = tk.Button(left_frame, text="Почати анімацію", width=20, command=start_animation)
start_button.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 3))

stop_button = tk.Button(left_frame, text="Зупинити анімацію", width=20, command=stop_animation)
stop_button.grid(row=7, column=0, columnspan=2, sticky="we", pady=3)

reset_button = tk.Button(left_frame, text="Скинути", width=20, command=reset_all)
reset_button.grid(row=8, column=0, columnspan=2, sticky="we", pady=3)

canvas_title = tk.Label(right_frame, text="Рух кульки на площині XY", font=("Arial", 14, "bold"))
canvas_title.pack(anchor="w")

canvas = tk.Canvas(
    right_frame,
    width=700,
    height=560,
    bg="white",
    highlightthickness=1,
    highlightbackground="gray60",
)
canvas.pack(pady=(10, 0))

note_label = tk.Label(
    right_frame,
    text="Кут задається у градусах.",
    font=("Arial", 10),
)
note_label.pack(anchor="w", pady=(8, 0))

update_velocity()
update_scale()
draw_plane()
show_ball(x0, y0)

root.mainloop()
