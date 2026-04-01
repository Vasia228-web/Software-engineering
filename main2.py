import math
import tkinter as tk
from tkinter import messagebox


GRAVITY = 9.81
CANVAS_WIDTH = 720
CANVAS_HEIGHT = 420
LEFT_MARGIN = 50
RIGHT_MARGIN = 30
TOP_MARGIN = 30
BOTTOM_MARGIN = 45


class ProjectileApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Моделювання руху тіла")
        self.root.geometry("1120x560")

        self.points = []
        self.current_index = 0
        self.after_id = None
        self.ball_id = None
        self.scale = 1
        self.ground_y = CANVAS_HEIGHT - BOTTOM_MARGIN

        self.range_var = tk.StringVar(value="-")
        self.height_var = tk.StringVar(value="-")
        self.flight_time_var = tk.StringVar(value="-")
        self.point_x_var = tk.StringVar(value="-")
        self.point_y_var = tk.StringVar(value="-")
        self.point_vx_var = tk.StringVar(value="-")
        self.point_vy_var = tk.StringVar(value="-")
        self.point_v_var = tk.StringVar(value="-")
        self.current_state_var = tk.StringVar(
            value="Анімація ще не запущена."
        )

        self.create_widgets()
        self.draw_empty_scene()

    def create_widgets(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=(0, 10))

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)

        input_frame = tk.LabelFrame(left_frame, text="Вхідні дані", padx=10, pady=10)
        input_frame.pack(fill="x", pady=(0, 10))

        tk.Label(input_frame, text="Початкова швидкість v0 (м/с):").grid(
            row=0, column=0, sticky="w"
        )
        self.speed_entry = tk.Entry(input_frame, width=18)
        self.speed_entry.grid(row=0, column=1, pady=4, padx=(8, 0))
        self.speed_entry.insert(0, "20")

        tk.Label(input_frame, text="Кут кидання a (градуси):").grid(
            row=1, column=0, sticky="w"
        )
        self.angle_entry = tk.Entry(input_frame, width=18)
        self.angle_entry.grid(row=1, column=1, pady=4, padx=(8, 0))
        self.angle_entry.insert(0, "45")

        tk.Label(input_frame, text="Крок часу dt (с):").grid(
            row=2, column=0, sticky="w"
        )
        self.step_entry = tk.Entry(input_frame, width=18)
        self.step_entry.grid(row=2, column=1, pady=4, padx=(8, 0))
        self.step_entry.insert(0, "0.1")

        tk.Label(input_frame, text="Час для координат t (с):").grid(
            row=3, column=0, sticky="w"
        )
        self.time_entry = tk.Entry(input_frame, width=18)
        self.time_entry.grid(row=3, column=1, pady=4, padx=(8, 0))
        self.time_entry.insert(0, "1")

        button_frame = tk.Frame(left_frame)
        button_frame.pack(fill="x", pady=(0, 10))

        tk.Button(
            button_frame,
            text="Запустити",
            width=16,
            command=self.start_simulation,
        ).pack(pady=4)

        tk.Button(
            button_frame,
            text="Зупинити",
            width=16,
            command=self.stop_animation,
        ).pack(pady=4)

        tk.Button(
            button_frame,
            text="Очистити",
            width=16,
            command=self.clear_all,
        ).pack(pady=4)

        result_frame = tk.LabelFrame(left_frame, text="Результати", padx=10, pady=10)
        result_frame.pack(fill="x", pady=(0, 10))

        tk.Label(result_frame, text="Дальність польоту L:").grid(
            row=0, column=0, sticky="w"
        )
        tk.Label(result_frame, textvariable=self.range_var).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

        tk.Label(result_frame, text="Максимальна висота H:").grid(
            row=1, column=0, sticky="w"
        )
        tk.Label(result_frame, textvariable=self.height_var).grid(
            row=1, column=1, sticky="w", padx=(8, 0)
        )

        tk.Label(result_frame, text="Час польоту T:").grid(
            row=2, column=0, sticky="w"
        )
        tk.Label(result_frame, textvariable=self.flight_time_var).grid(
            row=2, column=1, sticky="w", padx=(8, 0)
        )

        point_frame = tk.LabelFrame(
            left_frame,
            text="Координати і швидкість у момент t",
            padx=10,
            pady=10,
        )
        point_frame.pack(fill="x")

        tk.Label(point_frame, text="x(t):").grid(row=0, column=0, sticky="w")
        tk.Label(point_frame, textvariable=self.point_x_var).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

        tk.Label(point_frame, text="y(t):").grid(row=1, column=0, sticky="w")
        tk.Label(point_frame, textvariable=self.point_y_var).grid(
            row=1, column=1, sticky="w", padx=(8, 0)
        )

        tk.Label(point_frame, text="vx(t):").grid(row=2, column=0, sticky="w")
        tk.Label(point_frame, textvariable=self.point_vx_var).grid(
            row=2, column=1, sticky="w", padx=(8, 0)
        )

        tk.Label(point_frame, text="vy(t):").grid(row=3, column=0, sticky="w")
        tk.Label(point_frame, textvariable=self.point_vy_var).grid(
            row=3, column=1, sticky="w", padx=(8, 0)
        )

        tk.Label(point_frame, text="v(t):").grid(row=4, column=0, sticky="w")
        tk.Label(point_frame, textvariable=self.point_v_var).grid(
            row=4, column=1, sticky="w", padx=(8, 0)
        )

        self.canvas = tk.Canvas(
            right_frame,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="white",
            highlightthickness=1,
            highlightbackground="gray",
        )
        self.canvas.pack(fill="x")

        current_frame = tk.LabelFrame(
            right_frame,
            text="Поточний стан під час анімації",
            padx=10,
            pady=10,
        )
        current_frame.pack(fill="x", pady=(10, 10))

        tk.Label(
            current_frame,
            textvariable=self.current_state_var,
            justify="left",
            anchor="w",
        ).pack(fill="x")

    def read_float(self, entry, field_name):
        text = entry.get().strip().replace(",", ".")

        try:
            return float(text)
        except ValueError:
            messagebox.showerror("Помилка", f"Поле '{field_name}' заповнене неправильно.")
            return None

    def read_input_data(self):
        speed = self.read_float(self.speed_entry, "Початкова швидкість")
        angle = self.read_float(self.angle_entry, "Кут кидання")
        time_step = self.read_float(self.step_entry, "Крок часу")
        check_time = self.read_float(self.time_entry, "Час для координат")

        if None in (speed, angle, time_step, check_time):
            return None

        if speed <= 0:
            messagebox.showerror("Помилка", "Швидкість повинна бути більшою за 0.")
            return None

        if angle <= 0 or angle >= 90:
            messagebox.showerror("Помилка", "Кут повинен бути в межах від 0 до 90 градусів.")
            return None

        if time_step <= 0:
            messagebox.showerror("Помилка", "Крок часу повинен бути більшим за 0.")
            return None

        if check_time < 0:
            messagebox.showerror("Помилка", "Час не може бути від'ємним.")
            return None

        return speed, angle, time_step, check_time

    def calculate_point(self, speed, angle_rad, current_time):
        vx = speed * math.cos(angle_rad)
        vy = speed * math.sin(angle_rad) - GRAVITY * current_time
        x = speed * math.cos(angle_rad) * current_time
        y = speed * math.sin(angle_rad) * current_time - (GRAVITY * current_time ** 2) / 2

        if y < 0:
            y = 0

        total_speed = math.sqrt(vx ** 2 + vy ** 2)

        return {
            "t": current_time,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "v": total_speed,
        }

    def create_points(self, speed, angle_rad, time_step, flight_time):
        points = []
        current_time = 0.0

        while current_time < flight_time:
            points.append(self.calculate_point(speed, angle_rad, current_time))
            current_time += time_step

        points.append(self.calculate_point(speed, angle_rad, flight_time))
        return points

    def start_simulation(self):
        input_data = self.read_input_data()

        if input_data is None:
            return

        speed, angle, time_step, check_time = input_data
        angle_rad = math.radians(angle)

        flight_time = (2 * speed * math.sin(angle_rad)) / GRAVITY

        if check_time > flight_time:
            messagebox.showerror(
                "Помилка",
                f"Час t повинен бути від 0 до {flight_time:.2f} с, поки тіло ще в польоті.",
            )
            return

        self.stop_animation()

        flight_range = (speed ** 2 * math.sin(2 * angle_rad)) / GRAVITY
        max_height = (speed ** 2 * math.sin(angle_rad) ** 2) / (2 * GRAVITY)

        self.points = self.create_points(speed, angle_rad, time_step, flight_time)
        self.current_index = 0

        self.show_main_results(flight_range, max_height, flight_time)
        self.show_point_for_selected_time(speed, angle_rad, check_time)
        self.draw_scene(flight_range, max_height, check_time, speed, angle_rad)
        self.animate_ball(time_step)

    def show_main_results(self, flight_range, max_height, flight_time):
        self.range_var.set(f"{flight_range:.2f} м")
        self.height_var.set(f"{max_height:.2f} м")
        self.flight_time_var.set(f"{flight_time:.2f} с")

    def show_point_for_selected_time(self, speed, angle_rad, check_time):
        point = self.calculate_point(speed, angle_rad, check_time)

        self.point_x_var.set(f"{point['x']:.2f} м")
        self.point_y_var.set(f"{point['y']:.2f} м")
        self.point_vx_var.set(f"{point['vx']:.2f} м/с")
        self.point_vy_var.set(f"{point['vy']:.2f} м/с")
        self.point_v_var.set(f"{point['v']:.2f} м/с")

    def draw_empty_scene(self):
        self.canvas.delete("all")
        self.ground_y = CANVAS_HEIGHT - BOTTOM_MARGIN

        self.canvas.create_line(
            LEFT_MARGIN,
            self.ground_y,
            CANVAS_WIDTH - RIGHT_MARGIN,
            self.ground_y,
            width=2,
        )
        self.canvas.create_line(
            LEFT_MARGIN,
            self.ground_y,
            LEFT_MARGIN,
            TOP_MARGIN,
            width=2,
        )

        self.canvas.create_text(CANVAS_WIDTH - RIGHT_MARGIN, self.ground_y + 16, text="x")
        self.canvas.create_text(LEFT_MARGIN - 14, TOP_MARGIN, text="y")
        self.canvas.create_text(
            CANVAS_WIDTH / 2,
            15,
            text="Траєкторія польоту тіла",
            font=("Arial", 12, "bold"),
        )

    def to_canvas_x(self, x_value):
        return LEFT_MARGIN + x_value * self.scale

    def to_canvas_y(self, y_value):
        return self.ground_y - y_value * self.scale

    def draw_scene(self, flight_range, max_height, check_time, speed, angle_rad):
        self.draw_empty_scene()

        usable_width = CANVAS_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
        usable_height = CANVAS_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN

        scale_x = usable_width / max(flight_range, 1)
        scale_y = usable_height / max(max_height, 1)
        self.scale = min(scale_x, scale_y)

        line_points = []

        for point in self.points:
            line_points.append(self.to_canvas_x(point["x"]))
            line_points.append(self.to_canvas_y(point["y"]))

        if len(line_points) >= 4:
            self.canvas.create_line(line_points, fill="skyblue", width=2, smooth=True)

        start_x = self.to_canvas_x(0)
        start_y = self.to_canvas_y(0)

        self.ball_id = self.canvas.create_oval(
            start_x - 7,
            start_y - 7,
            start_x + 7,
            start_y + 7,
            fill="red",
            outline="black",
        )

        selected_point = self.calculate_point(speed, angle_rad, check_time)
        selected_x = self.to_canvas_x(selected_point["x"])
        selected_y = self.to_canvas_y(selected_point["y"])

        self.canvas.create_oval(
            selected_x - 5,
            selected_y - 5,
            selected_x + 5,
            selected_y + 5,
            fill="green",
            outline="black",
        )
        self.canvas.create_text(
            selected_x,
            selected_y - 14,
            text=f"t = {check_time:.2f} c",
            fill="darkgreen",
        )

        self.canvas.create_text(
            self.to_canvas_x(flight_range),
            self.ground_y + 16,
            text=f"L = {flight_range:.2f} м",
            fill="blue",
        )
        self.canvas.create_text(
            LEFT_MARGIN + 55,
            self.to_canvas_y(max_height) - 10,
            text=f"H = {max_height:.2f} м",
            fill="blue",
        )

    def animate_ball(self, time_step):
        if self.current_index >= len(self.points):
            self.after_id = None
            return

        point = self.points[self.current_index]
        x_canvas = self.to_canvas_x(point["x"])
        y_canvas = self.to_canvas_y(point["y"])

        self.canvas.coords(
            self.ball_id,
            x_canvas - 7,
            y_canvas - 7,
            x_canvas + 7,
            y_canvas + 7,
        )

        if self.current_index > 0:
            previous_point = self.points[self.current_index - 1]
            previous_x = self.to_canvas_x(previous_point["x"])
            previous_y = self.to_canvas_y(previous_point["y"])

            self.canvas.create_line(
                previous_x,
                previous_y,
                x_canvas,
                y_canvas,
                fill="red",
                width=2,
            )

        self.current_state_var.set(
            f"t = {point['t']:.2f} с, "
            f"x = {point['x']:.2f} м, "
            f"y = {point['y']:.2f} м, "
            f"vx = {point['vx']:.2f} м/с, "
            f"vy = {point['vy']:.2f} м/с, "
            f"v = {point['v']:.2f} м/с"
        )

        self.current_index += 1

        if self.current_index < len(self.points):
            delay = max(40, int(time_step * 500))
            self.after_id = self.root.after(delay, self.animate_ball, time_step)
        else:
            self.after_id = None

    def stop_animation(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def clear_all(self):
        self.stop_animation()

        self.points = []
        self.current_index = 0

        self.range_var.set("-")
        self.height_var.set("-")
        self.flight_time_var.set("-")
        self.point_x_var.set("-")
        self.point_y_var.set("-")
        self.point_vx_var.set("-")
        self.point_vy_var.set("-")
        self.point_v_var.set("-")
        self.current_state_var.set("Анімація ще не запущена.")
        self.draw_empty_scene()


def main():
    root = tk.Tk()
    app = ProjectileApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
