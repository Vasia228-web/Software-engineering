from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".mplconfig"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np


BASE_POINTS = np.array(
    [
        (1.0, 0.5),
        (2.0, 2.1),
        (3.0, 4.4),
        (4.0, 7.1),
        (5.0, 10.2),
    ],
    dtype=float,
)
ALLOWED_COUNTS = (5, 10, 20)


@dataclass
class DatasetState:
    name: str
    xs: np.ndarray
    ys: np.ndarray
    generated: bool = False

    @property
    def count(self) -> int:
        return len(self.xs)


@dataclass
class DatasetAnalysis:
    x_plot: np.ndarray
    y_lagrange: np.ndarray
    y_lsm: np.ndarray
    coeffs: np.ndarray
    residuals: np.ndarray
    mse: float
    mae: float
    r2: float
    lagrange_limits: tuple[float, float, float, float]
    lsm_limits: tuple[float, float, float, float]


def build_generated_dataset(source: DatasetState, count: int) -> DatasetState:
    new_xs = np.linspace(float(source.xs.min()), float(source.xs.max()), count)
    new_ys = np.interp(new_xs, source.xs, source.ys)
    return DatasetState(name=str(count), xs=new_xs, ys=new_ys, generated=True)


def lagrange(x: float | np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float | np.ndarray:
    x_values = np.asarray(x, dtype=float)
    result = np.zeros_like(x_values, dtype=float)

    for i in range(len(xs)):
        basis = np.ones_like(x_values, dtype=float)
        for j in range(len(xs)):
            if i == j:
                continue
            basis *= (x_values - xs[j]) / (xs[i] - xs[j])
        result += ys[i] * basis

    if np.isscalar(x) or x_values.ndim == 0:
        return float(result)
    return result


def build_dense_domain(xs: np.ndarray, points: int = 300) -> np.ndarray:
    return np.linspace(float(xs.min()), float(xs.max()), points)


def build_lsm(xs: np.ndarray, ys: np.ndarray, degree: int) -> np.ndarray:
    design_matrix = np.vander(xs, degree + 1, increasing=True)
    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, ys, rcond=None)
    return coeffs


def evaluate_polynomial(coeffs: np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
    x_values = np.asarray(x, dtype=float)
    result = np.zeros_like(x_values, dtype=float)

    for power, coeff in enumerate(coeffs):
        result += coeff * np.power(x_values, power)

    if np.isscalar(x) or x_values.ndim == 0:
        return float(result)
    return result


def calc_metrics(ys: np.ndarray, yhat: np.ndarray) -> tuple[float, float, float]:
    residuals = ys - yhat
    mse = float(np.mean(np.square(residuals)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(np.square(residuals)))
    ss_tot = float(np.sum(np.square(ys - np.mean(ys))))

    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return mse, mae, r2


def auto_scale(
    xs: np.ndarray,
    ys: np.ndarray,
    padding_ratio: float = 0.1,
    min_padding: float = 0.5,
) -> tuple[float, float, float, float]:
    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    x_padding = max((x_max - x_min) * padding_ratio, min_padding)
    y_padding = max((y_max - y_min) * padding_ratio, min_padding)

    return (
        x_min - x_padding,
        x_max + x_padding,
        y_min - y_padding,
        y_max + y_padding,
    )


def analyse_dataset(dataset: DatasetState, degree: int) -> DatasetAnalysis:
    x_plot = build_dense_domain(dataset.xs, points=400)
    y_lagrange = lagrange(x_plot, dataset.xs, dataset.ys)
    coeffs = build_lsm(dataset.xs, dataset.ys, degree)
    y_lsm = evaluate_polynomial(coeffs, x_plot)
    lsm_on_nodes = evaluate_polynomial(coeffs, dataset.xs)
    residuals = dataset.ys - lsm_on_nodes
    mse, mae, r2 = calc_metrics(dataset.ys, lsm_on_nodes)

    lagrange_limits = auto_scale(
        x_plot,
        np.concatenate([y_lagrange, dataset.ys]),
    )
    lsm_limits = auto_scale(
        x_plot,
        np.concatenate([y_lsm, dataset.ys]),
    )

    return DatasetAnalysis(
        x_plot=x_plot,
        y_lagrange=y_lagrange,
        y_lsm=y_lsm,
        coeffs=coeffs,
        residuals=residuals,
        mse=mse,
        mae=mae,
        r2=r2,
        lagrange_limits=lagrange_limits,
        lsm_limits=lsm_limits,
    )


def validate_points(xs: np.ndarray, ys: np.ndarray, expected_count: int) -> None:
    if len(xs) != len(ys):
        raise ValueError("Кількість X та Y повинна збігатися.")

    if len(xs) != expected_count:
        raise ValueError(f"Для набору {expected_count} потрібно рівно {expected_count} точок.")

    if len(np.unique(xs)) != len(xs):
        raise ValueError("Усі значення X мають бути різними.")


def parse_points_text(text: str, expected_count: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        normalized = stripped.replace(";", " ").replace(",", " ")
        parts = [part for part in normalized.split() if part]

        if len(parts) != 2:
            raise ValueError(
                "Кожен рядок повинен містити два числа: x y."
            )

        try:
            x_value = float(parts[0])
            y_value = float(parts[1])
        except ValueError as error:
            raise ValueError("Знайдено нечислове значення у списку точок.") from error

        xs.append(x_value)
        ys.append(y_value)

    xs_array = np.array(xs, dtype=float)
    ys_array = np.array(ys, dtype=float)
    validate_points(xs_array, ys_array, expected_count)

    order = np.argsort(xs_array)
    return xs_array[order], ys_array[order]


class ApproximationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ЛР4 - Інтерполяція та МНК")
        self.root.geometry("1120x720")
        self.root.minsize(980, 660)

        base_dataset = DatasetState(
            name="5",
            xs=BASE_POINTS[:, 0].copy(),
            ys=BASE_POINTS[:, 1].copy(),
            generated=False,
        )
        self.default_datasets = {
            "5": base_dataset,
            "10": build_generated_dataset(base_dataset, 10),
            "20": build_generated_dataset(base_dataset, 20),
        }
        self.datasets = {
            key: DatasetState(
                name=value.name,
                xs=value.xs.copy(),
                ys=value.ys.copy(),
                generated=value.generated,
            )
            for key, value in self.default_datasets.items()
        }

        self.active_dataset = tk.StringVar(value="5")
        self.degree_var = tk.StringVar(value="3")
        self.status_var = tk.StringVar(
            value="Оберіть набір точок або відредагуйте його через поле нижче."
        )
        self.summary_var = tk.StringVar()

        self.point_editor: ScrolledText | None = None
        self.summary_text: tk.Text | None = None
        self.figure: Figure | None = None
        self.canvas: FigureCanvasTkAgg | None = None
        self.axes: np.ndarray | None = None

        self.build_layout()
        self.load_dataset_into_editor("5")
        self.refresh_summary()
        self.refresh_plots()

    def build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y", padx=(0, 12))

        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        dataset_frame = ttk.LabelFrame(left, text="Набори даних", padding=10)
        dataset_frame.pack(fill="x")

        ttk.Label(dataset_frame, text="Активний набір:").grid(
            row=0, column=0, sticky="w"
        )

        dataset_combo = ttk.Combobox(
            dataset_frame,
            textvariable=self.active_dataset,
            values=[str(value) for value in ALLOWED_COUNTS],
            state="readonly",
            width=8,
        )
        dataset_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))
        dataset_combo.bind("<<ComboboxSelected>>", self.on_dataset_changed)

        ttk.Label(dataset_frame, text="Ступінь МНК:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )

        degree_combo = ttk.Combobox(
            dataset_frame,
            textvariable=self.degree_var,
            values=["2", "3", "4"],
            state="readonly",
            width=8,
        )
        degree_combo.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        degree_combo.bind("<<ComboboxSelected>>", self.on_degree_changed)

        ttk.Button(
            dataset_frame,
            text="Застосувати точки",
            command=self.apply_active_dataset,
        ).grid(row=2, column=0, columnspan=2, sticky="we", pady=(10, 4))

        ttk.Button(
            dataset_frame,
            text="Скинути набір",
            command=self.reset_active_dataset,
        ).grid(row=3, column=0, columnspan=2, sticky="we", pady=4)

        ttk.Button(
            dataset_frame,
            text="Згенерувати 10/20 з 5 точок",
            command=self.regenerate_extended_datasets,
        ).grid(row=4, column=0, columnspan=2, sticky="we", pady=(4, 0))

        editor_frame = ttk.LabelFrame(left, text="Редактор точок", padding=10)
        editor_frame.pack(fill="both", expand=True, pady=(12, 0))

        ttk.Label(
            editor_frame,
            text="Формат: один рядок = x y. Дозволено лише 5, 10 або 20 точок.",
            wraplength=280,
            justify="left",
        ).pack(anchor="w")

        self.point_editor = ScrolledText(editor_frame, width=34, height=22, font=("Courier New", 11))
        self.point_editor.pack(fill="both", expand=True, pady=(8, 0))

        ttk.Label(
            left,
            textvariable=self.status_var,
            wraplength=310,
            justify="left",
            foreground="#1f4f82",
        ).pack(fill="x", pady=(12, 0))

        plots_frame = ttk.LabelFrame(right, text="Графіки Лагранжа та МНК", padding=8)
        plots_frame.pack(fill="both", expand=True)

        self.figure = Figure(figsize=(9.2, 7.0), dpi=100, constrained_layout=True)
        self.axes = np.array(self.figure.subplots(3, 2), dtype=object)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        summary_frame = ttk.LabelFrame(right, text="Поточний стан наборів", padding=10)
        summary_frame.pack(fill="both", expand=False, pady=(12, 0))

        ttk.Label(
            summary_frame,
            text=(
                "Перший етап ЛР4: зберігання та редагування наборів даних.\n"
                "Далі сюди буде додано обчислення Лагранжа, МНК і графіки."
            ),
            justify="left",
        ).pack(anchor="w")

        self.summary_text = tk.Text(summary_frame, wrap="word", height=14, font=("Courier New", 10))
        self.summary_text.pack(fill="both", expand=True, pady=(10, 0))
        self.summary_text.configure(state="disabled")

    def dataset_as_text(self, dataset: DatasetState) -> str:
        return "\n".join(
            f"{x_value:.6g} {y_value:.6g}"
            for x_value, y_value in zip(dataset.xs, dataset.ys, strict=True)
        )

    def refresh_summary(self) -> None:
        if self.summary_text is None:
            return

        lines: list[str] = []
        degree = int(self.degree_var.get())
        for name in ("5", "10", "20"):
            dataset = self.datasets[name]
            source = "generated" if dataset.generated else "manual"
            lagrange_on_nodes = lagrange(dataset.xs, dataset.xs, dataset.ys)
            max_node_error = float(np.max(np.abs(lagrange_on_nodes - dataset.ys)))
            dense_x = build_dense_domain(dataset.xs, points=5)
            dense_y = lagrange(dense_x, dataset.xs, dataset.ys)
            analysis = analyse_dataset(dataset, degree)
            lines.append(f"Набір {name} точок ({source}):")
            lines.extend(
                f"  ({x_value:.3f}, {y_value:.3f})"
                for x_value, y_value in zip(dataset.xs, dataset.ys, strict=True)
            )
            lines.append(f"  Контроль Лагранжа у вузлах: max error = {max_node_error:.3e}")
            lines.append("  Пробні значення полінома Лагранжа:")
            lines.extend(
                f"    P({x_value:.3f}) = {y_value:.3f}"
                for x_value, y_value in zip(dense_x, dense_y, strict=True)
            )
            lines.append(f"  МНК, ступінь {degree}:")
            lines.append(
                "    coefficients = "
                + ", ".join(
                    f"a{index}={coeff:.5f}"
                    for index, coeff in enumerate(analysis.coeffs)
                )
            )
            lines.append(
                f"    MSE={analysis.mse:.6f}, MAE={analysis.mae:.6f}, R²={analysis.r2:.6f}"
            )
            lines.append("")

        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(lines).strip())
        self.summary_text.configure(state="disabled")

    def refresh_plots(self) -> None:
        if self.axes is None or self.figure is None or self.canvas is None:
            return

        degree = int(self.degree_var.get())

        for row_index, name in enumerate(("5", "10", "20")):
            dataset = self.datasets[name]
            analysis = analyse_dataset(dataset, degree)
            lagrange_axis = self.axes[row_index, 0]
            lsm_axis = self.axes[row_index, 1]

            lagrange_axis.clear()
            lagrange_axis.scatter(dataset.xs, dataset.ys, color="#0d6efd", zorder=3)
            lagrange_axis.plot(
                analysis.x_plot,
                analysis.y_lagrange,
                color="#f97316",
                linewidth=2,
            )
            lagrange_axis.set_title(f"Лагранж - {name} точок", fontsize=10)
            lagrange_axis.grid(True, alpha=0.25)
            lagrange_axis.set_xlim(
                analysis.lagrange_limits[0],
                analysis.lagrange_limits[1],
            )
            lagrange_axis.set_ylim(
                analysis.lagrange_limits[2],
                analysis.lagrange_limits[3],
            )
            lagrange_axis.set_xlabel("x")
            lagrange_axis.set_ylabel("y")

            lsm_axis.clear()
            lsm_axis.scatter(dataset.xs, dataset.ys, color="#0d6efd", zorder=3)
            lsm_axis.plot(
                analysis.x_plot,
                analysis.y_lsm,
                color="#0f766e",
                linewidth=2,
            )
            lsm_axis.set_title(f"МНК - {name} точок", fontsize=10)
            lsm_axis.grid(True, alpha=0.25)
            lsm_axis.set_xlim(
                analysis.lsm_limits[0],
                analysis.lsm_limits[1],
            )
            lsm_axis.set_ylim(
                analysis.lsm_limits[2],
                analysis.lsm_limits[3],
            )
            lsm_axis.set_xlabel("x")
            lsm_axis.set_ylabel("y")
            lsm_axis.text(
                0.02,
                0.96,
                f"MSE={analysis.mse:.4f}\nMAE={analysis.mae:.4f}\nR²={analysis.r2:.4f}",
                transform=lsm_axis.transAxes,
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#94a3b8"},
            )

        self.canvas.draw_idle()

    def load_dataset_into_editor(self, name: str) -> None:
        if self.point_editor is None:
            return

        dataset = self.datasets[name]
        self.point_editor.delete("1.0", tk.END)
        self.point_editor.insert("1.0", self.dataset_as_text(dataset))
        self.status_var.set(
            f"Завантажено набір {name}. Можна редагувати {dataset.count} точок і застосувати зміни."
        )

    def on_dataset_changed(self, _event: object | None = None) -> None:
        self.load_dataset_into_editor(self.active_dataset.get())

    def on_degree_changed(self, _event: object | None = None) -> None:
        self.refresh_summary()
        self.refresh_plots()
        self.status_var.set(
            f"Ступінь МНК змінено на {self.degree_var.get()}. Параметри перераховано."
        )

    def apply_active_dataset(self) -> None:
        if self.point_editor is None:
            return

        name = self.active_dataset.get()
        expected_count = int(name)
        text = self.point_editor.get("1.0", tk.END)

        try:
            xs, ys = parse_points_text(text, expected_count)
        except ValueError as error:
            messagebox.showerror("Помилка вводу", str(error))
            return

        self.datasets[name] = DatasetState(name=name, xs=xs, ys=ys, generated=False)
        self.refresh_summary()
        self.refresh_plots()
        self.status_var.set(
            f"Набір {name} оновлено. Збережено {expected_count} коректних точок."
        )

    def reset_active_dataset(self) -> None:
        name = self.active_dataset.get()
        default = self.default_datasets[name]
        self.datasets[name] = DatasetState(
            name=name,
            xs=default.xs.copy(),
            ys=default.ys.copy(),
            generated=default.generated,
        )
        self.load_dataset_into_editor(name)
        self.refresh_summary()
        self.refresh_plots()

    def regenerate_extended_datasets(self) -> None:
        source = self.datasets["5"]
        self.datasets["10"] = build_generated_dataset(source, 10)
        self.datasets["20"] = build_generated_dataset(source, 20)
        self.refresh_summary()
        self.refresh_plots()
        self.status_var.set(
            "Набори 10 і 20 точок перевизначено лінійною інтерполяцією з базових 5 точок."
        )


def main() -> None:
    root = tk.Tk()
    app = ApproximationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
