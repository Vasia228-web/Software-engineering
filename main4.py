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

from matplotlib.animation import FuncAnimation
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
MODE_LABELS = {
    "all": "Усі",
    "lagrange": "Лагранж",
    "lsm": "МНК",
}
LABEL_TO_MODE = {label: key for key, label in MODE_LABELS.items()}


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
        self.display_mode_var = tk.StringVar(value=MODE_LABELS["all"])
        self.status_var = tk.StringVar(
            value="Оберіть набір точок або відредагуйте його через поле нижче."
        )
        self.summary_var = tk.StringVar()

        self.point_editor: ScrolledText | None = None
        self.summary_text: tk.Text | None = None
        self.figure: Figure | None = None
        self.canvas: FigureCanvasTkAgg | None = None
        self.axes: np.ndarray | None = None
        self.residual_figure: Figure | None = None
        self.residual_canvas: FigureCanvasTkAgg | None = None
        self.residual_axis = None
        self.animation: FuncAnimation | None = None
        self.animation_context: dict[str, object] = {}

        self.build_layout()
        self.load_dataset_into_editor("5")
        self.refresh_summary()
        self.refresh_plots()
        self.refresh_residual_plot()

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

        ttk.Label(dataset_frame, text="Режим показу:").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )

        mode_combo = ttk.Combobox(
            dataset_frame,
            textvariable=self.display_mode_var,
            values=list(MODE_LABELS.values()),
            state="readonly",
            width=12,
        )
        mode_combo.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_changed)

        ttk.Button(
            dataset_frame,
            text="Застосувати точки",
            command=self.apply_active_dataset,
        ).grid(row=3, column=0, columnspan=2, sticky="we", pady=(10, 4))

        ttk.Button(
            dataset_frame,
            text="Скинути набір",
            command=self.reset_active_dataset,
        ).grid(row=4, column=0, columnspan=2, sticky="we", pady=4)

        ttk.Button(
            dataset_frame,
            text="Згенерувати 10/20 з 5 точок",
            command=self.regenerate_extended_datasets,
        ).grid(row=5, column=0, columnspan=2, sticky="we", pady=(4, 0))

        ttk.Button(
            dataset_frame,
            text="Анімувати активний набір",
            command=self.start_animation,
        ).grid(row=6, column=0, columnspan=2, sticky="we", pady=(8, 0))

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

        detail_frame = ttk.Frame(summary_frame)
        detail_frame.pack(fill="both", expand=True, pady=(10, 0))

        self.summary_text = tk.Text(
            detail_frame,
            wrap="word",
            height=14,
            width=70,
            font=("Courier New", 10),
        )
        self.summary_text.pack(side="left", fill="both", expand=True)
        self.summary_text.configure(state="disabled")

        residual_frame = ttk.LabelFrame(detail_frame, text="Діаграма залишків", padding=8)
        residual_frame.pack(side="right", fill="both", expand=False, padx=(12, 0))

        self.residual_figure = Figure(figsize=(3.4, 3.0), dpi=100, constrained_layout=True)
        self.residual_axis = self.residual_figure.add_subplot(111)
        self.residual_canvas = FigureCanvasTkAgg(self.residual_figure, master=residual_frame)
        self.residual_canvas.draw()
        self.residual_canvas.get_tk_widget().pack(fill="both", expand=True)

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
            marker = ">>" if name == self.active_dataset.get() else "  "
            lines.append(f"{marker} Набір {name} точок ({source}):")
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
        mode = self.current_mode()

        for row_index, name in enumerate(("5", "10", "20")):
            dataset = self.datasets[name]
            analysis = analyse_dataset(dataset, degree)
            lagrange_axis = self.axes[row_index, 0]
            lsm_axis = self.axes[row_index, 1]
            active = name == self.active_dataset.get()

            lagrange_axis.set_visible(mode in ("all", "lagrange"))
            lsm_axis.set_visible(mode in ("all", "lsm"))

            if lagrange_axis.get_visible():
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
                self.highlight_axis(lagrange_axis, active)

            if lsm_axis.get_visible():
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
                self.highlight_axis(lsm_axis, active)

        self.canvas.draw_idle()

    def refresh_residual_plot(self) -> None:
        if self.residual_axis is None or self.residual_canvas is None:
            return

        dataset = self.datasets[self.active_dataset.get()]
        analysis = analyse_dataset(dataset, int(self.degree_var.get()))
        self.residual_axis.clear()

        indexes = np.arange(dataset.count)
        colors = ["#ef4444" if value < 0 else "#22c55e" for value in analysis.residuals]
        self.residual_axis.bar(indexes, analysis.residuals, color=colors, width=0.65)
        self.residual_axis.axhline(0.0, color="#334155", linewidth=1)
        self.residual_axis.set_title(
            f"Залишки МНК - {dataset.count} точок",
            fontsize=10,
        )
        self.residual_axis.set_xlabel("Індекс точки")
        self.residual_axis.set_ylabel("y - ŷ")
        self.residual_axis.set_xticks(indexes)
        self.residual_axis.grid(True, axis="y", alpha=0.25)
        self.residual_canvas.draw_idle()

    def highlight_axis(self, axis, active: bool) -> None:
        axis.set_facecolor("#fffbea" if active else "white")
        border_color = "#f59e0b" if active else "#94a3b8"
        border_width = 2.3 if active else 1.0
        for spine in axis.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(border_width)

    def current_mode(self) -> str:
        return LABEL_TO_MODE[self.display_mode_var.get()]

    def dataset_row_index(self, name: str) -> int:
        return ("5", "10", "20").index(name)

    def configure_lagrange_axis(self, axis, dataset: DatasetState, analysis: DatasetAnalysis) -> None:
        axis.clear()
        axis.scatter(dataset.xs, dataset.ys, color="#0d6efd", zorder=2)
        axis.set_title(f"Лагранж - {dataset.count} точок", fontsize=10)
        axis.grid(True, alpha=0.25)
        axis.set_xlim(
            analysis.lagrange_limits[0],
            analysis.lagrange_limits[1],
        )
        axis.set_ylim(
            analysis.lagrange_limits[2],
            analysis.lagrange_limits[3],
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        self.highlight_axis(axis, True)

    def configure_lsm_axis(self, axis, dataset: DatasetState, analysis: DatasetAnalysis) -> None:
        axis.clear()
        axis.scatter(dataset.xs, dataset.ys, color="#0d6efd", zorder=2)
        axis.set_title(f"МНК - {dataset.count} точок", fontsize=10)
        axis.grid(True, alpha=0.25)
        axis.set_xlim(
            analysis.lsm_limits[0],
            analysis.lsm_limits[1],
        )
        axis.set_ylim(
            analysis.lsm_limits[2],
            analysis.lsm_limits[3],
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.text(
            0.02,
            0.96,
            f"MSE={analysis.mse:.4f}\nMAE={analysis.mae:.4f}\nR²={analysis.r2:.4f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#94a3b8"},
        )
        self.highlight_axis(axis, True)

    def start_animation(self) -> None:
        if self.axes is None or self.canvas is None:
            return

        self.stop_animation()
        self.refresh_plots()

        dataset_name = self.active_dataset.get()
        dataset = self.datasets[dataset_name]
        analysis = analyse_dataset(dataset, int(self.degree_var.get()))
        row_index = self.dataset_row_index(dataset_name)
        lagrange_axis = self.axes[row_index, 0]
        lsm_axis = self.axes[row_index, 1]
        lagrange_visible = lagrange_axis.get_visible()
        lsm_visible = lsm_axis.get_visible()

        context: dict[str, object] = {
            "dataset": dataset,
            "analysis": analysis,
            "row_index": row_index,
            "baseline": np.full_like(analysis.x_plot, float(np.mean(dataset.ys))),
            "frame_count": 0,
            "lagrange_line": None,
            "lagrange_nodes": None,
            "lsm_line": None,
            "lagrange_visible": lagrange_visible,
            "lsm_visible": lsm_visible,
        }

        if lagrange_visible:
            self.configure_lagrange_axis(lagrange_axis, dataset, analysis)
            lagrange_line, = lagrange_axis.plot([], [], color="#f97316", linewidth=2.2)
            lagrange_nodes = lagrange_axis.scatter([], [], color="#b45309", s=38, zorder=4)
            context["lagrange_line"] = lagrange_line
            context["lagrange_nodes"] = lagrange_nodes

        if lsm_visible:
            self.configure_lsm_axis(lsm_axis, dataset, analysis)
            baseline = context["baseline"]
            lsm_line, = lsm_axis.plot(
                analysis.x_plot,
                baseline,
                color="#0f766e",
                linewidth=2.2,
            )
            context["lsm_line"] = lsm_line

        self.animation_context = context
        frame_count = max(60, dataset.count * 8)
        self.animation_context["frame_count"] = frame_count
        self.animation = FuncAnimation(
            self.figure,
            self.animate_frame,
            frames=frame_count,
            interval=90,
            blit=False,
            repeat=False,
        )
        self.canvas.draw_idle()
        self.status_var.set(
            f"Запущено анімацію для набору {dataset_name}: вузли Лагранжа додаються по черзі, крива МНК плавно зростає."
        )

    def stop_animation(self) -> None:
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None

    def animate_lagrange(self, frame: int) -> None:
        if not self.animation_context.get("lagrange_visible"):
            return

        dataset: DatasetState = self.animation_context["dataset"]  # type: ignore[assignment]
        analysis: DatasetAnalysis = self.animation_context["analysis"]  # type: ignore[assignment]
        lagrange_line = self.animation_context["lagrange_line"]
        lagrange_nodes = self.animation_context["lagrange_nodes"]

        if lagrange_line is None or lagrange_nodes is None:
            return

        total_nodes = dataset.count
        node_count = min(total_nodes, 1 + frame // max(1, int(60 / total_nodes)))
        partial_xs = dataset.xs[:node_count]
        partial_ys = dataset.ys[:node_count]
        partial_curve = lagrange(analysis.x_plot, partial_xs, partial_ys)

        lagrange_line.set_data(analysis.x_plot, partial_curve)
        lagrange_nodes.set_offsets(np.column_stack([partial_xs, partial_ys]))

    def animate_lsm(self, frame: int) -> None:
        if not self.animation_context.get("lsm_visible"):
            return

        analysis: DatasetAnalysis = self.animation_context["analysis"]  # type: ignore[assignment]
        baseline = self.animation_context["baseline"]
        lsm_line = self.animation_context["lsm_line"]

        if lsm_line is None:
            return

        total_frames = max(1, int(self.animation_context.get("frame_count", 1)))
        alpha = min(1.0, frame / total_frames)
        animated_curve = baseline + alpha * (analysis.y_lsm - baseline)
        lsm_line.set_data(analysis.x_plot, animated_curve)

    def animate_frame(self, frame: int):
        self.animate_lagrange(frame)
        self.animate_lsm(frame)
        return []

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
        self.switch_dataset(self.active_dataset.get())

    def on_degree_changed(self, _event: object | None = None) -> None:
        self.stop_animation()
        self.refresh_summary()
        self.refresh_plots()
        self.refresh_residual_plot()
        self.status_var.set(
            f"Ступінь МНК змінено на {self.degree_var.get()}. Параметри перераховано."
        )

    def on_mode_changed(self, _event: object | None = None) -> None:
        self.switch_mode(self.display_mode_var.get())

    def switch_dataset(self, name: str) -> None:
        self.stop_animation()
        self.active_dataset.set(name)
        self.load_dataset_into_editor(name)
        self.refresh_summary()
        self.refresh_plots()
        self.refresh_residual_plot()

    def switch_mode(self, mode: str) -> None:
        self.stop_animation()
        if mode in LABEL_TO_MODE:
            self.display_mode_var.set(mode)
        elif mode in MODE_LABELS:
            self.display_mode_var.set(MODE_LABELS[mode])
        else:
            raise ValueError("Невідомий режим відображення.")

        self.refresh_plots()
        self.status_var.set(
            f"Режим відображення змінено на «{self.display_mode_var.get()}»."
        )

    def apply_active_dataset(self) -> None:
        if self.point_editor is None:
            return

        self.stop_animation()
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
        self.refresh_residual_plot()
        self.status_var.set(
            f"Набір {name} оновлено. Збережено {expected_count} коректних точок."
        )

    def reset_active_dataset(self) -> None:
        self.stop_animation()
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
        self.refresh_residual_plot()

    def regenerate_extended_datasets(self) -> None:
        self.stop_animation()
        source = self.datasets["5"]
        self.datasets["10"] = build_generated_dataset(source, 10)
        self.datasets["20"] = build_generated_dataset(source, 20)
        self.refresh_summary()
        self.refresh_plots()
        self.refresh_residual_plot()
        self.status_var.set(
            "Набори 10 і 20 точок перевизначено лінійною інтерполяцією з базових 5 точок."
        )


def main() -> None:
    root = tk.Tk()
    app = ApproximationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
