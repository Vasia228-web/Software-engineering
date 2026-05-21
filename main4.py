from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

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


def build_generated_dataset(source: DatasetState, count: int) -> DatasetState:
    new_xs = np.linspace(float(source.xs.min()), float(source.xs.max()), count)
    new_ys = np.interp(new_xs, source.xs, source.ys)
    return DatasetState(name=str(count), xs=new_xs, ys=new_ys, generated=True)


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
        self.status_var = tk.StringVar(
            value="Оберіть набір точок або відредагуйте його через поле нижче."
        )
        self.summary_var = tk.StringVar()

        self.point_editor: ScrolledText | None = None
        self.summary_text: tk.Text | None = None

        self.build_layout()
        self.load_dataset_into_editor("5")
        self.refresh_summary()

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

        ttk.Button(
            dataset_frame,
            text="Застосувати точки",
            command=self.apply_active_dataset,
        ).grid(row=1, column=0, columnspan=2, sticky="we", pady=(10, 4))

        ttk.Button(
            dataset_frame,
            text="Скинути набір",
            command=self.reset_active_dataset,
        ).grid(row=2, column=0, columnspan=2, sticky="we", pady=4)

        ttk.Button(
            dataset_frame,
            text="Згенерувати 10/20 з 5 точок",
            command=self.regenerate_extended_datasets,
        ).grid(row=3, column=0, columnspan=2, sticky="we", pady=(4, 0))

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

        summary_frame = ttk.LabelFrame(right, text="Поточний стан наборів", padding=10)
        summary_frame.pack(fill="both", expand=True)

        ttk.Label(
            summary_frame,
            text=(
                "Перший етап ЛР4: зберігання та редагування наборів даних.\n"
                "Далі сюди буде додано обчислення Лагранжа, МНК і графіки."
            ),
            justify="left",
        ).pack(anchor="w")

        self.summary_text = tk.Text(summary_frame, wrap="word", height=28, font=("Courier New", 11))
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
        for name in ("5", "10", "20"):
            dataset = self.datasets[name]
            source = "generated" if dataset.generated else "manual"
            lines.append(f"Набір {name} точок ({source}):")
            lines.extend(
                f"  ({x_value:.3f}, {y_value:.3f})"
                for x_value, y_value in zip(dataset.xs, dataset.ys, strict=True)
            )
            lines.append("")

        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(lines).strip())
        self.summary_text.configure(state="disabled")

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

    def regenerate_extended_datasets(self) -> None:
        source = self.datasets["5"]
        self.datasets["10"] = build_generated_dataset(source, 10)
        self.datasets["20"] = build_generated_dataset(source, 20)
        self.refresh_summary()
        self.status_var.set(
            "Набори 10 і 20 точок перевизначено лінійною інтерполяцією з базових 5 точок."
        )


def main() -> None:
    root = tk.Tk()
    app = ApproximationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
