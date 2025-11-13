#!/usr/bin/env python3
"""
Скрипт собирает ВЕСЬ код из .py-файлов проекта в один файл:
- Обходит дерево, пропуская .venv и служебные каталоги.
- Для каждого .py-файла записывает в all_code.txt:
  # относительный путь к файлу
  весь исходный код (с сохранением отступов, комментариев, пустых строк)
"""

import os
from pathlib import Path
from typing import List

# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #
EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "migrations/versions"}
TARGET_EXT = ".py"
OUTPUT_FILE = Path("all_code.txt")


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #
def is_excluded(path: Path) -> bool:
    """Исключаем скрытые и служебные каталоги."""
    return any(part.startswith(".") and part != "." for part in path.parts) or \
           any(part in EXCLUDE_DIRS for part in path.parts)


def process_py_file(file_path: Path, output_lines: List[str]) -> None:
    """Читает .py-файл целиком и добавляет в общий список."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Предупреждение: Не удалось прочитать {file_path} (не UTF-8)")
        return
    except Exception as e:
        print(f"Ошибка при чтении {file_path}: {e}")
        return

    output_lines.append(f"# {'='*20} {file_path.as_posix()} {'='*20}\n\n")
    output_lines.append(content)
    if not content.endswith("\n"):
        output_lines.append("\n")
    output_lines.append("\n\n")  # разделитель между файлами


# --------------------------------------------------------------------------- #
# Основная логика
# --------------------------------------------------------------------------- #
def main(root_dir: Path = Path(".")) -> None:
    if not root_dir.is_dir():
        print(f"Ошибка: {root_dir} не является директорией")
        return

    py_files = [
        p for p in root_dir.rglob(f"*{TARGET_EXT}")
        if p.is_file() and not is_excluded(p)
    ]

    if not py_files:
        print("Не найдено .py-файлов для обработки.")
        return

    output_lines: List[str] = []
    output_lines.append(f"# СБОРКА ВСЕГО КОДА ИЗ ПРОЕКТА\n")
    output_lines.append(f"# Корень: {root_dir.resolve()}\n")
    output_lines.append(f"# Время: {Path(__file__).stat().st_mtime}\n")
    output_lines.append(f"# Файлов: {len(py_files)}\n\n")
    output_lines.append("#" + "="*78 + "\n\n")

    for py_file in sorted(py_files):
        process_py_file(py_file, output_lines)

    OUTPUT_FILE.write_text("".join(output_lines), encoding="utf-8")
    print(f"Готово! Весь код сохранён в {OUTPUT_FILE.resolve()}")
    print(f"   → {len(py_files)} файлов обработано")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Собрать весь код из .py-файлов проекта в один файл"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Корневая директория проекта (по умолчанию текущая)",
    )
    args = parser.parse_args()

    main(Path(args.root))
