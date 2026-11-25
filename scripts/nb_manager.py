#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Jupyter Notebook Real-Time Manager for Teste_CCR_V7.ipynb and other notebooks.
Enables safe inspecting, reading, and updating of notebook cells.
"""

import sys
import os
import json
import shutil
from datetime import datetime

# Garantir UTF-8 na saída do console no Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_NB = r"Teste_CCR_V7.ipynb"

def load_notebook(nb_path):
    if not os.path.isabs(nb_path):
        nb_path = os.path.abspath(nb_path)
    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    return nb, nb_path

def save_notebook(nb, nb_path, backup=True):
    if backup:
        bak_path = f"{nb_path}.bak"
        shutil.copy2(nb_path, bak_path)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

def list_cells(nb_path=DEFAULT_NB):
    nb, full_path = load_notebook(nb_path)
    cells = nb.get("cells", [])
    print(f"=== Notebook: {os.path.basename(full_path)} ({len(cells)} cells) ===")
    for i, cell in enumerate(cells):
        ctype = cell.get("cell_type", "unknown")
        src_lines = cell.get("source", [])
        if isinstance(src_lines, list):
            first_line = src_lines[0].strip() if src_lines else "<empty>"
        else:
            first_line = str(src_lines).split("\n")[0].strip()
        print(f"[{i:2d}] ({ctype:5s}): {first_line[:75]}")

def get_cell(cell_idx, nb_path=DEFAULT_NB):
    nb, full_path = load_notebook(nb_path)
    cells = nb.get("cells", [])
    idx = int(cell_idx)
    if idx < 0 or idx >= len(cells):
        raise IndexError(f"Cell index {idx} out of range (0-{len(cells)-1})")
    cell = cells[idx]
    src = "".join(cell.get("source", []))
    print(f"=== Cell {idx} ({cell.get('cell_type')}) ===")
    print(src)

def set_cell_code(cell_idx, new_code_path_or_str, nb_path=DEFAULT_NB):
    nb, full_path = load_notebook(nb_path)
    cells = nb.get("cells", [])
    idx = int(cell_idx)
    if idx < 0 or idx >= len(cells):
        raise IndexError(f"Cell index {idx} out of range (0-{len(cells)-1})")
    
    if os.path.exists(new_code_path_or_str):
        with open(new_code_path_or_str, "r", encoding="utf-8") as f:
            code_text = f.read()
    else:
        code_text = new_code_path_or_str
    
    # Split into lines keeping newline
    lines = [line + "\n" for line in code_text.splitlines()]
    if code_text and not code_text.endswith("\n") and lines:
        lines[-1] = lines[-1].rstrip("\n")

    cells[idx]["source"] = lines
    save_notebook(nb, full_path, backup=True)
    print(f"Successfully updated cell {idx} in {os.path.basename(full_path)} (backup saved to .bak).")

def replace_in_cell(cell_idx, target_str, replacement_str, nb_path=DEFAULT_NB):
    nb, full_path = load_notebook(nb_path)
    cells = nb.get("cells", [])
    idx = int(cell_idx)
    if idx < 0 or idx >= len(cells):
        raise IndexError(f"Cell index {idx} out of range (0-{len(cells)-1})")
    
    cell_src = "".join(cells[idx].get("source", []))
    if target_str not in cell_src:
        raise ValueError(f"Target string not found in cell {idx}")
    
    new_src = cell_src.replace(target_str, replacement_str)
    lines = [line + "\n" for line in new_src.splitlines()]
    if new_src and not new_src.endswith("\n") and lines:
        lines[-1] = lines[-1].rstrip("\n")

    cells[idx]["source"] = lines
    save_notebook(nb, full_path, backup=True)
    print(f"Successfully replaced text in cell {idx} of {os.path.basename(full_path)}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  nb_manager.py list [nb_path]")
        print("  nb_manager.py get <cell_idx> [nb_path]")
        print("  nb_manager.py set <cell_idx> <code_or_file> [nb_path]")
        print("  nb_manager.py replace <cell_idx> <target_str> <replacement_str> [nb_path]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "list":
        path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NB
        list_cells(path)
    elif cmd == "get":
        cell_idx = int(sys.argv[2])
        path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_NB
        get_cell(cell_idx, path)
    elif cmd == "set":
        cell_idx = int(sys.argv[2])
        code_input = sys.argv[3]
        path = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_NB
        set_cell_code(cell_idx, code_input, path)
    elif cmd == "replace":
        cell_idx = int(sys.argv[2])
        tgt = sys.argv[3]
        rep = sys.argv[4]
        path = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_NB
        replace_in_cell(cell_idx, tgt, rep, path)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
