import tkinter as tk
from tkinter import colorchooser
import colorsys

def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, 1
    c = 1 - r / 255
    m = 1 - g / 255
    y = 1 - b / 255
    k = min(c, m, y)
    c = (c - k) / (1 - k)
    m = (m - k) / (1 - k)
    y = (y - k) / (1 - k)
    return round(c, 4), round(m, 4), round(y, 4), round(k, 4)

def cmyk_to_rgb(c, m, y, k):
    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)
    return int(r), int(g), int(b)

def rgb_to_hls(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return round(h * 360, 2), round(l, 4), round(s, 4)

def hls_to_rgb(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
    return int(r * 255), int(g * 255), int(b * 255)

class ColorConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Converter")
        self.root.geometry("600x600")
        self.lock = False

        self.rgb_vars = [tk.IntVar(), tk.IntVar(), tk.IntVar()]
        self.cmyk_vars = [tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()]
        self.hls_vars = [tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()]

        self.create_section("RGB", ["R", "G", "B"], self.rgb_vars, 0, 255, self.update_from_rgb, start_row=0)
        self.create_section("CMYK", ["C", "M", "Y", "K"], self.cmyk_vars, 0, 1, self.update_from_cmyk, start_row=5)
        self.create_section("HLS", ["H", "L", "S"], self.hls_vars, [0, 0, 0], [360, 1, 1], self.update_from_hls, start_row=10)

        self.color_display = tk.Label(self.root, text="", bg="#000000", width=20, height=5)
        self.color_display.grid(row=14, column=0, columnspan=6, pady=20)

        tk.Button(self.root, text="Pick a color", command=self.pick_color).grid(row=15, column=0, columnspan=6)

    def create_section(self, name, labels, vars_list, min_val, max_val, command, start_row):
        tk.Label(self.root, text=name, font=('Arial', 12, 'bold')).grid(row=start_row, column=0, pady=5, sticky="w")
        for i, label in enumerate(labels):
            tk.Label(self.root, text=label).grid(row=start_row + 1, column=i)
            entry = tk.Entry(self.root, textvariable=vars_list[i], width=6)
            entry.bind("<Return>", lambda e, cmd=command: cmd())
            entry.grid(row=start_row + 2, column=i)
            if isinstance(min_val, list):
                minv, maxv = min_val[i], max_val[i]
            else:
                minv, maxv = min_val, max_val
            scale = tk.Scale(self.root, from_=minv, to=maxv, orient="horizontal",
                             resolution=0.01 if maxv <= 1 else 1,
                             variable=vars_list[i],
                             command=lambda x, cmd=command: cmd())
            scale.grid(row=start_row + 3, column=i, padx=5, pady=5)

    def update_color_display(self, r, g, b):
        self.color_display.config(bg=f'#{r:02x}{g:02x}{b:02x}')

    def update_from_rgb(self):
        if self.lock: return
        self.lock = True
        r, g, b = [v.get() for v in self.rgb_vars]
        c, m, y, k = rgb_to_cmyk(r, g, b)
        h, l, s = rgb_to_hls(r, g, b)
        for var, val in zip(self.cmyk_vars, [c, m, y, k]): var.set(round(val, 4))
        for var, val in zip(self.hls_vars, [h, l, s]): var.set(round(val, 4))
        self.update_color_display(r, g, b)
        self.lock = False

    def update_from_cmyk(self):
        if self.lock: return
        self.lock = True
        c, m, y, k = [v.get() for v in self.cmyk_vars]
        r, g, b = cmyk_to_rgb(c, m, y, k)
        h, l, s = rgb_to_hls(r, g, b)
        for var, val in zip(self.rgb_vars, [r, g, b]): var.set(val)
        for var, val in zip(self.hls_vars, [h, l, s]): var.set(round(val, 4))
        self.update_color_display(r, g, b)
        self.lock = False

    def update_from_hls(self):
        if self.lock: return
        self.lock = True
        h, l, s = [v.get() for v in self.hls_vars]
        r, g, b = hls_to_rgb(h, l, s)
        c, m, y, k = rgb_to_cmyk(r, g, b)
        for var, val in zip(self.rgb_vars, [r, g, b]): var.set(val)
        for var, val in zip(self.cmyk_vars, [c, m, y, k]): var.set(round(val, 4))
        self.update_color_display(r, g, b)
        self.lock = False

    def pick_color(self):
        color = colorchooser.askcolor()[0]
        if color:
            r, g, b = map(int, color)
            for var, val in zip(self.rgb_vars, [r, g, b]): var.set(val)
            self.update_from_rgb()

root = tk.Tk()
app = ColorConverterApp(root)
root.mainloop()

