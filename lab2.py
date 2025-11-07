import tkinter as tk
from tkinter import filedialog, simpledialog, ttk
from PIL import Image, ImageTk, ImageOps
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io

def np_from_pil(img):
    return np.asarray(img).astype(np.int32)

def pil_from_np(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def to_gray(img):
    if img.mode == "L":
        return img
    return ImageOps.grayscale(img)

def pad_reflect(a, pad_y, pad_x):
    return np.pad(a, ((pad_y,pad_y),(pad_x,pad_x)), mode='reflect')

def convolve2d(image, kernel):
    a = np_from_pil(image)
    if a.ndim == 3:
        a = np_from_pil(ImageOps.grayscale(image))
    k = np.array(kernel, dtype=np.int32)
    kh, kw = k.shape
    py, px = kh//2, kw//2
    p = pad_reflect(a, py, px)
    out = np.zeros_like(a, dtype=np.int32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = p[i:i+kh, j:j+kw]
            out[i,j] = int((region * k).sum())
    return pil_from_np(out)

def convolve2d_color_each_channel(image, kernel):
    arr = np_from_pil(image)
    if arr.ndim == 2:
        return convolve2d(image, kernel)
    k = np.array(kernel, dtype=np.int32)
    kh, kw = k.shape
    py, px = kh//2, kw//2
    out = np.zeros_like(arr, dtype=np.int32)
    for c in range(3):
        channel = arr[:,:,c]
        p = pad_reflect(channel, py, px)
        res = np.zeros_like(channel, dtype=np.int32)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i,j] = int((p[i:i+kh,j:j+kw]*k).sum())
        out[:,:,c] = res
    return pil_from_np(out)

def sobel_abs_sum(img):
    g = to_gray(img)
    a = np_from_pil(g)
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    py,px = 1,1
    p = pad_reflect(a, py, px)
    out = np.zeros_like(a, dtype=np.int32)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            rx = (p[i:i+3,j:j+3]*Gx).sum()
            ry = (p[i:i+3,j:j+3]*Gy).sum()
            out[i,j] = abs(rx)+abs(ry)
    return pil_from_np(out)

def add_images(base_img, filter_img, scale=1.0):
    b = np_from_pil(to_gray(base_img)).astype(np.int32)
    f = np_from_pil(filter_img).astype(np.int32)
    res = b + (f * scale)
    return pil_from_np(res)

def linear_contrast_stretch(img, fmin=None, fmax=None):
    g = to_gray(img)
    a = np_from_pil(g).astype(np.int32)
    if fmin is None or fmax is None:
        amin = a.min()
        amax = a.max()
    else:
        amin = int(fmin)
        amax = int(fmax)
    if amax == amin:
        return g
    out = (a - amin) * (255.0/(amax-amin))
    return pil_from_np(out)

def hist_image(img):
    g = to_gray(img)
    a = np_from_pil(g).astype(np.int32)
    hist, _ = np.histogram(a.flatten(), bins=256, range=(0,255))
    return hist

def hist_equalize_gray(img):
    g = to_gray(img)
    a = np_from_pil(g).astype(np.int32)
    hist = hist_image(g)
    H = hist.sum()
    if H == 0:
        return g
    hnorm = hist / H
    Sh = np.cumsum(hnorm)
    lut = np.floor(255 * Sh).astype(np.uint8)
    flat = a.flatten()
    mapped = lut[flat]
    return pil_from_np(mapped.reshape(a.shape))

def hist_spec_uniform(img):
    g = to_gray(img)
    a = np_from_pil(g).astype(np.int32)
    H = a.size
    L = 256
    target_per_level = H / L
    hist = hist_image(g).astype(np.int32)
    cum = np.cumsum(hist)
    mapping = np.zeros(256, dtype=np.uint8)
    level = 0
    for i in range(256):
        while level < 255 and cum[i] > (level+1)*target_per_level:
            level += 1
        mapping[i] = level
    flat = a.flatten()
    mapped = mapping[flat]
    return pil_from_np(mapped.reshape(a.shape))

def color_equalize_two_methods(img):
    if img.mode != "RGB":
        return None
    arr = np_from_pil(img)
    r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    def eq_channel(c):
        hist,_ = np.histogram(c.flatten(), bins=256, range=(0,255))
        H = hist.sum()
        if H==0:
            return c
        hnorm = hist/H
        Sh = np.cumsum(hnorm)
        lut = np.floor(255*Sh).astype(np.uint8)
        flat = c.flatten()
        return lut[flat].reshape(c.shape)
    per_channel = np.stack([eq_channel(r), eq_channel(g), eq_channel(b)], axis=2).astype(np.uint8)
    img_ycbcr = img.convert("YCbCr")
    ycb = np_from_pil(img_ycbcr)
    y = ycb[:,:,0]
    hist,_ = np.histogram(y.flatten(), bins=256, range=(0,255))
    H = hist.sum()
    if H==0:
        y_eq = y
    else:
        hnorm = hist/H
        Sh = np.cumsum(hnorm)
        lut = np.floor(255*Sh).astype(np.uint8)
        y_eq = lut[y.flatten()].reshape(y.shape)
    ycb[:,:,0] = y_eq
    by_y = Image.fromarray(ycb.astype(np.uint8), mode="YCbCr").convert("RGB")
    return Image.fromarray(per_channel, mode="RGB"), by_y

kernels = {
    "Sharpen -1 -1 -1 / -1 9 -1 / -1 -1 -1": [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]],
    "Sharpen 1 -2 1 / -2 5 -2 / 1 -2 1": [[1,-2,1],[-2,5,-2],[1,-2,1]],
    "Sharpen 0 -1 0 / -1 5 -1 / 0 -1 0": [[0,-1,0],[-1,5,-1],[0,-1,0]],
    "Sharpen 0 -1 0 / -1 20 -1 / 0 -1 0": [[0,-1,0],[-1,20,-1],[0,-1,0]],
    "Laplacian 0 -1 0 / -1 4 -1 / 0 -1 0": [[0,-1,0],[-1,4,-1],[0,-1,0]],
    "LoG 5x5": [[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]]
}

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("High-frequency filters, Histogram & Contrast - Tk")
        self.img = None
        self.proc = None
        self.setup_ui()

    def setup_ui(self):
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True)
        left = ttk.Frame(frm)
        left.grid(row=0,column=0,sticky="nswe",padx=5,pady=5)
        mid = ttk.Frame(frm)
        mid.grid(row=0,column=1,sticky="nswe",padx=5,pady=5)
        right = ttk.Frame(frm)
        right.grid(row=0,column=2,sticky="nswe",padx=5,pady=5)
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(2, weight=1)
        self.orig_label = tk.Label(left, text="Original", compound="top")
        self.orig_label.pack()
        self.orig_canvas = tk.Label(left)
        self.orig_canvas.pack()
        self.proc_label = tk.Label(mid, text="Processed", compound="top")
        self.proc_label.pack()
        self.proc_canvas = tk.Label(mid)
        self.proc_canvas.pack()
        self.hist_fig = Figure(figsize=(4,3))
        self.hist_ax = self.hist_fig.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=right)
        self.hist_canvas.get_tk_widget().pack()
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=5, pady=5)
        ttk.Button(ctrl, text="Load Image", command=self.load_image).pack(side="left")
        ttk.Button(ctrl, text="Save Processed", command=self.save_image).pack(side="left")
        ttk.Label(ctrl, text="Filter:").pack(side="left", padx=(10,0))
        self.kvar = tk.StringVar()
        cb = ttk.Combobox(ctrl, textvariable=self.kvar, values=list(kernels.keys()), state="readonly", width=45)
        cb.pack(side="left")
        cb.current(0)
        ttk.Button(ctrl, text="Apply Kernel", command=self.apply_kernel).pack(side="left", padx=5)
        ttk.Button(ctrl, text="Apply Sobel (|Gx|+|Gy|)", command=self.apply_sobel).pack(side="left", padx=5)
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=6)
        csfrm = ttk.Frame(self.root)
        csfrm.pack(fill="x", padx=5)
        ttk.Label(csfrm, text="Linear contrast stretch: ").pack(side="left")
        ttk.Button(csfrm, text="Auto Stretch", command=self.auto_stretch).pack(side="left", padx=3)
        ttk.Button(csfrm, text="Manual Stretch", command=self.manual_stretch).pack(side="left", padx=3)
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=6)
        hefrm = ttk.Frame(self.root)
        hefrm.pack(fill="x", padx=5)
        ttk.Label(hefrm, text="Histogram equalization: ").pack(side="left")
        ttk.Button(hefrm, text="Linear HE (steps 1-4)", command=self.linear_he).pack(side="left", padx=3)
        ttk.Button(hefrm, text="Nonlinear redistrib (uniform target)", command=self.nonlinear_he).pack(side="left", padx=3)
        ttk.Button(hefrm, text="Color equalize (per-channel & Y) ", command=self.color_eq_compare).pack(side="left", padx=6)
        ttk.Button(self.root, text="Reset to Original", command=self.reset).pack(pady=6)
        self.update_placeholder()

    def update_placeholder(self):
        placeholder = Image.new("L",(300,300),color=128)
        ph = ImageTk.PhotoImage(placeholder.resize((300,300)))
        self.orig_canvas.img = ph
        self.proc_canvas.img = ph
        self.orig_canvas.config(image=ph)
        self.proc_canvas.config(image=ph)
        self.hist_ax.clear()
        self.hist_ax.bar(range(256), np.zeros(256))
        self.hist_canvas.draw()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),("All files","*.*")])
        if not path:
            return
        im = Image.open(path).convert("RGB")
        self.img = im
        self.proc = im.copy()
        self.show_images()

    def save_image(self):
        if self.proc is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg")])
        if not path:
            return
        self.proc.save(path)

    def show_images(self):
        if self.img is None:
            return
        def fit_and_tk(im, w=300, h=300):
            im2 = im.copy()
            im2.thumbnail((w,h))
            return ImageTk.PhotoImage(im2)
        orig_tk = fit_and_tk(self.img)
        proc_tk = fit_and_tk(self.proc)
        self.orig_canvas.img = orig_tk
        self.proc_canvas.img = proc_tk
        self.orig_canvas.config(image=orig_tk)
        self.proc_canvas.config(image=proc_tk)
        self.update_histogram(self.proc)

    def update_histogram(self, image):
        self.hist_ax.clear()
        g = to_gray(image)
        arr = np_from_pil(g).astype(np.int32)
        hist,_ = np.histogram(arr.flatten(), bins=256, range=(0,255))
        self.hist_ax.bar(np.arange(256), hist, width=1.0)
        self.hist_ax.set_xlim(0,255)
        self.hist_canvas.draw()

    def apply_kernel(self):
        if self.img is None:
            return
        key = self.kvar.get()
        kernel = kernels[key]
        if self.img.mode == "RGB":
            proc = convolve2d_color_each_channel(self.img, kernel)
            proc = proc.convert("RGB")
        else:
            proc = convolve2d(self.img, kernel)
        self.proc = proc
        self.show_images()

    def apply_sobel(self):
        if self.img is None:
            return
        proc = sobel_abs_sum(self.img)
        self.proc = proc
        self.show_images()

    def auto_stretch(self):
        if self.img is None:
            return
        proc = linear_contrast_stretch(self.img)
        self.proc = proc
        self.show_images()

    def manual_stretch(self):
        if self.img is None:
            return
        g = to_gray(self.img)
        a = np_from_pil(g)
        amin = int(a.min())
        amax = int(a.max())
        fmin = simpledialog.askinteger("f_min", "Enter f_min", initialvalue=amin, minvalue=0, maxvalue=254)
        if fmin is None:
            return
        fmax = simpledialog.askinteger("f_max", "Enter f_max", initialvalue=amax, minvalue=1, maxvalue=255)
        if fmax is None:
            return
        proc = linear_contrast_stretch(self.img, fmin=fmin, fmax=fmax)
        self.proc = proc
        self.show_images()

    def linear_he(self):
        if self.img is None:
            return
        if self.img.mode == "RGB":
            gray = to_gray(self.img)
            eq = hist_equalize_gray(gray)
            self.proc = eq
        else:
            self.proc = hist_equalize_gray(self.img)
        self.show_images()

    def nonlinear_he(self):
        if self.img is None:
            return
        if self.img.mode == "RGB":
            gray = to_gray(self.img)
            eq = hist_spec_uniform(gray)
            self.proc = eq
        else:
            self.proc = hist_spec_uniform(self.img)
        self.show_images()

    def color_eq_compare(self):
        if self.img is None:
            return
        if self.img.mode != "RGB":
            return
        per_channel, by_y = color_equalize_two_methods(self.img)
        self.proc = per_channel
        self.show_images()
        top = tk.Toplevel(self.root)
        top.title("Comparison: Per-channel (left) vs Equalize Y (right)")
        l1 = tk.Label(top)
        l2 = tk.Label(top)
        l1.pack(side="left")
        l2.pack(side="left")
        tk1 = ImageTk.PhotoImage(per_channel.copy().resize((300,300)))
        tk2 = ImageTk.PhotoImage(by_y.copy().resize((300,300)))
        l1.img = tk1
        l2.img = tk2
        l1.config(image=tk1)
        l2.config(image=tk2)

    def reset(self):
        if self.img is None:
            return
        self.proc = self.img.copy()
        self.show_images()

root = tk.Tk()
app = App(root)
root.mainloop()

