import os
import random
from PIL import Image, ImageTk, UnidentifiedImageError
import tkinter as tk

IMAGE_FOLDER = r"./images"
DELAY_SECONDS = 3
MAX_WIDTH = 1600
MAX_HEIGHT = 900

def get_image_files(folder):
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts]

def show_images(image_paths, delay=3):
    root = tk.Tk()
    root.title("随机图片放映")
    label = tk.Label(root)
    label.pack()

    def update_image():
        if not image_paths:
            label.config(text="没有有效图片可显示")
            return

        while True:
            img_path = random.choice(image_paths)
            try:
                img = Image.open(img_path)
                img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label.config(image=photo)
                label.image = photo
                root.geometry(f"{photo.width()}x{photo.height()}")
                break
            except (UnidentifiedImageError, OSError) as e:
                print(f"无法读取图片: {img_path}", e)
                image_paths.remove(img_path)
                if not image_paths:
                    label.config(text="所有图片都无法读取")
                    return

        root.after(int(delay * 1000), update_image)

    update_image()
    root.mainloop()

if __name__ == "__main__":
    images = get_image_files(IMAGE_FOLDER)
    if images:
        show_images(images, delay=DELAY_SECONDS)
    else:
        print("未找到图片文件。")