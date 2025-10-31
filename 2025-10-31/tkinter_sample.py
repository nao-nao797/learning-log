
import tkinter as tk

# ウィンドウの作成
root = tk.Tk()
root.title("こんにちは Tkinter")
root.geometry("300x150+200+0")

# ラベルの作成
label = tk.Label(root, text="ボタンを押してください")
label.pack(side="left", padx=10, pady=10)

# ボタンを押したときの処理
def on_click():
    label.config(text="こんにちは！")

# ボタンの作成
button = tk.Button(root, text="クリック", command=on_click)
button.pack(side="left")

# ウィンドウの表示
root.mainloop()
