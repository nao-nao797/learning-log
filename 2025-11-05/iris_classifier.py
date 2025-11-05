import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class IrisClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Classifier GUI")
        self.root.geometry("600x600")  # ウィンドウサイズ指定

        font_settings = ("Arial", 12)

        self.load_button = tk.Button(root, text="Load Iris Data", command=self.load_data,
                                     width=20, height=2, font=font_settings)
        self.load_button.pack(pady=10)

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model,
                                      width=20, height=2, font=font_settings)
        self.train_button.pack(pady=10)

        self.plot_button = tk.Button(root, text="Show Decision Boundary", command=self.plot_decision_boundary,
                                     width=25, height=2, font=font_settings)
        self.plot_button.pack(pady=10)

        self.result_label = tk.Label(root, text="Accuracy: N/A")
        self.result_label.pack()

        self.canvas = None

    def load_data(self):
        iris = load_iris()
        self.X = iris.data[:, :2]  # Use first two features for visualization
        self.y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)
        messagebox.showinfo("Info", "Iris data loaded successfully.")

    def train_model(self):
        self.model = SVC(kernel='linear')
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        self.result_label.config(text=f"Accuracy: {acc:.2f}")

    def plot_decision_boundary(self):
        if not hasattr(self, 'model'):
            messagebox.showwarning("Warning", "Train the model first.")
            return

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, alpha=0.8)
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, edgecolors='k')
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_title('Decision Boundary')

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()


# ✅ rootの定義とアプリ起動
if __name__ == "__main__":
    root = tk.Tk()
    app = IrisClassifierApp(root)
    root.mainloop()

