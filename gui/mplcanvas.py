from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class StepCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=2, dpi=300):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.ax.axis("equal")
        super(StepCanvas, self).__init__(self.fig)


class PDCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.ax.axis("equal")
        super(PDCanvas, self).__init__(self.fig)


class StepVCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=2.5, height=1, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.ax.axis("equal")
        super(StepVCanvas, self).__init__(self.fig)


class PDVCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=2, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.ax.axis("equal")
        super(PDVCanvas, self).__init__(self.fig)
