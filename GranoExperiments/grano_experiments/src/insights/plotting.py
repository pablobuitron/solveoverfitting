from typing import List, Optional
import matplotlib.pyplot as plt

class TrendsPlotter:
    """
    Generates matplotlib Figure objects for one or more trends.
    Accepts multiple trends in a single plot.
    """

    def __init__(
        self,
        title: str = "Loss Trend",
        xlabel: str = "Epoch",
        ylabel: str = "Loss"
    ):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def generate_figure(
        self,
        trends: List[List[float]],
        labels: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Generates a matplotlib Figure for one or more loss trends.

        Args:
            trends: list of lists, each inner list is a loss trend
            labels: optional list of labels for each trend (must match loss_trends length)

        Returns:
            matplotlib Figure object
        """
        if not trends:
            raise ValueError("No loss trends provided.")

        if labels is not None and len(labels) != len(trends):
            raise ValueError("Length of trends_labels must match length of loss_trends.")

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, trend in enumerate(trends):
            label = labels[i] if labels else f"Trend {i + 1}"
            ax.plot(trend, label=label)

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig
