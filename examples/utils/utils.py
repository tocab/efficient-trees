import time

import matplotlib.pyplot as plt


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()  # High-precision timer
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def plot_tree(tree: dict, filename="decision_tree.pdf") -> None:
    """
    Plot the tree structure and save it as a PDF.

    :param tree: The decision tree represented as a nested dictionary.
    :param filename: The name of the file to save the plot. Default is "decision_tree.pdf".
    """
    # Determine tree depth for dynamic scaling
    max_depth = _get_max_depth(tree)
    fig_height = max(4, max_depth * 2)  # Adjust height based on tree depth
    fig_width = max(10, 2**max_depth)  # Adjust width based on tree complexity

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")  # Turn off the axes
    _plot_node(ax, tree, x=0.5, y=1.0, dx=0.5, depth=0, max_depth=max_depth)
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Tree plot saved as {filename}")


def _plot_node(ax: plt.Axes, node: dict, x: float, y: float, dx: float, depth: int, max_depth: int) -> None:
    """
    Recursive helper function to plot nodes and branches.

    :param ax: The matplotlib axis object.
    :param node: The current node to plot.
    :param x: The x-coordinate of the current node.
    :param y: The y-coordinate of the current node.
    :param dx: The horizontal distance between nodes at each level.
    :param depth: The current depth in the tree.
    :param max_depth: The maximum depth of the tree for scaling purposes.
    """
    if node["type"] == "leaf":
        ax.text(
            x,
            y,
            f"Leaf\nValue: {node['value']}",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", edgecolor="black"),
            fontsize=10,
            linespacing=1.5,
        )
    else:
        ax.text(
            x,
            y,
            (
                f"Feature: {node['feature']}\n"
                f"<= {node['threshold']:.2f}\n"
                f"Criterion value: {node.get('criterion_value', 0):.2f}\n"
                f"Information Gain: {node.get('information_gain', 0):.2f}\n"
                f"Targets: {node.get('target_distribution', 'N/A')}"
            ),
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightgreen", edgecolor="black"),
            fontsize=10,
            linespacing=1.5,  # Adjust line spacing for clarity
        )
        # Plot left child
        child_dx = dx / 2  # Adjust horizontal spacing for children
        child_y = y - 1 / max_depth  # Adjust vertical spacing
        ax.plot([x, x - child_dx], [y, child_y], "k-")  # Draw line to left child
        _plot_node(ax, node["left"], x - child_dx, child_y, child_dx, depth + 1, max_depth)
        # Plot right child
        ax.plot([x, x + child_dx], [y, child_y], "k-")  # Draw line to right child
        _plot_node(ax, node["right"], x + child_dx, child_y, child_dx, depth + 1, max_depth)


def _get_max_depth(node: dict):
    """Helper function to calculate the maximum depth of the tree."""
    if node["type"] == "leaf":
        return 1
    left_depth = _get_max_depth(node["left"])
    right_depth = _get_max_depth(node["right"])
    return 1 + max(left_depth, right_depth)
