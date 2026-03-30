import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext line_profiler
    return


@app.cell
def _():
    from typing import Tuple

    import numpy as np


    def skew_midpoint(
        vert1: np.ndarray, direct1: np.ndarray, vert2: np.ndarray, direct2: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Find the midpoint of the line segment that is the shortest distance."""
        perp_both = np.cross(direct1, direct2)
        scale = np.dot(perp_both, perp_both)

        sp_diff = vert2 - vert1

        temp = np.cross(sp_diff, direct2)
        on1 = vert1 + direct1 * np.dot(perp_both, temp) / scale

        temp = np.cross(sp_diff, direct1)
        on2 = vert2 + direct2 * np.dot(perp_both, temp) / scale

        scale = np.linalg.norm(on1 - on2)

        res = (on1 + on2) * 0.5
        return float(scale), res

    return (np,)


@app.cell
def _(np):
    a, b, c, d = (
        np.array([0, 0, 0]),
        np.array([4, 5, 6]),
        np.array([7, 8, 9]),
        np.array([10, 11, 12]),
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %lprun -f skew_midpoint skew_midpoint(a,b,c,d)
    return


if __name__ == "__main__":
    app.run()
