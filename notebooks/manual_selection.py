import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ipywidgets GUI for the OpenPTV-Python
    """)
    return


@app.cell
def _():
    # Function to handle clicks on the image
    import imageio as iio
    import matplotlib.pyplot as plt

    # '%matplotlib notebook' command supported automatically in marimo
    # Import necessary libraries
    from io import BytesIO

    import matplotlib.widgets as widgets
    from IPython.display import display


    # Function to load image
    def load_image(change):
        global img, ax
        file_content = change["new"][0]["content"]
        img = iio.imread(BytesIO(file_content))
        ax.clear()
        ax.imshow(img, cmap="gray")
        plt.draw()


    # Function to handle clicks on the image
    def onclick(event):
        global clicks
        if event.inaxes != ax:
            return
        if event.button == 1:  # Left click to add point
            if len(clicks) < 4:
                clicks.append((event.xdata, event.ydata))
                ax.plot(event.xdata, event.ydata, "ro")
                plt.draw()
                if len(clicks) == 4:
                    print("Clicked points:", clicks)
                    calibration_data = list(zip(particle_numbers, clicks))
                    print("Calibration data:", calibration_data)
        elif event.button == 3:  # Right click to remove last point
            if clicks:
                clicks.pop()
                ax.clear()
                ax.imshow(img, cmap="gray")
                for click in clicks:
                    ax.plot(click[0], click[1], "ro")
                plt.draw()


    # File upload widget
    file_upload = widgets.FileUpload(accept="image/*", multiple=False)  # type: ignore
    file_upload.observe(load_image, names="value")

    # Integer input widgets
    particle_numbers = [
        widgets.IntText(value=i, description=f"Particle {i + 1}")
        for i in range(4)  # type: ignore
    ]

    # Display widgets
    display(file_upload)

    for widget in particle_numbers:
        display(widget)

    # Initialize variables
    clicks = []
    particle_numbers = [widget.value for widget in particle_numbers]

    # Connect the click event
    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
