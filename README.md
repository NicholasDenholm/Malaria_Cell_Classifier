# Interactive Cell Classification and Analysis

## Introduction
This notebook allows you to classify and analyze cells in a grayscale image using an interactive approach. The goal is to identify and highlight infected cells based on how much dark pixels (malaria) that they contain. You can adjust **infection_threshold** the **minimum** and **maximum size** of the regions (cells) interactively using sliders to explore how these parameters affect the cell classification.

## How to Use
1. **Min Size Adjustments**: Use the "Min Size" slider to adjust the minimum size of a region to be considered as a valid cell. The lower this value, the more small regions will be considered as cells.
2. **Max Size Adjustments**: Use the "Max Size" slider to adjust the maximum size of a region to be considered as a valid cell. Increasing this will consider larger regions as cells.
3. **Malaria Infected Cell Detection**: Infected cells are identified based on the percentage of dark pixels in each region. The **min_infected_percentage** parameter can be adjusted to define the threshold at which a cell is considered "infected".

## Outputs
- **Labeled Image**: The image where different cells are labeled with unique grayscale values from 0.5 - 1.0.
- **Histogram**: The distribution of pixel values in the labeled image.
- **Infected Cell Count**: The number of cells that are considered infected based on the defined threshold.

Modify the parameters below and see the effects on the image in real time.

---

### Note: 
Make sure image data used in this notebook is pre-processed and formatted correctly for optimal results.


The seedfill function is an implementation of a seed-fill algorithm for flood filling in an image, similar to the "paint bucket" tool in graphics software. It works by starting from a given "seed" pixel and filling all connected pixels with a specified color, as long as they have the same background color as the seed.

