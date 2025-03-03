import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel


def seedfill(im, seed_row, seed_col, fill_color,bckg):
    """
    im: The image on which to perform the seedfill algorithm
    seed_row and seed_col: position of the seed pixel
    fill_color: Color for the fill
    bckg: Color of the background, to be filled
    Returns: Number of pixels filled
    Behavior: Modifies image by performing seedfill
    """
    size = 0  # keep track of patch size
    n_row, n_col = im.shape
    front = {(seed_row,seed_col)}  # initial front
    while len(front) > 0:
        r, c = front.pop()  # remove an element from front
        if im[r, c] == bckg: 
            im[r, c] = fill_color  # color the pixel
            size+= 1
            # look at all neighbors
            for i in range(max(0,r-1), min(n_row,r+2) ):
                for j in range(max(0,c-1), min(n_col,c+2) ):
                    # if background, add to front
                    if im[i,j] == bckg and (i,j) not in front:
                        front.add((i,j))
    return size

def fill_cells(edge_image):
    """
    Args:
        edge_image: A black-and-white image, with black background and
                    white edges
    Returns: A new image where each closed region is filled with a different
             grayscale value
    """
    filled_image = edge_image.copy()
    n_regions_found_so_far = 0
    # start by filling the background to dark gray, from pixel (0,0)
    s = seedfill(filled_image, 0 , 0, 0.1, 0)
    for i in range(filled_image.shape[0]):
        for j in range(filled_image.shape[1]):
            # if pixel is black, seedfill from here
            if filled_image[i,j] == 0:
                col = 0.5 + 0.001 * n_regions_found_so_far
                seedfill(filled_image, i ,j, col, 0)
                n_regions_found_so_far+= 1
    
    #io.imsave("filled_image.jpg", filled_image)

    #plt.imshow(filled_image, cmap="hot")
    #plt.title("Filled Image")  # Optional title for better context
    #plt.colorbar()
    #plt.axis('off')  # Optional: turn off axis for a cleaner display
    #plt.show()

    return filled_image

def find_coordinates_for_greyscale(labeled_image, greyscale_val):
    """
    Function to find all coordinates of the pixels that match the given `greyscale_val` in the labeled image.
    Args:
        labeled_image: The labeled image where each pixel has a grayscale value.
        greyscale_val: The grayscale value for which coordinates are to be found.
    Returns:
        A list of (row, col) coordinates where the pixel matches `greyscale_val`.
    """
    coordinates = []
    n_rows, n_cols = labeled_image.shape
    
    for i in range(n_rows): 
        for j in range(n_cols):
            if labeled_image[i, j] == greyscale_val:
                coordinates.append((i, j))
    
    return coordinates

def ensure_rgb_format(image):
    """
    Ensure the image is in RGB format.
    If the image is grayscale, convert it to RGB by repeating the grayscale values across 3 channels.
    Args:
        image: Input image, which might be grayscale.
    Returns:
        RGB image.
    """
    if len(image.shape) == 2:  # If the image is grayscale (2D array)
        image_rgb = np.stack((image,)*3, axis=-1)  # Stack the grayscale image to create 3 channels
    elif image.shape[2] == 1:  # If the image has a single channel but 3D
        image_rgb = np.repeat(image, 3, axis=-1)  # Repeat the single channel across 3 channels
    else:
        image_rgb = image  # Already in RGB format
    return image_rgb

def classify_cells(original_image, labeled_image, \
                   min_size, max_size, \
                   infected_grayscale=0.5, min_infected_percentage=0.02):

    n_row, n_col = original_image.shape

    # Ensure original_image is in RGB format
    rgb_image = ensure_rgb_format(original_image)

    # Builds a set of all grayscale values (cells) in the labeled image
    grayscales = {labeled_image[i,j] for i in range(n_row) for j in range(n_col) \
                  if labeled_image[i,j] >= 0.5 and labeled_image[i,j] < 1}

    # sets rows and cols to the labeled image array [i,j] 
    n_rows, n_cols = labeled_image.shape

    # Total number of cells (before applying the size filter)
    total_cells = len(grayscales)

    # Initialize a list to track cell sizes
    cell_sizes = []
    infected_sizes = []
    
    # Initailize a count of infected cells and store infected cell sizes
    infected = 0
    infected_cells_coords = {}
    
    #---------Pixel Iteration----------#
    for greyscale_val in grayscales:
        row_list = []
        col_list = []    
        coordinates_to_check = [] 
        
        for i in range(n_rows):
            for j in range(n_cols):
                if labeled_image[i,j] == greyscale_val:
                    col_list.append(j)
                    row_list.append(i)
                    coordinates_to_check.append([i,j])
        
        # Calculate cell size
        total_pixels = len(coordinates_to_check)
        if total_pixels < min_size or total_pixels > max_size:
            continue
        
        # Track cell size
        cell_sizes.append(total_pixels)

        dark_count = 0
        for coor in coordinates_to_check:
            if original_image[coor[0]][coor[1]] < infected_grayscale:
                dark_count +=1
        
        percent = dark_count / total_pixels
        if percent >= min_infected_percentage:
            infected += 1
            infected_sizes.append(total_pixels)

            coords = find_coordinates_for_greyscale(labeled_image, greyscale_val)
            infected_cells_coords[greyscale_val] = coords

            for i, j in zip(row_list, col_list):
                rgb_image[i, j] = [250/255, 1/255, 1/255]  # Scale values to [0, 1]
                #rgb_image[i, j] = [200, 1, 1]  # Red color (RGB)

    #---------Figure generation----------#
    # Define the grid layout for subplots (2 rows and 2 columns for 4 plots)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot the labeled image
    axs[0, 0].imshow(labeled_image, cmap="gray")
    axs[0, 0].set_title("Grayscale Image")
    axs[0, 0].axis('off')

    # Histogram of grayscale values in labeled image
    #axs[0, 1].hist(labeled_image.ravel(), bins=range(256), fc='k', ec='k')
    #axs[0, 1].set_title('Histogram of Labeled Image Grayscale Values')

    # Plot the infected vs non-infected cells count
    non_infected = total_cells - infected
    axs[0, 1].bar(['Infected', 'Non-Infected'], [infected, non_infected], color=['red', 'blue'])
    axs[0, 1].set_title('Infected vs Non-Infected Cells')
    axs[0, 1].set_ylabel('Number of Cells')

    # Plot the distribution of cell sizes (all cells and infected cells)
    axs[1, 0].hist(cell_sizes, bins=50, color='blue', alpha=0.7, label='All Cells')
    axs[1, 0].hist(infected_sizes, bins=50, color='red', alpha=0.7, label='Infected Cells')
    axs[1, 0].legend()
    axs[1, 0].set_title('Distribution of Cell Sizes')
    axs[1, 0].set_xlabel('Number of Pixels')
    axs[1, 0].set_ylabel('Frequency')

    # Plot the percentage of infected cells as a pie chart
    infected_percentage = (infected / total_cells) * 100
    axs[1, 1].pie([infected_percentage, 100 - infected_percentage], 
                  labels=['Infected', 'Non-Infected'], autopct='%1.1f%%', colors=['red', 'blue'])
    axs[1, 1].set_title('Percentage of Infected Cells')

    # Adjust layout for a clean fit
    plt.tight_layout()

    # Show the entire figure
    plt.show()

    # Print summary
    print(f'Total number of cells: {total_cells}')
    print(f'Number of infected cells: {infected}')
    print(f'Percentage of infected cells: {infected_percentage:.2f}%')

    return infected, rgb_image, infected_cells_coords

def run(image):

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns, adjust size as needed

    # Plot original image on the first subplot
    axs[0].imshow(image, cmap='gray')  # Displaying the image (grayscale or RGB)
    axs[0].set_title("Original Image")  # Title for the original image
    axs[0].axis('off')  # Optional: turn off axis for a cleaner display

    # Plot histogram of the original image on the second subplot
    axs[1].hist(image.ravel(), bins=range(256), color='k', edgecolor='k')  # Histogram of grayscale values
    axs[1].set_title('Histogram of Original Image Color Values')  # Title for the histogram
    axs[1].set_xlabel('Pixel Intensity')  # Label for the x-axis
    axs[1].set_ylabel('Frequency')  # Label for the y-axis

    # Adjust layout for clean presentation
    plt.tight_layout()
    # Show the entire figure
    plt.show()

    # Converting the image to graytone
    image_gray = rgb2gray(image)
    image_sobel = sobel(image_gray)
    image_sobel_T005 = np.where(image_sobel>=0.05,1.0, 0.0)
    
    n_row, n_col = image_sobel_T005.shape
    sobel_clean = image_sobel_T005.copy()
    for i in range(n_row):
        for j in range(n_col):
            if np.min(image_gray[max(0,i-1):min(n_row,i+2), max(0,j-1):min(n_col,j+2)])<0.5:
                sobel_clean[i,j] = 0
    image_filled = fill_cells(sobel_clean)

    
    # Adjust depending on cell size and image resolution
    min_size = 1000
    max_size = 5000
    cell_data = classify_cells(image_gray, image_filled, min_size, max_size)

    plt.imshow(cell_data[1])
    plt.title("Infected Cells Labeled in Red")
    plt.axis('off')  # Optional: turn off axis for a cleaner display
    plt.show()
    
    return cell_data[0]

if __name__ == "__main__":  # do not remove this line   
    
    #### Specify which image you want! ####
    image = plt.imread("malaria_1.jpeg")

    cells_infected = run(image)
    #print(f"{str(cells_infected)} infected cells") 
