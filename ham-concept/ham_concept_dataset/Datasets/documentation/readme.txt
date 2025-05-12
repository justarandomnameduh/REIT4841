The ground_truth_annotations/ directory contains the expert-annotated data used to train the XAI. Each subdirectory represents an annotator and each annotator's annotations are saved as individual json files representing each image. The annotations are stored as coordinates and can be converted to binary masks using the function below:

def polygon2mask(polygon, image_size=224):
    """
    Create an image mask from polygon coordinates
    """
    vertex_row_coords, vertex_col_coords, shape = polygon[:, 1], polygon[:, 0], (450, 600)
    
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=float)
    mask[fill_row_coords, fill_col_coords] = 1.
    mask = transform.resize(mask, (image_size, image_size))
    return mask
    
    

The metadata/ directory contains metadata of the ground truth annotation as well as the reader study.


The study_annotations/ directory contains annotations from phase 1 of our reader study. The annotations of each participant are stored in the same format as the ground truth annotations.
