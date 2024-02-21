"""Principal RBM alpha
"""
import os
from typing import Dict, Optional, Union

import numpy as np
import scipy.io

DATA_FOLDER = "../data/"
ALPHA_DIGIT_PATH = os.path.join(DATA_FOLDER, "binaryalphadigs.mat")

def load_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load Binary AlphaDigits data from a .mat file.

    Parameters:
    - file_path (str): Path to the .mat file containing the data.

    Returns:
    - data (dict): Loaded data dictionary.
    """
    if file_path is None:
        raise ValueError("File path must be provided.")
    return scipy.io.loadmat(file_path)

def map_character_to_index(character: Union[str, int]) -> int:
    """
    Map alphanumeric character to its corresponding index.

    Parameters:
    - character (str or int): Alphanumeric character or its index.

    Returns:
    - char_index (int): Corresponding index for the character.
    """
    if isinstance(character, int) and 0 <= character <= 35:
        return character
    elif (isinstance(character, str) and character.isdigit()
          and 0 <= int(character) <= 9):
        return int(character)
    elif (isinstance(character, str) and character.isalpha()
          and 'A' <= character.upper() <= 'Z'):
        return ord(character.upper()) - ord('A') + 10
    else:
        raise ValueError(
            "Invalid character input. It should be an alphanumeric" 
            "character '[0-9|A-Z]' or its index representing '[0-35]'."
        )

def lire_alpha_digit(character: Optional[Union[str, int]] = None,
                     file_path: Optional[str] = ALPHA_DIGIT_PATH,
                     data_mat: Optional[Dict[str, np.ndarray]] = None,
                     use_data: bool = False,
                     ) -> np.ndarray:
    """
    Read Binary AlphaDigits data from a .mat file or use already loaded data,
    get the index associated with the alphanumeric character, and flatten the
    images.

    Parameters:
    - file_path (str, optional): Path to the .mat file containing the data. 
        Default is None.
    - data_mat (dict, optional): Already loaded data dictionary. 
        Default is None.
    - use_data (bool): Flag to indicate whether to use already loaded data.
        Default is False.
    - character (str or int, optional): Alphanumeric character or its index 
        whose data needs to be extracted. Default is None.

    Returns:
    - flattened_images (numpy.ndarray): Flattened images for the specified character.
    """
    if not use_data:
        data_mat = load_data(file_path)

    char_index = map_character_to_index(character)

    # Select the row corresponding to the character index
    char_data: np.ndarray = data_mat['dat'][char_index]

    # Flatten each image into a one-dimensional vector
    flattened_images = np.array([image.flatten() for image in char_data])

    return flattened_images
