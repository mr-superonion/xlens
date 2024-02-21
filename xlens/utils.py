def find_number_after_keyword(file_path, keyword):
    """This function searches through the file specified by file_path for a
    line containing the keyword, then extracts and returns the number following
    the keyword as a float.

    Args:
    file_path (str):    the path to the file to be searched
    keyword (str):      the keyword to search for, followed by ": "

    Returns:
    number (float):     the number after the keyword, or None if not found
    """
    # Ensure the keyword is formatted correctly (with ": " at the end)
    if not keyword.endswith(": "):
        keyword += ": "

    # Open the file for reading
    with open(file_path, "r") as file:
        # Iterate through each line in the file
        for line in file:
            # Check if the keyword is in the current line
            if keyword in line:
                # Split the line at the keyword and take the second part (the
                # number)
                number_str = line.split(keyword)[1].strip()
                # Convert the number string to a float
                try:
                    number = float(number_str)
                    return number
                except ValueError:
                    print(f"Could not convert {number_str} to float.")
                    return None
    # Return None if the keyword was not found or if there was an issue
    # converting the number
    return None
