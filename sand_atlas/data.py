import requests

server_url = "https://data.scigem-eng.sydney.edu.au/"


def list():
    """
    Requests a list of all sand types from the sand atlas server.

    Parameters:
        server_url (str): The URL of the Flask server.
        folder_name (str): The name of the folder to query for files.

    Returns:
        list: A list of filenames if the request is successful.
        None: If the request fails.
    """
    try:
        # Formulate the request URL
        url = f"{server_url}/sand-atlas-list"
        # Send a GET request with folder name as a parameter
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  # Return the list of files as JSON
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def get_all(sand, quality):
    """
    Retrieve all sand data with the specified quality.

    Parameters:
    sand (str): The name or identifier of the sand data to retrieve.
    quality (str): The quality level of the sand data. Must be one of 'ORIGINAL', '100', '30', '10', or '3'.

    Returns:
    dict or None: A dictionary containing the sand data if the request is successful, or None if an error occurs.

    Raises:
    ValueError: If the quality parameter is not one of the allowed values.
    requests.exceptions.RequestException: If there is an issue with the HTTP request.
    """
    if str(quality) not in ["ORIGINAL", "100", "30", "10", "3"]:
        raise ValueError("Quality must be one of 'ORIGINAL', '100', '30', '10', or '3'.")
    try:
        # Formulate the request URL
        url = f"{server_url}/sand-atlas-get-all"
        # Send a GET request with folder name as a parameter
        response = requests.get(url, params={"sand": sand, "quality": quality})
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file to the disk in chunks to avoid memory overload
        with open(f"{sand}_{quality}.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):  # Save in chunks of 8KB
                file.write(chunk)

        # return response.json()  # Return the list of files as JSON
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def get_by_id(sand, quality, id):
    """
    Retrieve a specific sand particle by its ID.

    Parameters:
    sand (str): The name or the sand type to retrieve.
    quality (str): The quality level of the sand data. Must be one of 'ORIGINAL', '100', '30', '10', or '3'.
    id (int): The id of the sand data item to retrieve.

    Returns:
    dict or None: A dictionary containing the sand data if the request is successful, or None if an error occurs.

    Raises:
    ValueError: If the quality parameter is not one of the allowed values.
    requests.exceptions.RequestException: If there is an issue with the HTTP request.
    """
    if str(quality) not in ["ORIGINAL", "100", "30", "10", "3"]:
        raise ValueError("Quality must be one of 'ORIGINAL', '100', '30', '10', or '3'.")
    try:
        # Formulate the request URL
        url = f"{server_url}/sand-atlas-get-by-id"
        # Send a GET request with folder name as a parameter
        response = requests.get(url, params={"sand": sand, "quality": quality, "id": id})
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file to the disk in chunks to avoid memory overload
        with open(f"{sand}_{quality}_{id}.stl", "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):  # Save in chunks of 8KB
                file.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
