"""Getting data off course server and unzip
"""

SERVER_URL = 'https://icarus.cs.weber.edu/~hvalle/cs4580/data/'

from urllib.request import urlretrieve
import zipfile

def download_file(url, file_name):
    full_url = url + file_name
    urlretrieve(full_url,file_name)
    print(f'Downloaded {file_name} into pwd')
    

def unzip_file(file_name): 
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall()
    print(f'Unzipped {file_name} into pwd')

#TODO Create a function to download files directly from kaggle


def main():
    """Retreiving Data File
    """
    #data = 'hotel-booking-demand.zip'
    data = 'movies.csv'
    download_file(SERVER_URL,data)
    #unzip_file(data)

    # TODO: Set user input options to extract files
    # from different sources: -url, -kaggle
    


if __name__ == '__main__':
    main()