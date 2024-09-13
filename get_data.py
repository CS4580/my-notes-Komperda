"""Getting data off course server and unzip
"""

SERVER_URL = 'https://icarus.cs.weber.edu/~hvalle/cs4580/data/'

def download_file(url, file_name):
    from urllib.request import urlretrieve
    # TODO: Download to pwd
    full_url = url + file_name
    urlretrieve(full_url,file_name)
    print(f'Downloaded {file_name} into pwd')
    

def unzip_file(file_name):
    # TODO: Unzip file and 
    import zipfile
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall()



def main():
    """Driven Function
    """
    data = 'pandas01Data.zip'
    download_file(SERVER_URL,data)
    unzip_file(data)
    


if __name__ == '__main__':
    main()