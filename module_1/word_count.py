from urllib.request import urlopen
"""Read file from web and do analysis of data
"""

def count_words_from_web_file(url_address):
    words = 0
    #  Read file from web
    with urlopen(url_address) as data: # the with command will open and close when it leaves scope (modern way)
        for line in data:
            # print(line, type(line))
            line = line.decode('utf-8') # convert bytes to string
            print(line, type(line))
            line_words = line.split() # string method to split, default is space
            for word in line_words:
                #  Count number of words
                words += 1
    return words

def main():
    """Driven Function
    """
    
    file_address = 'http://icarus.cs.weber.edu/~hvalle/sample_data/poem.txt'
    total_words = count_words_from_web_file(file_address)

    print(f'There are a total of {total_words} words in the file')

if __name__ == '__main__':
    main()