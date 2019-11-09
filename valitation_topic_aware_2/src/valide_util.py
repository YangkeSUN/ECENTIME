import gzip
import json

#def smart_open(file_path, *args):
#def smart_json_load(file_path):
#def text_save(content, filename, mode='a'):
#def text_read(filename):

def smart_open(file_path, *args):
    if (file_path[-3:] == '.gz'):
        return gzip.open(file_path, *args)
    return open(file_path, *args)
#end def smart_open

def smart_json_load(file_path):
    with smart_open(file_path) as file1:
        return json.load(file1)

def text_save(content, filename, mode='a'):
    file1 = smart_open(filename, mode)
    for i in range(len(content)):
        file1.write(str(content[i]) + '\n')
    file1.close()
#end def text_save

def text_read(filename):
    try:
        file1 = smart_open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file1.readlines()
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file1.close()
    return content
#end def text_read
