import re
import warnings

class Decoder(object):
    @staticmethod
    def read_file_as_string(filepath):
        with open(filepath, 'r') as myfile:
            data = myfile.read()
        return data
    
    @staticmethod
    def split_logs_by_iteration(string):
        # this function returns a list of strings
        return re.findall("(ITERATION [0-9]{1} of [0-9]{1}[\s\S]{1,}?Done with ITERATION [0-9]{1})", string)
    
    @staticmethod
    def get_records_by_iteration(string_ls):
        # this function returns a list of dictionaries
        return [eval(string.split('\n')[-2]) for string in string_ls]
    
    @staticmethod
    def get_runhistory(string, sort=True):
        # the input string is the logged string from one iteration
        string_ls =  re.findall("Fitting configuration:[\s\S]{1,}?Score obtained by this configuration:[\s\S]{1,}?\\n", string)
        history = []
        
        for string in string_ls:
            without_newline_ls = string.split('\n')
            configuration = eval(without_newline_ls[1])
            score_string = without_newline_ls[-2]
            score_string = score_string[score_string.find('n:') + 2: ]
            score = None
            
            if score_string.find('inf') != -1:
                score = float('inf')
            else:
                score = eval(score_string)
    
            history.append((configuration, score))
        
        if sort:
            history = sorted(history, key=lambda tup: tup[1])
            
        return history
    
    @staticmethod
    def decode_log_file(path):
        string = Decoder.read_file_as_string(path)
        string_ls = Decoder.split_logs_by_iteration(string)
        dict_ls = Decoder.get_records_by_iteration(string_ls)

        for string, d in zip(string_ls, dict_ls):
            history = Decoder.get_runhistory(string, sort=True)
            d['runhistory'] = history
        
        metadata = {d["dataset"]: d for d in dict_ls}
        return metadata
            