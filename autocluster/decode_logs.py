import re

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
    def get_runhistory(string):
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
            
        return history
            