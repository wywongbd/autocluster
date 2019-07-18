import re

class Decoder(object):
    @staticmethod
    def read_file_as_string_ls(filepath):
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()
        return lines
    
    @staticmethod
    def split_logs_by_iteration(string_ls):
        # this function returns a list of list of strings
        # every list in the list belongs to one iteration
        
        start_idx_ls = []
        for i, string in enumerate(string_ls):
            iteration_start = re.findall("ITERATION [0-9]{1} of [0-9]{1}", string)
            
            if len(iteration_start) > 0:
                start_idx_ls.append(i)
        
        string_ls_ls = []
        for i, idx in enumerate(start_idx_ls):
            if i != len(start_idx_ls) - 1:
                string_ls_ls.append(string_ls[idx : start_idx_ls[i + 1]])
            else:
                string_ls_ls.append(string_ls[idx: ])
        
        return string_ls_ls
            