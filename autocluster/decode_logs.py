class Decoder(object):
    @staticmethod
    def read_file_as_string(filepath):
        with open(filepath, 'r') as file:
            data = file.read().replace('\n', '')
    
    @staticmethod
    def split_logs_by_iteration(log_str):
        pass