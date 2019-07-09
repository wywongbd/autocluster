from constants import Constants

class StringUtils(object):
    @staticmethod
    def encode_parameter(parameter_name, algorithm_name):
        return parameter_name + Constants.parameter_algorithm_separator + algorithm_name
    
    @staticmethod
    def decode_parameter(encoded_parameter, algorithm_name):
        str_ls = encoded_parameter.split(Constants.parameter_algorithm_separator)
        assert(str_ls[1] == algorithm_name)
        return str_ls[0]
        