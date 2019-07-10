class StringUtils(object):
    # static variables
    parameter_algorithm_separator = "___"
    
    @staticmethod
    def encode_parameter(parameter_name, algorithm_name):
        # if parameter is already encoded, don't encode further
        if StringUtils.parameter_is_encoded(parameter_name):
            return parameter_name
        else:
            return parameter_name + StringUtils.parameter_algorithm_separator + algorithm_name
    
    @staticmethod
    def decode_parameter(encoded_parameter, algorithm_name):
        str_ls = encoded_parameter.split(StringUtils.parameter_algorithm_separator)
        assert(str_ls[-1] == algorithm_name)
        return str_ls[0]
    
    @staticmethod
    def parameter_is_encoded(parameter_string):
        return parameter_string.find(StringUtils.parameter_algorithm_separator) != -1
            
        