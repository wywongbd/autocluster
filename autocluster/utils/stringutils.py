class StringUtils(object):
    # static variables
    parameter_algorithm_separator = "___"
    parameter_conditions_separator = "|"
    
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
        
        # basic check to make sure everything is working as expected
        assert(str_ls[-1] == algorithm_name)
        
        decoded = str_ls[0]
        
        # if decoded.find(StringUtils.parameter_conditions_separator) != -1:
        #     decoded = decoded.split(StringUtils.parameter_conditions_separator)[0]
        
        return decoded
    
    @staticmethod
    def parameter_is_encoded(parameter_string):
        return (parameter_string.find(StringUtils.parameter_algorithm_separator) != -1) \
            or (parameter_string.find(StringUtils.parameter_conditions_separator) != -1)
            
        