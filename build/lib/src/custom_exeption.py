import traceback
import sys

class CustomExeption(Exception):
    
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = self.get_detail_error_message(error_message, error_detail)

    @staticmethod #static method no need to create an object to call this function
    def get_detail_error_message(error_message, error_detail:sys):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "N/A"
        line_number = exc_tb.tb_lineno if exc_tb else "N/A"
        
        return f"Error in {file_name}, line {line_number} : {error_message}"

    def __str__(self):
        return self.error_message

