import sys #will provide exception info
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() # exc_info() returns the tuple (type(e), e, e.__traceback__).
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_msg = str(error)
    error_message = f"Error occured in python script name {file_name}, line number {line_no}, error message {error_msg}"

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys) :
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
# if __name__== "__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Divide by zero error")
#         raise CustomException(e, sys)
    
#     Traceback (most recent call last):
#   File "src/exception.py", line 22, in <module>
#     a=1/0
# ZeroDivisionError: division by zero

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "src/exception.py", line 24, in <module>
#     raise CustomException(e, sys)
# __main__.CustomException: Error occured in python script name src/exception.py, line number 22, error message division by zero

    