from src.logger import get_logger
from src.custom_exeption import CustomExeption
import sys

logger = get_logger(__name__)

def devide_number(a,b):
    try:
        result = a/b
        logger.info("deviding two numbers")
        return result
    except Exception as e:
        logger.error("Error occured")
        raise CustomExeption("Custom Error: devition by zero",sys)

if __name__ == "__main__":
    try:
        logger.info("starting main program")
        devide_number(10,2)
    except CustomExeption as ce: # keeps program runs and log the errors
        logger.error(str(ce))