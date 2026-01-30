import logging
import torch

# Configure logging (if not already set)
log_filename = "test.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler()
    ]
)

class TensorConversion:
    def __init__(self):
        pass  # No specific initialization required

    def convert_to_tensor(self, fingerprint_array):
        """
        Converts a fingerprint array to a PyTorch tensor.
        """
        logging.debug("Converting fingerprint array to tensor. Shape: %s", fingerprint_array.shape)
        tensor = torch.tensor(fingerprint_array).float()
        logging.debug("Converted tensor shape: %s", tensor.shape)  # Log tensor shape
        return tensor

    def reshape_fingerprints_array_convert_to_tensor(self, fingerprint_array):
        """
        Reshapes a 2D fingerprint array to (batch_size, 1, feature_dim) and converts it to a PyTorch tensor.
        """
        logging.debug("Received fingerprint array with shape: %s", fingerprint_array.shape)

        if len(fingerprint_array.shape) == 2:
            reshaped_array = fingerprint_array.reshape((fingerprint_array.shape[0], 1, fingerprint_array.shape[1]))
            logging.debug("Reshaped fingerprint array to: %s", reshaped_array.shape)

            tensor = torch.tensor(reshaped_array).float()
            logging.debug("Converted reshaped tensor shape: %s", tensor.shape)  # Log tensor shape
            return tensor
        else:
            error_msg = f"Unexpected input shape: {fingerprint_array.shape}"
            logging.error(error_msg)
            raise ValueError(error_msg)
