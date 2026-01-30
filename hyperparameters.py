import xml.etree.ElementTree as ET
from typing import List, Optional, Union

class HyperParameters:
    def __init__(self, config_path: Optional[str] = None):
        # Define hyperparameters with default values
        self.gnn_in_channels: Optional[int] = None
        self.gnn_hidden_channels: Optional[int] = None
        self.gnn_out_channels: Optional[int] = None
        self.mlp_input_size: Optional[int] = None
        self.mlp_hidden_sizes: Optional[List[int]] = None
        self.mlp_output_size: Optional[int] = None
        self.dense_size_1: Optional[int] = None
        self.dense_size_2: Optional[int] = None
        self.gru_input_size: Optional[int] = None
        self.gru_hidden_size: Optional[int] = None
        self.lr: Optional[float] = None
        self.weight_decay: Optional[float] = None
        
        # Load parameters if config file is provided
        if config_path:
            self.load_from_xml(config_path)

    def _convert_value(self, value: str) -> Union[int, float, str]:
        """Convert a string value into int, float, or keep it as a string"""
        value = value.strip()
        
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def load_from_xml(self, file_path: str):
        """Load hyperparameters from an XML file"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except (ET.ParseError, FileNotFoundError) as e:
            raise RuntimeError(f"Error loading XML file: {e}")

        for param in root:
            attr_name = param.tag
            
            if attr_name == 'mlp_hidden_sizes':
                # Special handling for mlp_hidden_sizes
                sizes = []
                for size in param.findall('size'):
                    sizes.append(self._convert_value(size.text))
                setattr(self, attr_name, sizes)
            else:
                # Handle regular parameters
                if hasattr(self, attr_name):
                    setattr(self, attr_name, self._convert_value(param.text))
                else:
                    print(f"Warning: Unrecognized parameter '{attr_name}' in XML. Ignoring.")

# Usage
config = HyperParameters(config_path='MLPconfig.xml')
# print(config.gnn_hidden_channels)  # Example Output: 68
# print(config.mlp_hidden_sizes)     # Example Output: [1024, 2469]