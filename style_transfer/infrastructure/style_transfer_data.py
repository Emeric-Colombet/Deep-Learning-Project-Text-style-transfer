"""This module load subtitles for our style transfer project """
from dataclasses import dataclass
import pandas as pd 


@dataclass
class StyleTransferData:
    """Represent spanish subtitles in two styles."""
    raw_data_file_name : str
    cleaned_data_file_name : str
    def load_clean_data(self) -> pd.DataFrame:
        """Collect properly arranged subtitles. 
        |id|start_time_range|movie/serie|euro_caption|latin_caption|
        """
        return "Dataset"
    def _print_to_csv(self):
        """Read raw data subtitles, and compute operations to save it in the good format"""
        pass
    def _load_raw_data(self):
        """Load raw data"""
        pass
