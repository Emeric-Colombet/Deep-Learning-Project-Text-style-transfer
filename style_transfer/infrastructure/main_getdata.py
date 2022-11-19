from style_transfer.infrastructure.style_transfer_data import StyleTransferData
import os

DATA_PATH_RELATIVE = '../../data'
data_path = os.path.abspath(DATA_PATH_RELATIVE)

df_all = StyleTransferData().load_clean_data()
StyleTransferData.print_to_csv(df_all, data_path, 'all_subtitles.csv')

