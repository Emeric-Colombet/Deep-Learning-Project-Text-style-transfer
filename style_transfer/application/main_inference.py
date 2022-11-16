from style_transfer.domain.style_transfer_model import BaseStyleTransferModel
from style_transfer.interface.app_transfer_style import TransferStyleApp
model = BaseStyleTransferModel()
app = TransferStyleApp(model)
app.RUN()
