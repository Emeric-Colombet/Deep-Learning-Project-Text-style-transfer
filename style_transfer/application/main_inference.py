from style_transfer.domain.style_transfer_model import BaseStyleTransferModel
from style_transfer.interface.http_router import HTTPRouter
model = BaseStyleTransferModel().load()
http_router = HTTPRouter(model)
http_router.RUN()
