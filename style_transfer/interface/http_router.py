from style_transfer.domain.style_transfer_model import BaseStyleTransferModel
from dataclasses import dataclass
@dataclass
class HTTPRouter:
    model : BaseStyleTransferModel
    def RUN(self):
        pass
    def transform_style(self,sentence):
        prediction = self.model.predict(sentence)
        return prediction