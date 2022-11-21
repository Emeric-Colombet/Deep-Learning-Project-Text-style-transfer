from style_transfer.infrastructure.style_transfer_data import StyleTransferData
from style_transfer.domain.style_transfer_model import BaseStyleTransferModel
from style_transfer.domain.preprocess_data import PreprocessData
std = StyleTransferData('foo.csv','foo2.csv')
bstm = BaseStyleTransferModel()
ppd = PreprocessData()
print(f"Is the loader working? {std.load_clean_data()=='Dataset'}")
print(f"Is fitting method implemented? {bstm.fit(X=1,y=1)=='fit'}")
try :
    ppd.transform('df')
except NotImplementedError:
    print("All right, the transform method is not implemented")

