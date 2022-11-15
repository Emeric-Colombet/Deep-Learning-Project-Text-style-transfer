from style_transfer.infrastructure.style_transfer_data import StyleTransferData
from style_transfer.domain.style_transfer_model import BaseStyleTransferModel
from style_transfer.domain.prepocess_data import PreprocessData
std = StyleTransferData('foo.csv','foo2.csv')
bstm = BaseStyleTransferModel()
pd = PreprocessData()
print(f"Is the loader working? {std.load_clean_data()=='Dataset'}")
print(f"Is fitting method implemented? {bstm.fit(X=1,y=1)=='fit'}")
try :
    pd.transform('df')
except NotImplementedError:
    print("All right, the transform method is not implemented")

