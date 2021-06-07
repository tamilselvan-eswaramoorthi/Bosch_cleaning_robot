import gdown
import zipfile

url = 'https://drive.google.com/uc?id=1Un2QH_tB8tPJ7k1R72XUvhN0DtinzUEX'
output = 'model.zip'
gdown.download(url, output, quiet=False)

with zipfile.ZipFile('model.zip', 'r') as zip_ref:
    zip_ref.extractall('.')