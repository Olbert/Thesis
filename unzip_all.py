import gzip
import shutil
from os import listdir
from os.path import isfile, join


# Images
# zipped = "E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Reconstructed\Original\Original"
# unzipped = "E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Ready\images"

# Masks
zipped = "E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Reconstructed\Silver-standard-machine-learning\Silver-standard"
unzipped = "E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Ready\masks"

# Get list of files
files = [f for f in listdir(zipped) if isfile(join(zipped, f))]

# Unzip
for i in range(0, len(files)):
    with gzip.open(zipped + "\\" + files[i], 'rb') as f_in:
        with open(unzipped + "\\" + files[i][:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Get list of files
files_num = len([f for f in listdir(zipped) if isfile(join(zipped, f))])
print(files_num, " files were unzipped")
