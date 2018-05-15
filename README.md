#Setup
```
cd 231N-project
./download_data.sh
```
#Loading Data
```
from data import Data

d = Data()
d.get_train()
d.get_dev()		# etc
```