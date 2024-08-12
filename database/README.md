In this folder, i have two sql databases:

- original: "travel2.sqlite"
- backup: "travel2.backup.sqlite"

but i didnt push them to gitub remote repo

or you can manually create them from:

```python
from utils.prepare_database import DataPreparer
preparer = DataPreparer(verbose=True)
preparer.prepare_all()
```
