In this folder, i have two sql databases:

- original: "travel2.sqlite"
- backup: "travel2.backup.sqlite"

and one vector database folder:

- "chroma_langchain_db"

but i didnt push them to gitub remote repo
you can create them by:

```python
from utils.prepare_database import DataPreparer
preparer = DataPreparer(verbose=True)
preparer.prepare_all()
```
