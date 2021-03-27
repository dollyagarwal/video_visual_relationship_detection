import json
import os
from config import *
import pandas as pd

def loadJSON():
  with open(os.path.join(location,annotated_json)) as f:
    annotation=json.load(f)

  labels=pd.DataFrame(annotation).transpose()
  labels['path']=location+'/train/'+labels.index.astype(str)
  labels=labels[['path',0,1,2]] ##Rearrange columns
  labels.rename(columns={0:'object1',1:'relation',2:'object2'},inplace=True) ##Rename Columns
  labels.head()
  labels.to_csv(os.path.join(location,annotated_file),index=False)

if __name__ == "__main__":
  loadJSON()