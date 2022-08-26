
import pandas as pd
import json
file='data/37692_dats_2.2.json'
json1=pd.read_json(file,lines=True)

#json1=json.load(file)
df=pd.DataFrame(json1)
for row in df:
    print (row)
    
    
    ##