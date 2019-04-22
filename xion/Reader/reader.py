import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize

def create_df(filename):
    with open(filename) as json_data:
        data = json.load(json_data)

    filenames=[]
    frames=[]
    temp = json_normalize(data)
    for fname,d in temp.iteritems():
        filenames.append(fname)
        frame = pd.DataFrame(list(d.values)[0])
        frame['filename'] = fname
        frames.append(frame)

    df = pd.concat(frames)
    return df

def startRead(filename):
    df = create_df(filename)



if __name__=='__main__':
    startRead('/Users/kaushal/PycharmProjects/InvoiceNet/xion/XMLs/XMLInvoice.json')