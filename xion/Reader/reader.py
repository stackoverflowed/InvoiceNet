import argparse
import re
from enum import Enum
import sys
import datefinder
import pandas as pd
import numpy as np
import json

from nltk import ngrams
from pandas.io.json import json_normalize
import string

from model import InvoiceNetCloudScan


def clean_text(text):
    return str(text).strip().lstrip('.').strip().replace(',','')

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
    df['text']=df['text'].map(lambda t:clean_text(t))
    return df


class LABELS(Enum):
    UNK = (0, 'UNKNOWN', '')
    INVOICE_NO = (1, 'Invoice_No', '')
    DATE = (2,'Date','')
    DUE_ON = (3, 'Due on', '')
    TOTAL_VAL = (4,'Total','')
    TAX = (5,'Tax','')


def ngrammer(tokens, length=2):
    """
    Generates n-grams from the given tokens
    :param tokens: list of tokens in the text
    :param length: n-grams of up to this length
    :return: n-grams as tuples
    """
    for n in range(1, min(len(tokens) + 1, length+1)):
        for gram in ngrams(tokens, n):
            yield gram


def get_label(gram_row, df_excel,labels):

    filtered_df = pd.merge(df_excel,gram_row,left_on=['FILENAME'],right_on=['filename'])
    if not pd.merge(filtered_df,gram_row,left_on=['Invoice Page','Invoice Value'],right_on= ['pageNo','text']).empty:
        labels.append(LABELS.INVOICE_NO.value[0])
    if not pd.merge(filtered_df,gram_row,left_on=['Date Page','Date Value'],right_on= ['pageNo','text']).empty:
        labels.append(LABELS.DATE.value[0])
    if not pd.merge(filtered_df, gram_row, left_on=['Due on Page', 'Due on Value'],
                    right_on=['pageNo', 'text']).empty:
        labels.append(LABELS.DUE_ON.value[0])
    if not pd.merge(filtered_df,gram_row,left_on=['Total Page','Total Value'],right_on= ['pageNo','text']).empty:
        labels.append(LABELS.TOTAL_VAL.value[0])
    if not pd.merge(filtered_df,gram_row,left_on=['Tax Page','Tax Value'],right_on= ['pageNo','text']).empty:
        labels.append(LABELS.TAX.value[0])




def get_labels_ngrams(df,line_df,df_excel):
    #df_invoice = df_excel['FILENAME','Invoice Label','Invoice Page','Invoice Value']
    #joined_df_invoice = pd.merge(df,df_invoice,how='left',left_on=['filename','pageNo'],right_on=['FILENAME','Invoice Page'])

    grams = {'raw_text': [],
             'processed_text': [],
             'text_pattern': [],
             'length': [],
             'line_size': [],
             'position_on_line': [],
             'has_digits': [],
             'bottom_margin': [],
             'top_margin': [],
             'left_margin': [],
             'right_margin': [],
             'page_width': [],
             'page_height': [],
             'parses_as_amount': [],
             'parses_as_date': [],
             'parses_as_number': [],
             'label': [],
             'closest_ngrams': []
             }

    for index,row in line_df.iterrows():
        tokens = row['text'].split(' ')
        num_ngrams = len(grams['raw_text'])
        for ngram in ngrammer(tokens):
            grams['parses_as_date'].append(0.0)
            grams['parses_as_amount'].append(0.0)
            grams['parses_as_number'].append(0.0)
            processed_text = []
            for word in ngram:
                if bool(list(datefinder.find_dates(word))):
                    processed_text.append('date')
                    grams['parses_as_date'][-1] = 1.0
                elif bool(re.search(r'\d\.\d', word)) or '$' in word:
                    processed_text.append('amount')
                    grams['parses_as_amount'][-1] = 1.0
                elif word.isnumeric():
                    processed_text.append('number')
                    grams['parses_as_number'][-1] = 1.0
                else:
                    processed_text.append(word.lower())
            raw_text = ' '.join(ngram)
            grams['raw_text'].append(raw_text)
            grams['processed_text'].append(' '.join(processed_text))
            grams['text_pattern'].append(re.sub('[a-z]', 'x', re.sub('[A-Z]', 'X', re.sub('\d', '0', re.sub(
                '[^a-zA-Z\d\ ]', '?', raw_text)))))
            grams['length'].append(len(' '.join(ngram)))
            grams['line_size'].append(len(tokens))
            grams['position_on_line'].append(tokens.index(ngram[0]) / len(tokens))
            grams['has_digits'].append(1.0 if bool(re.search(r'\d', raw_text)) else 0.0)

            minX,maxX=0,0
            top,bottom=0,0
            page_height,page_width=0,0
            labels=[]
            for index,gram in enumerate(ngram):
                gram_row = df[(df['filename']==row['filename'])&(df['pageNo']==row['pageNo'])&(df['text']==gram)]
                if index==0:
                    page_height,page_width=gram_row['pageHeight'].values[0],gram_row['pageWidth'].values[0]
                    minX = gram_row['l']/gram_row['pageWidth']
                    top=gram_row['t']/gram_row['pageHeight']
                    bottom=gram_row['b']/gram_row['pageHeight']
                if index==len(ngram)-1:
                    maxX = gram_row['r']/gram_row['pageWidth']
                get_label(gram_row,df_excel,labels)

            if not labels:
                labels.append(LABELS.UNK.value[0])
            grams['page_width'].append(page_width)
            grams['page_height'].append(page_height)
            grams['left_margin'].append(minX.values[0])
            grams['right_margin'].append(maxX.values[0])
            grams['bottom_margin'].append(bottom.values[0])
            grams['top_margin'].append(top.values[0])
            grams['label'].append(labels[0])

        for i in range(num_ngrams, len(grams['raw_text'])):
            grams['closest_ngrams'].append([-1] * 4)
            distance = [sys.maxsize] * 6
            for j in range(num_ngrams, len(grams['raw_text'])):
                d = [grams['top_margin'][i] - grams['bottom_margin'][j],
                     grams['top_margin'][j] - grams['bottom_margin'][i],
                     grams['left_margin'][i] - grams['right_margin'][j],
                     grams['left_margin'][j] - grams['right_margin'][i],
                     abs(grams['left_margin'][i] - grams['left_margin'][j])]
                if i == j:
                    continue
                # If in the same line, check for closest ngram to left and right
                if d[0] == d[1]:
                    if distance[2] > d[2] > 0:
                        distance[2] = d[2]
                        grams['closest_ngrams'][i][2] = j
                    if distance[3] > d[3] > 0:
                        distance[3] = d[3]
                        grams['closest_ngrams'][i][3] = j
                # If this ngram is above current ngram
                elif distance[0] > d[0] >= 0 and distance[4] > d[4]:
                    distance[0] = d[0]
                    distance[4] = d[4]
                    grams['closest_ngrams'][i][0] = j
                # If this ngram is below current ngram
                elif distance[1] > d[1] >= 0 and distance[5] > d[4]:
                    distance[1] = d[1]
                    distance[5] = d[4]
                    grams['closest_ngrams'][i][1] = j



    final_df =  pd.DataFrame(data=grams)
    #final_df = final_df[final_df['label']!=0]

    final_df.to_pickle('pickle/grams_df.pickle',protocol=3)
    return final_df




# def get_invoice_entries(df):
#
#     df_excel['FILENAME'] = df_excel['FILENAME'].map(lambda fname:'{}.pdf.xml'.format(fname))
#     df_excel['Invoice Value']=df_excel['Invoice Value'].map(lambda invno:clean_text(invno))
#     final_df = pd.merge(df,df_excel[['FILENAME','Invoice Value']],left_on=['filename','text'],right_on=['FILENAME','Invoice Value'],how='inner')
#     return final_df

#
# def get_left_text(df,**kwargs):
#     inv_nos = kwargs['inv_nos']
#     for index,row in df.iterrows():
#         for inv_no in inv_nos:
#
#     return sorted_left.reset_index(drop=True)

def get_line_df(df):
    def f(x):
        return pd.Series(dict(l=x['l'].min(),
                              r=x['r'].max(),
                              text=' '.join(x['text'])))
    df = df.groupby(['filename', 'pageNo', 'b']).apply(f).reset_index()
    return df


def startRead(filename,excel_file):
    df = create_df(filename).sort_values(['filename','pageNo','b','l']).reset_index(drop=True)
    df['pageNo'] = df['pageNo'].astype(np.int8)
    df['text']=df['text'].map(clean_text)
    df_excel = pd.read_excel(excel_file, header=0,converts={'Total Value':str,'Tax Value':str}).applymap(clean_text)
    df_excel['FILENAME'] = df_excel['FILENAME'].map(lambda fname: '{}.pdf.xml'.format(fname))
    df_excel['Total Value']=df_excel['Total Value'].map(lambda amt: amt.lstrip('$'))
    df_excel['Tax Value'] = df_excel['Tax Value'].map(lambda amt: amt.lstrip('$'))
    page_col_list = ['Invoice Page','Date Page','Total Page','Tax Page','Due on Page']
    for page_col in page_col_list:
        df_excel[page_col]=pd.to_numeric(df_excel[page_col],errors='coerce', downcast='integer').fillna(-1).astype(np.int8)

    #lines_df = get_line_df(df)
    #invoice_df = get_invoice_entries(df)
    #bottoms = set(invoice_df.b.unique())
    #inv_nos = [clean_text(invoice_no) for invoice_no in df_excel['Invoice Value'].unique()]

    #filtered_df = df[df['b'].isin(bottoms)]
    #sorted_left = get_left_text(filtered_df)
    line_df = get_line_df(df)
    features=get_labels_ngrams(df,line_df,df_excel)
    print('')


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", type=str, choices=["train", "test"],
                    required=True, help="train|test", default="train")
    ap.add_argument("--data", default="data/dftrain.pk",
                    help="path to training data")
    ap.add_argument("--model_path", default="./model",
                    help="path to directory where trained model should be stored")
    ap.add_argument("--load_weights", default="./model/InvoiceNetCloudScan.model",
                    help="path to load weights")
    ap.add_argument("--checkpoint_dir", default="./checkpoints",
                    help="path to directory where checkpoints should be stored")
    ap.add_argument("--log_dir", default="./logs",
                    help="path to directory where tensorboard logs should be stored")
    ap.add_argument("--num_hidden", type=int, default=256,
                    help="size of hidden layer")
    ap.add_argument("--epochs", type=int, default=50,
                    help="number of epochs")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="size of mini-batch")
    ap.add_argument("--num_layers", type=int, default=1,
                    help="number of layers")
    ap.add_argument("--num_input", type=int, default=17,
                    help="size of input layer")
    ap.add_argument("--num_output", type=int, default=4,
                    help="size of output layer")
    ap.add_argument("--shuffle", action='store_true',
                    help="shuffle dataset")
    ap.add_argument("--oversample", type=int, default=0,
                    help="oversample minority classes to prevent class imbalance")

    args = ap.parse_args()
    net = InvoiceNetCloudScan(config=args)

    #startRead('/Users/kaushal/PycharmProjects/InvoiceNet/xion/XMLs/XMLInvoice.json',excel_file='../excel/Invoice Data.xlsx')
    #startRead('/Users/kaushal/PycharmProjects/InvoiceNet/xion/XMLs/XMLInvoiceBig.json',
    #          excel_file='../excel/Invoice Data.xlsx')

    features = pd.read_pickle('pickle/grams_df.pickle')

    if args.mode == 'train':
        net.train(features)
    else:
        net.load_weights(args.load_weights)
        predictions = net.evaluate(features)
        net.f1_score(predictions, features.label.values)
        nonzeros = features[features['label']!=0]
        for idx,row in nonzeros.iterrows():
            print('{} | {} | {}'.format(row['raw_text'],row['label'],predictions[idx]))


if __name__=='__main__':
    main()

