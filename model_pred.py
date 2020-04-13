import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re, pickle
le = LabelEncoder()
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pickle
import random, json , requests
#from pymongo import MongoClient
import csv



# url = 'https://tcsdemo3.service-now.com/api/now/table/incident'
# user = 'tcs_2'
# pwd = 'Mobapp147'

# # Set proper headers
# headers = {"Content-Type":"application/json","Accept":"application/json"}


def doc_path(file_name):
    dfnew = pd.read_csv(file_name)
    dfnew.drop(['Unnamed: 0','MASTERSERVICE','FIRSTOCCURRENCE','LASTOCCURRENCE','LASTFAULTSUMMARY'],axis=1,inplace= True)
    dfnew.dropna(inplace=True)
    print("original data frame shape : ", dfnew.shape)

    # # ##Generating features
    dfnew['TTT'] = dfnew['HIGHESTSEVERITY'].apply(lambda x: 1 if 'Critical (5)' in x or 'Major (4)' in x else 0).astype(int)
    dfnew['SUMMARY'] = dfnew['FIRSTEVENTSUMMARY'].apply(lambda x: 0 if 'Cleared' in x or 'Warning' in x else 1).astype(int)
    dfnew['FIRSTEVENTSUMMARYCODE'] = dfnew['FIRSTEVENTSUMMARY'].apply(lambda x: 0 if 'Cleared' in x else (1 if 'Warning' in x else 2) )
    dfnew['tallyequal']='dummy'
    dfnew['tallygreater']='dummy'
    dfnew['tallylesser']='dummy'
    dfnew['tallyequal'] = dfnew.apply(lambda x: 1 if x['TALLY'] == x['CLEARTALLY'] else 0,axis=1).astype(int)
    dfnew['tallygreater'] = dfnew.apply(lambda x: 1 if x['TALLY'] > x['CLEARTALLY'] else 0,axis=1).astype(int)
    dfnew['tallylesser'] = dfnew.apply(lambda x: 1 if x['TALLY'] < x['CLEARTALLY'] else 0,axis=1).astype(int)


    ###Preprocessing
    #rename columns
    dfnew.rename(columns={"CLASS": "clss", "MASTER": "mstr", "NODE": "nd","SUMMARY":"smry"}, inplace=True)

    #convert all column names to lower case
    dfnew.columns = map(str.lower, dfnew.columns)

    #convert all data to lower case
    dfnew = dfnew.apply(lambda x: x.astype(str).str.lower())

    #remove punctuations from data frame
    cols=dfnew.columns
    dfnew.loc[:, cols] = dfnew[cols].apply(lambda s: s.str.replace(rf'[{punctuation}]', ''))

    ###label encoding
    '''Use Pickle Transformation'''
    cat_var=['nd','alarmkey','clss','highestseverity','mstr']
    for cat in cat_var:
        pkl = pd.read_pickle(cat+'.pkl')
        #print(pkl)
        dfnew[cat] = pkl.transform(dfnew[cat].astype(str))
    pkl=None
    
    ####  NLTK  
    #remove stop words
    dfnew.apply(lambda x: [item for item in x if item not in stop])

    #tokenize and lemmatize function
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in tokens])
        return lemmatized_output

    #apply tokenize and lemmatize function
    dfnew['firsteventsummary']=dfnew['firsteventsummary'].apply(tokenize)


    ##Vectorization
    '''Use Pickle Transformation'''

    vect=pd.read_pickle('vectorized.pkl')
    docs=dfnew['firsteventsummary'].tolist()
    tfidf_vectorizer_vectors=vect.transform(docs).todense()
    colu=vect.get_feature_names()
    vect=None

    ### creating vectorized output
    df1 = pd.DataFrame(tfidf_vectorizer_vectors, columns=colu)
    print("vectorized dtaframe shape :", df1.shape )

    dfnew.reset_index(drop=True, inplace=True)
    df1.reset_index(drop=True, inplace=True)

    res = pd.concat([dfnew, df1], axis=1)
    print("final dataframe shape :",res.shape)

    ## create train and test 
    X=res.drop(['firsteventsummary','troubleticket','unnamed: 0.1'],axis=1)
    #print(X)
    y=res['troubleticket']
    # print(y)
    res=None

    # prediction transportation to test model
    X['tallyequal']=X['tallyequal'].astype(int)
    X['cleartally']=X['cleartally'].astype(int)
    X['smry']=X['summary'].astype(int)
    X['tally']=X['tally'].astype(int)
    X['tallygreater']=X['tallygreater'].astype(int)
    X['tallylesser']=X['tallylesser'].astype(int)
    X['firsteventsummarycode']=X['firsteventsummarycode'].astype(int)
    X['ttt']=X['ttt'].astype(int)


    ## Load model
    Pkl_Filename =  "C:\\Users\\Karthi\\Desktop\\api\\VotingClassifier_NetcoolAlarms_subset.pkl"
    with open(Pkl_Filename, 'rb') as file:  
        votingclassifier = pickle.load(file)
    
    # predict class
    y_hat = votingclassifier.predict(X)
    # print(classification_report(y, y_hat))
    
    
    ##
    y_hat= y_hat.tolist()
    y_hat=pd.DataFrame(y_hat)
    y_hat['predictedvalue'] = y_hat
    y_hat=y_hat.drop([0],axis=1)
    outconcat = pd.concat([y, y_hat], axis=1)
    
    outconcat['result']=outconcat.apply(lambda x: 'CorrectPrediction' if x['troubleticket'] == x['predictedvalue'] else 'IncorrectPrediction',axis=1).astype(str)
    outconcat['incident']=outconcat.apply(lambda x: 'ConvertToTicket' if x['troubleticket']and x['predictedvalue'] == '1' else 'FalseAlarm',axis=1).astype(str)
    
    #dfnew_true = dfnew.loc[dfnew['TROUBLETICKET'] == 1]
    outconcat.rename(columns = {'troubleticket':'actualvalue'}, inplace = True)
    dfnew_true = outconcat.loc[outconcat['incident'] == 'ConvertToTicket']
    t_df = dfnew[dfnew.index.isin(dfnew_true.index)]
    t_df.reset_index(drop=True, inplace=True)
    # print(dfnew)
    a_df = dfnew[dfnew.index.isin(outconcat.index)]
    ##Duplication column
    boolean = a_df['firsteventsummary'].duplicated()
    a_df['duplicated']=""
    for i in range(len(boolean)):
        if(boolean[i] == True):
            a_df['duplicated'][i] = "Yes"
        else:
            a_df['duplicated'][i] = "No"

    ##Status segment
    # csvfile = open('status.csv', 'r')
    # reader = csv.DictReader( csvfile )
    # conn = MongoClient('mongodb://localhost:27017')
    # # database 
    # db = conn.database
    
    # #create collection status
    # db.status
    # def status_fn():
    #     header= ['month', 'incident id', 'priority', 'status', 'assigned group','incident last resolved date', 'summary'] 

    #     for each in reader:
    #         row={}
    #         for field in header:
    #             row[field]=each[field]
    #         db.status.insert(row)

    # status_fn()
    
    
    # ##Converting t_df to CSV
    # df_dup = t_df[['alarmkey','clss','firsteventsummary','troubleticket']]
    # # df_dup = df_dup.to_frame()
    # df_dup.to_csv("my_file.csv")

    ##Data to mongodb
    # csvfile = open("E:\\netcool\\api\\my_file.csv", 'r')
    # reader = csv.DictReader( csvfile )
    # conn = MongoClient('mongodb://localhost:27017')
    # db = conn.database
    # collection = db.segment
    # header = ['alarmkey','clss','firsteventsummary','troubleticket']

    # key1 = "duplicate"
    # for each in reader:
    #     row={}
    #     if(each['firsteventsummary'] in collection.distinct('firsteventsummary')):
    #         itm = db.segment.find_one({'firsteventsummary':each['firsteventsummary']})
    #         db.segment.update({'_id': itm.get('_id')}, {'$set': {'value.' + key1: 'yes'},"$currentDate":{"lastModified":True}})
    #         try:
    #             s_id = db.status.find_one({'summary': 'database sldp1 1 user password due expire profile 131941'}).get('_id')
    #             status = db.status.find_one({'_id':s_id})
    #             if(status['status'] == 'closed'):
    #                 print("Entered")
    #                 collection.insert_one({
    #                     "alarmkey":"35",
    #                     "clss":"2",
    #                     "firsteventsummary":each['firsteventsummary'],
    #                     "troubleticket":"1"
    #                 })
    #                 print("Create new ticket")
    #                 header= ['month', 'incident id', 'priority', 'status', 'assigned group','incident last resolved date', 'summary'] 

    #                 db.status.insert({'incident id':'IN15436364',
    #                                 'summary':each['firsteventsummary'],
    #                                 'status':'open',
    #                                 'assignment group':'hardware',
    #                                 })


    #             else:
    #                 print("already ticket raised")
    #         except (TypeError, AttributeError):
    #             print("None")

    #     else:
    #         for field in header:
    #             row[field]=each[field]
    #         collection.insert(row)
    # lst_id = []
    # lst_alarm = []
    # lst_clss = []
    # lst_sum = []
    # lst_tt = []
    # for obj in collection.find():
    #     lst_id.append(obj['_id'])
    #     lst_alarm.append(obj['alarmkey'])
    #     lst_clss.append(obj['clss'])
    #     lst_sum.append(obj['firsteventsummary'])
    #     lst_tt.append(obj['troubleticket'])
    # d = {'Id':lst_id,'Alarmkey':lst_alarm,'Class':lst_clss,'Summary':lst_sum,'Troubleticket':lst_tt}
    # df_csv = pd.DataFrame(d)
    # df_csv.to_excel('output.xlsx', index = False)
# print(t_df)
    # print(dfnew_true)
    dgfgg = pd.concat([a_df,outconcat],axis = 1)
    dgfgg['incident']=dgfgg.apply(lambda x: 'ConvertToTicket' if x['duplicated'] == "No" and x['incident'] == 'ConvertToTicket' else 'AlreadyConvertedToTicket' if x['duplicated'] == "Yes" and x['incident'] == 'ConvertToTicket' else 'FalseAlarm',axis=1).astype(str)
    dgfgg = dgfgg.drop(['unnamed: 0.1','alarmkey','clss','cleartally','highestseverity','mstr','nd','tally','ttt','smry','firsteventsummarycode','tallyequal','tallygreater','tallylesser'],axis = 1)

    
    # for i in range(len(df_csv)):
    #     data={}
    #     data["short_description"]= df_csv['Summary'][i]
    #     data["assignment_group"]='TCS_ST_107'
    #     data["Caller"]='Geeta Naidu'
    #     data["Category"]='Hardware'
    #     data["Subcategory"]=random.choice(['CPU','Disk','Memory','Monitor'])
    #     data["Impact"]='1'
    #     data["Urgency"]='1-High'
    #     data["Priotiry"]='1-Critical'
    #     data["Major Incident"]='Y'
    #     jdat =json.dumps(data)
    #     print(jdat)
        
    #     response = requests.post(url, auth=(user, pwd), headers=headers,data=jdat)

    # if response.status_code != 201: 
    #     print('Status:', response.status_code, 'Headers:', response.headers, 'Error Response:', response.json())
    #     exit()
    # data = response.json()
    
    #print(outconcat)
#     print("outconca shape :",outconcat.shape)
#     print(outconcat['result'].value_counts())
    # dic = dgfgg.to_dict()
    # return dic
    return dgfgg.to_html(header="true", table_id="table")
