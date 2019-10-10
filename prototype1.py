
import pandas as pd
import numpy as np
import urllib
from io import StringIO


def get_cleaned_data(location):
    #link = "https://firebasestorage.googleapis.com/v0/b/warewolves.appspot.com/o/PM_train.csv?alt=media&token=89333a8a-bef3-458d-8269-fe0bfd5572fa"
    link=location
    f = urllib.request.urlopen(link)
    myfile = f.read()
    s=str(myfile,'utf-8')
    data = StringIO(s)
    dataset_train=pd.read_csv(data,sep=' ',header=None).drop([26,27],axis=1)
    col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
    dataset_train.columns=col_names
    dataset_train['ttf'] = dataset_train.groupby(['id'])['cycle'].transform(max)-dataset_train['cycle']
    dataset_train.head()
    df_train=dataset_train.copy()
    period=30
    df_train['label_bc'] = df_train['ttf'].apply(lambda x: 1 if x <= period else 0)
    result=df_train

    return(result)



#location="https://firebasestorage.googleapis.com/v0/b/warewolves.appspot.com/o/PM_train.csv?alt=media&token=89333a8a-bef3-458d-8269-fe0bfd5572fa"
#data=get_cleaned_data(location)
#data.head()


def data_analysis(location,machine_name,sensor_names,cycles):
    df_train=get_cleaned_data(location)

    data_part=df_train[df_train["id"]==machine_name]
    data_part=data_part[(data_part["cycle"]>=cycles[0]) & (data_part["cycle"]<=cycles[1])]

    data_part.set_index('cycle', inplace=True)
    columns_keep=sensor_names
    data_part=data_part[columns_keep]

    #print(data_part.head())
    result=data_part.to_dict()
    #print(result)

    new_dict={}
    for sensor,info in result.items():
        new_dict[sensor]={}
        new_dict[sensor]["Day"]=list(result[sensor].keys())
        new_dict[sensor]["Value"]=list(result[sensor].values())

    #print(new_dict)
    return(new_dict)


#location="https://firebasestorage.googleapis.com/v0/b/warewolves.appspot.com/o/PM_train.csv?alt=media&token=89333a8a-bef3-458d-8269-fe0bfd5572fa"
#machine_name=1
#cycles=[5,15]
#sensor_names=["s1","s2","s3","s4"]
#dict_output=data_analysis(location,machine_name,sensor_names,cycles)
#print(dict_output)


# In[ ]:


#def null_rows():
    #df_train=get_cleaned_data(location)
#df_train.columns[df_train.isnull().any()].tolist()


def no_failues(location):
    df_train=get_cleaned_data(location)
    data_part=df_train

    rul = pd.DataFrame(data_part.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']

    df=rul.groupby("max")["id"].apply(list)
    result=df.to_dict()


    new_dict={}

    for day,info in result.items():
        new_dict[day]={}
        new_dict[day]["Machine"]=info

    max_day=max(result.keys())
    for i in range(max_day):
        if i not in new_dict.keys():
            new_dict[i]={}
            new_dict[i]["Machine"]=[]


    #print(new_dict)
    return(new_dict)



#location="https://firebasestorage.googleapis.com/v0/b/warewolves.appspot.com/o/PM_train.csv?alt=media&token=89333a8a-bef3-458d-8269-fe0bfd5572fa"
#dict_output=no_failues(location)
#print(dict_output)



def correlated_sensors(location):
    df_train=get_cleaned_data(location)

    sensor_columns=df_train.columns[5:-2]
    data_part=df_train[sensor_columns]
    #print(data_part.corrcoef())

    correlated_columns=[]
    sensor_columns=df_train.columns[5:-2]
    for column in sensor_columns:
        corr_mat = np.corrcoef(df_train[column],df_train["ttf"])
        score=abs(round(corr_mat[0,1],2))
        if(score>0.5):
            correlated_columns.append(column)
    return(correlated_columns)


#location="https://firebasestorage.googleapis.com/v0/b/warewolves.appspot.com/o/PM_train.csv?alt=media&token=89333a8a-bef3-458d-8269-fe0bfd5572fa"
#sensors=correlated_sensors(location)



# import pandas as pd
# import numpy as np
# import urllib
# from io import StringIO

# def get_cleaned_data():
# 	link = "https://firebasestorage.googleapis.com/v0/b/warewolves.appspot.com/o/PM_train.csv?alt=media&token=89333a8a-bef3-458d-8269-fe0bfd5572fa"
# 	f = urllib.request.urlopen(link)
# 	myfile = f.read()
# 	s=str(myfile,'utf-8')
# 	data = StringIO(s)
# 	dataset_train=pd.read_csv(data,sep=' ',header=None).drop([26,27],axis=1)



# 	#dataset_train=pd.read_csv("/Users/divalicious/Desktop/PredictiveMaintainence/PM_train.txt",sep=' ',header=None).drop([26,27],axis=1)
# 	col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
# 	dataset_train.columns=col_names
# 	dataset_train['ttf'] = dataset_train.groupby(['id'])['cycle'].transform(max)-dataset_train['cycle']
# 	dataset_train.head()
# 	df_train=dataset_train.copy()
# 	period=30
# 	df_train['label_bc'] = df_train['ttf'].apply(lambda x: 1 if x <= period else 0)
# 	result=dataset_train["id"][0]


# 	return(result)



# re=get_cleaned_data()
# print(type(re))


#




