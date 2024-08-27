from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib


def feature_engineer(data):
   
   seed=12345
   data_model = data.drop(['customer_id','row_number','surname'],axis=1)
   df_train,df_test = train_test_split(data_model,random_state=seed,test_size=0.2)
   
   df_test.to_csv('./files/datasets/intermediate/test.csv',index=False)
   #Separamos los datos 
   features= df_train.drop(['exited'],axis=1)
   target=df_train['exited']

   features_train, features_valid, target_train, target_valid=train_test_split(
      features,target,random_state=seed,test_size=0.25)
   
   # Codificamos las variables categoricas
   
   one_hot=OneHotEncoder(drop='first')
   one_hot.fit(features_train[['geography','gender']])
   joblib.dump(one_hot, './files/modeling_output/model_fit/one_hot_encoder.joblib')
   features_train[one_hot.get_feature_names_out()]=one_hot.transform(features_train[['geography','gender']]).todense()
   features_valid[one_hot.get_feature_names_out()]=one_hot.transform(features_valid[['geography','gender']]).todense()
   
   features_train=features_train.drop(['geography','gender'],axis=1)
   features_valid=features_valid.drop(['geography','gender'],axis=1)
   
   return features_train,features_valid,target_train,target_valid 