#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pyforest as pf


# In[3]:


df=pd.read_csv(r"C:\Users\akash\OneDrive\Documents\dataset\apple_quality.csv.xls")


# In[4]:


# pf.lazy_imports()


# In[5]:


df=pd.read_csv(r'https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv')


# In[6]:


df


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.columns


# In[10]:


df['job'].unique()


# In[11]:


df['marital'].unique()


# In[12]:


df['education'].unique()


# In[13]:


df['default'].unique()


# In[13]:


df['housing'].unique()


# In[14]:


df['loan'].unique()


# In[15]:


df['contact'].unique()


# In[16]:


df['month'].unique()


# In[17]:


df['poutcome'].unique()


# In[18]:


# education
df['education']=df['education'].replace(['basic.4y','basic.6y','basic.9y'],'Basic')


# In[19]:


df['education'].unique()


# In[14]:


df['marital'].value_counts()


# - Most of the people are Married

# In[15]:


df['marital'].value_counts().plot.bar(color='green')
plt.title('Marital status')


# In[16]:


df['job'].value_counts()


# In[17]:


df['age'].plot.hist()


# In[18]:


df['age'].median()


# - most of people are in 38 age group

# In[19]:


df['education'].value_counts()


# In[20]:


df.groupby('education')['age'].agg(['median','max','min'])


# - Most of the people have Basic education

# In[21]:


df['loan'].value_counts()


# In[22]:


df['loan'].value_counts().plot.pie()


# - Higher number of people have no loan

# In[23]:


df['poutcome'].value_counts()


# In[24]:


df['contact'].value_counts()


# In[25]:


df['contact'].value_counts().plot.barh(figsize=(6,1))


# In[26]:


df['y'].value_counts()


# In[27]:


sns.countplot(x='y',data=df,palette='hls')
plt.show()


# In[28]:


count_no_sub =len(df[df['y']==0])
count_sub =len(df[df['y']==1])
count_sub_NoSub = len(df['y'])
prcnt_no_sub = (count_no_sub/count_sub_NoSub)*100
prcnt_sub = (count_sub/count_sub_NoSub)*100
print(f'percentage of subscription {prcnt_sub} %')
print(f'percentage of no subscription {prcnt_no_sub} %')


# The ratio of no-subscription to subscription instances is 89:11

# In[29]:


df.groupby('education')[['age','duration','campaign','pdays','y']].mean()


# In[30]:


df.groupby('job')[['age','duration','campaign','pdays','y']].mean()


# In[31]:


df.groupby('marital')[['age','duration','campaign','pdays','y']].mean()


# In[32]:


pd.crosstab(df.job,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')


# The frequency of purchase of the deposit depends a great deal on the job title. Thus, the job title can be a good predictor of the outcome variable.

# In[33]:


table=pd.crosstab(df.marital,df.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')


# The marital status does not seem a strong predictor for the outcome variable.

# In[34]:


table=pd.crosstab(df.education,df.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')


# Education seems a good predictor of the outcome variable.

# In[35]:


pd.crosstab(df.day_of_week,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')


# Day of week may not be a good predictor of the outcome.

# In[36]:


pd.crosstab(df.month,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')


# Month might be a good predictor of the outcome variable.

# In[37]:


pd.crosstab(df.poutcome,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')


# Poutcome seems to be a good predictor of the outcome variable

# In[79]:


df.head()


# In[80]:


df['job'].unique()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()


# In[81]:


df_fuel_enc = ohe.fit_transform(df[['job']])
df_fuel_enc.toarray()


# In[83]:


df_fuel_enc = pd.DataFrame(df_fuel_enc.toarray(),columns= ohe.get_feature_names_out(['job']))
df_fuel_enc.head()


# In[85]:


df = pd.concat([df,df_fuel_enc],axis=1)
df.head()


# In[86]:


df['education'].unique()


# In[106]:


df_edu_enc = ohe.fit_transform(df[['education']])
df_edu_enc.toarray()


# In[107]:


df_edu_enc = pd.DataFrame(df_edu_enc.toarray(),columns= ohe.get_feature_names_out(['education']))
df_edu_enc.head()


# In[176]:


df = pd.concat([df,df_edu_enc],axis=1)


# In[ ]:


df.drop(columns=['education','job'],inplace=True)


# In[102]:


df['pdays'].unique()


# In[103]:


df['pdays']=df['pdays'].replace([999],-1)


# In[104]:


df['pdays'].unique()


# In[114]:


from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()


# In[115]:


df['pdays'] = Scaler.fit_transform(df[['pdays']])


# In[112]:


df['marital'].unique()


# In[118]:


df_mar_enc = ohe.fit_transform(df[['marital']])
df_mar_enc.toarray()


# In[119]:


df_mar_enc = pd.DataFrame(df_mar_enc.toarray(),columns= ohe.get_feature_names_out(['marital']))
df_mar_enc.head()


# In[134]:


df = pd.concat([df,df_mar_enc],axis=1)
df.head()


# In[120]:


df['default'].unique()


# In[121]:


df_def_enc = ohe.fit_transform(df[['default']])
df_def_enc.toarray()


# In[ ]:


df_def_enc = pd.DataFrame(df_def_enc.toarray(),columns= ohe.get_feature_names_out(['default']))
df_def_enc.head()


# In[149]:


df = pd.concat([df,df_def_enc],axis=1)


# In[123]:


df['housing'].unique()


# In[124]:


df_hou_enc = ohe.fit_transform(df[['housing']])
df_hou_enc.toarray()


# In[125]:


df_hou_enc = pd.DataFrame(df_hou_enc.toarray(),columns= ohe.get_feature_names_out(['housing']))
df_hou_enc.head()


# In[150]:


df = pd.concat([df,df_hou_enc],axis=1)


# In[126]:


df.drop(columns=['housing','default','marital'],inplace=True)


# In[127]:


df['loan'].unique()


# In[128]:


df_loan_enc = ohe.fit_transform(df[['loan']])
df_loan_enc.toarray()


# In[130]:


df_loan_enc = pd.DataFrame(df_loan_enc.toarray(),columns= ohe.get_feature_names_out(['loan']))
df_loan_enc.head()


# In[151]:


df = pd.concat([df,df_loan_enc],axis=1)


# In[153]:


df['contact'].unique()


# In[154]:


df_contact_enc = ohe.fit_transform(df[['contact']])
df_contact_enc.toarray()


# In[155]:


df_contact_enc = pd.DataFrame(df_contact_enc.toarray(),columns= ohe.get_feature_names_out(['contact']))
df_contact_enc.head()


# In[156]:


df = pd.concat([df,df_contact_enc],axis=1)


# In[157]:


df['month'].unique()


# In[158]:


df_month_enc = ohe.fit_transform(df[['month']])
df_month_enc.toarray()


# In[159]:


df_month_enc = pd.DataFrame(df_month_enc.toarray(),columns= ohe.get_feature_names_out(['month']))
df_month_enc.head()


# In[160]:


df = pd.concat([df,df_month_enc],axis=1)


# In[162]:


df['day_of_week'].unique()


# In[163]:


df_day_enc = ohe.fit_transform(df[['day_of_week']])
df_day_enc.toarray()


# In[164]:


df_day_enc = pd.DataFrame(df_day_enc.toarray(),columns= ohe.get_feature_names_out(['day_of_week']))
df_day_enc.head()


# In[165]:


df = pd.concat([df,df_day_enc],axis=1)


# In[166]:


df.drop(columns=['day_of_week','contact','month'],inplace=True)


# In[167]:


df.info()


# In[168]:


df['poutcome'].unique()


# In[169]:


df_poutcome_enc = ohe.fit_transform(df[['poutcome']])
df_poutcome_enc.toarray()


# In[170]:


df_poutcome_enc = pd.DataFrame(df_poutcome_enc.toarray(),columns= ohe.get_feature_names_out(['poutcome']))
df_poutcome_enc.head()


# In[171]:


df = pd.concat([df,df_poutcome_enc],axis=1)


# In[173]:


df.drop(columns=['loan','poutcome'],inplace=True)


# In[174]:


df


# In[187]:


df['age'] = Scaler.fit_transform(df[['age']])
df['duration'] = Scaler.fit_transform(df[['duration']])
df['campaign'] = Scaler.fit_transform(df[['campaign']])
df['cons_price_idx'] = Scaler.fit_transform(df[['cons_price_idx']])
df['cons_conf_idx'] = Scaler.fit_transform(df[['cons_conf_idx']])
df['euribor3m'] = Scaler.fit_transform(df[['euribor3m']])
df['nr_employed'] = Scaler.fit_transform(df[['nr_employed']])
df['previous'] = Scaler.fit_transform(df[['previous']])
df['emp_var_rate'] = Scaler.fit_transform(df[['emp_var_rate']])


# In[188]:


df


# In[193]:


df = df.rename(columns= {'y':'subscription'})


# In[189]:


from sklearn.model_selection import train_test_split


# In[195]:


X = df.drop('subscription',axis=1)
y = df['subscription']


# In[197]:


X.head()


# In[200]:


y


# In[201]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=7)


# In[202]:


X_train


# In[203]:


y_train


# In[204]:


X_test


# In[205]:


y_test


# In[206]:


from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings('ignore')


log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)


# In[207]:


y_pred = log_reg.predict(X_test)


# In[226]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,f1_score,auc,recall_score,roc_auc_score,roc_curve


# In[211]:


Accuracy = accuracy_score(y_test,y_pred)
Accuracy


# In[212]:


precision = precision_score(y_test,y_pred)
precision


# In[213]:


recall = recall_score(y_test, y_pred)
recall


# In[215]:


f1 = f1_score(y_test, y_pred)
f1


# In[220]:


conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix


# In[224]:


y_prob = log_reg.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
roc_auc


# In[235]:


fpr, tpr,roc_auc = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()


# - Accuracy: 0.9109007040543822
# - Precision: 0.6608084358523726
# - Recall: 0.4100327153762268
# - F1 Score: 0.506056527590848
# - ROC AUC: [       inf 0.99999927 0.99998552 ... 0.00757455 0.00757356 0.00107693]
# - Confusion Matrix: 
# [[7128  193]
#  [ 541  376]]

# *The model achieves a high accuracy of 91%, but the precision (66%) and recall (41%) indicate that while it correctly identifies many true positive instances.*

# In[ ]:
numerical_features = [
    'age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate',
    'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed'
]

# Fit the scaler on the training dataset

# scaler = StandardScaler()
Scaler.fit(df[numerical_features])

import pickle

import joblib

# Save the model
joblib.dump(log_reg, 'logistic_regression_model.pkl')

# Save the scaler
joblib.dump(Scaler, 'scaler.pkl')

# Save the encoder
joblib.dump(ohe, 'encoder.pkl')






# # Save the model
# with open('logistic_regression_model.pkl', 'wb') as model_file:
#     pickle.dump(log_reg, model_file)
#
# # Save the scaler
# with open('scaler.pkl', 'wb') as scaler_file:
#     pickle.dump(Scaler, scaler_file)



categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome']
numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate',
                      'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']

from sklearn.preprocessing import OneHotEncoder, StandardScaler
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# encoder.fit(df[categorical_features])

df_encoded = pd.DataFrame(encoder.transform(df[categorical_features]),
                          columns=encoder.get_feature_names_out(categorical_features))

Scaler.fit(df[numerical_features])

# Scale the numerical features
df_scaled = pd.DataFrame(Scaler.transform(df[numerical_features]), columns=numerical_features)

# Combine scaled numerical and encoded categorical features
df_final = pd.concat([df_scaled, df_encoded], axis=1)



joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(Scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')