#!/usr/bin/env python
# coding: utf-8

# In[74]:


pip install seaborn


# In[75]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[76]:


import warnings
warnings.filterwarnings('ignore')


# In[77]:


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[78]:


application_data = pd.read_csv('application_data.csv')
previous_application = pd.read_csv('previous_application.csv')


# In[79]:


application_data.head()


# In[80]:


application_data.info(verbose=True)


# In[81]:


previous_application.head()


# In[82]:


previous_application.info()


# In[83]:


application_data.shape


# In[84]:


previous_application.shape


# In[85]:


# Checking % of missing values

null_data_percentage = application_data.isnull().sum()/len(application_data)*100
null_data_percentage


# In[86]:


#Checking columns whose missing values >40%
major_missing_data_columns = null_data_percentage[null_data_percentage >= 40]
major_missing_data_columns


# In[87]:


#Number of major null values in columns

len(major_missing_data_columns)


# In[88]:


# dropping the major missing values in columns(49 columns) and storing the new values in a different dataframe

application_data1 = application_data.drop(columns=major_missing_data_columns.index)


# In[89]:


#Shape of the new df

application_data1.shape


# In[90]:


# Checking % of missing values again

(application_data1.isnull().sum()/ len(application_data) *100).sort_values(ascending=False)


# In[91]:


#Handling missing values in 'OCCUPATION_TYPE'

#checking 'NAME_INCOME_TYPE' value_counts where 'OCCUPATION_TYPE' has Null value

application_data1[application_data1['OCCUPATION_TYPE'].isnull()]['NAME_INCOME_TYPE'].value_counts()


# In[92]:


application_data1.OCCUPATION_TYPE.value_counts(dropna=False)


# In[93]:


application_data1[application_data1['NAME_INCOME_TYPE']=='Pensioner']['OCCUPATION_TYPE'].value_counts()

# For now we will leave the 'OCCUPAYION_TYPE' with null values 


# In[94]:


#Checking 'EXT_SOURCE_2'

np.round(application_data1['EXT_SOURCE_2'].describe(),4)


# In[95]:


#Checking 'EXT_SOURCE_3'

np.round(application_data1['EXT_SOURCE_3'].describe(),4)


# In[96]:


#EXT_SOURCE_3 has high number of null values so it can be dropped

application_data1.drop(columns=['EXT_SOURCE_3'],inplace=True)


# In[97]:


application_data1['EXT_SOURCE_2'].value_counts()


# In[98]:


# EXT_SOURCE_2 is a continuous variable. So, lets check for any outliers
plt.style.use('ggplot')
plt.figure(figsize=[10,5])
sns.boxplot(application_data1['EXT_SOURCE_2'])
plt.show()


# In[99]:


# Since EXT_SOURCE_2 has no outlier, we can choose mean to impute the column

imputval = round(application_data1['EXT_SOURCE_2'].mean(),2)


# In[100]:


#AMT_REQ_CREDIT_BUREAU refers to this financial company or any other company which may have hit Bureau to check applicant's credit score. This indicates:
#How many places is the applicant looking for loan parallely.
#How many such applications or loans were applied for and/or taken in last one yaer This is an important indicator NaN can either signify connection between our Server and Bureau failed or the applicant has not got any loans

for i in application_data1.columns:
    if i.startswith("AMT_REQ"):
        print(application_data1[i].value_counts())
        print("\n\n")


# In[101]:


for i in application_data1.columns:
    if i.startswith("AMT_REQ"):
        application_data1[i].fillna(value=(application_data1[i].mode()[0]), inplace=True)

application_data1.iloc[:,66:71].isnull().sum()


# In[102]:


#checking 'AMT_GOODS_PRICE'

application_data1[application_data1['AMT_ANNUITY'].isnull()]


# In[103]:


# Since AMT_ANNUITY is a continuous variable. So checking for outliers
sns.boxplot(application_data1['AMT_ANNUITY'])
plt.show()


# In[104]:


avgpercent_AA=application_data1.AMT_ANNUITY.mean()
application_data1['AMT_ANNUITY'].fillna(value=avgpercent_AA, inplace=True)


# In[105]:


#checking 'AMT_GOODS_PRICE'

application_data1[application_data1['AMT_GOODS_PRICE'].isnull()]


# In[106]:


#All the rows with AMT_GOODS_PRICE as null have NAME_CONTRACT_TYPE as 'Revolving loans'
#Revolving loans are generally not for purchasing any partifuclar item. 
#So these null values can be convereted to 0, as there are no good purchased.

application_data1['AMT_GOODS_PRICE'].fillna(value=0, inplace=True)


# In[107]:


# Checking the values present in columns starting with 'DAYS'
for i in application_data1.columns:
    if i.startswith("DAYS"):
        print(application_data1[i].value_counts())
        print("\n\n")


# In[108]:


# The columns starting with 'DAYS'
filter_col = [col for col in application_data1 if col.startswith('DAYS')]
filter_col


# In[109]:


# Applying abs() function to columns starting with 'DAYS' to convert the negative values to positive
application_data1[filter_col]= abs(application_data1[filter_col])


# In[110]:


for i in application_data1.columns:
    if i.startswith("DAYS"):
        print(application_data1[i].value_counts())
        print("\n\n")


# In[111]:


#Checking gender column

application_data1.CODE_GENDER.value_counts()


# In[112]:


# Replacing XNA value with F

application_data1.loc[application_data1.CODE_GENDER == 'XNA','CODE_GENDER'] = 'F'
application_data1.CODE_GENDER.value_counts()


# In[113]:


application_data1.ORGANIZATION_TYPE.value_counts()


# In[114]:


application_data1['ORGANIZATION_TYPE'] = application_data1['ORGANIZATION_TYPE'].replace('XNA',np.NaN)


# In[115]:


application_data1['ORGANIZATION_TYPE'].value_counts(dropna=False)


# In[116]:


application_data1.AMT_INCOME_TOTAL.describe()


# In[117]:


#Binning continous variables
# Binning 'AMT_INCOME_RANGE' based on quantiles

application_data1['AMT_INCOME_RANGE'] = pd.qcut(application_data1.AMT_INCOME_TOTAL, q=[0, 0.2, 0.5, 0.8, 0.95, 1], labels=['VERY_LOW', 'LOW', "MEDIUM", 'HIGH', 'VERY_HIGH'])
application_data1['AMT_INCOME_RANGE'].head(10)


# In[118]:


# Converting 'DAYS_BIRTH' to years
application_data1['DAYS_BIRTH']= (application_data1['DAYS_BIRTH']/365).astype(int)
application_data1.rename({'DAYS_BIRTH':'AGE_IN_YEARS'}, axis=1, inplace=True)


# In[119]:


application_data1['AGE_IN_YEARS'].describe()


# In[247]:


# Binning 'DAYS_BIRTH'
application_data1['DAYS_BIRTH_BINS']=pd.cut(application_data1['AGE_IN_YEARS'],bins=np.arange(20,71,5))


# In[248]:


application_data1['DAYS_BIRTH_BINS'].value_counts(dropna=False)


# In[144]:


## Adding one more column that will be used for analysis later
application_data1['CREDIT_INCOME_RATIO']=round((application_data1['AMT_CREDIT']/application_data1['AMT_INCOME_TOTAL']))


# In[184]:


#Getting the percentage of social circle who defaulted
application_data1['SOCIAL_CIRCLE_30_DAYS_DEF_PERC']=application_data1['DEF_30_CNT_SOCIAL_CIRCLE']/application_data1['OBS_30_CNT_SOCIAL_CIRCLE']
application_data1['SOCIAL_CIRCLE_60_DAYS_DEF_PERC']=application_data1['DEF_60_CNT_SOCIAL_CIRCLE']/application_data1['OBS_60_CNT_SOCIAL_CIRCLE']


# In[122]:


application_data1.info()


# In[123]:


#Checking again which all columns  have Null values

application_data1.columns[application_data1.isnull().any()].tolist()

#We can leave these columns as is, as they won't necessarily be needed for our analysis. We an furthur impute them if required.


# In[124]:


#Checking TARGET columns for insights

application_data1.TARGET.value_counts(normalize=True).plot.bar(color='blue', align='center', edgecolor='red')
plt.title("Payment Difficulty", fontdict={'fontweight':10,'fontsize':20})
plt.xlabel("\n 0 - No Payment Difficulty   |   1 - Difficulty")
plt.xticks(rotation = 0)
plt.ylabel("Normalised Values")
plt.show()

#Here we can see that no. of applicants with payment difficulty are way less compared to the applicants with no payment difficulty.


# In[125]:


#checking exact Target 0 to Target 1 ratio

application_data1[application_data1.TARGET==0].shape[0]/application_data1[application_data1.TARGET==1].shape[0]

#From this we can infer that 1 in every 11 applicant has a payment difficulty


# In[126]:


#FLAG_OWN_REALTY' needs to be changed to Binary from yes/no for easier analysis

application_data1['OWN_CAR_flag']=np.where(application_data1.FLAG_OWN_CAR =="Y",1,0)
application_data1['OWN_REALTY_flag']= np.where(application_data1.FLAG_OWN_REALTY =="Y",1,0)


# In[127]:


application_data1['OWN_CAR_flag'].value_counts()


# In[128]:


application_data1['OWN_REALTY_flag'].value_counts()


# In[129]:


application_data1.columns[application_data1.dtypes=="int64"].tolist()


# In[ ]:





# In[130]:


# Since AMT_ANNUITY is a continuous variable. So checking for outliers
sns.boxplot(application_data1['AMT_ANNUITY'])
plt.show()


# In[131]:


imputVAL = round(application_data1['AMT_ANNUITY'].median(),2)
print(f'Since AMT_ANNUITY has outliers, the column can be imputed using the median of the coumn i.e. {imputVAL}')


# In[132]:


# Since AMT_ANNUITY is a continuous variable. So checking for outliers
sns.boxplot(application_data1['AMT_ANNUITY'])
plt.show()


# In[133]:


# Since this is count of family members, this is a continuous variable and we can impute the mean/median
sns.boxplot(application_data1['CNT_FAM_MEMBERS'])
plt.show()


# Maxium number of family members is 20.
# There are some ouliers but we don't have to impute them as it is completely possible that there are 20 family members.

# In[134]:


# AMT_GOODS_PRICE is a continuous variable. So checking for outliers
sns.boxplot(application_data1['AMT_GOODS_PRICE'])
plt.show()


# In[135]:


# Checking all columns with object type data

application_data1.columns[application_data1.dtypes=="object"].tolist()


# In[136]:


#Checking all object types Variables and their values

for i in application_data1.columns:
    if application_data1[i].dtypes=="object":
        print(application_data1[i].value_counts(normalize=True, dropna= False))
        plt.figure(figsize=[6,6])
        application_data1[i].value_counts(normalize=True, dropna=False).plot.pie(labeldistance=None)
        plt.legend()
        plt.show()
      


# In[137]:


#  Object Data
a = 4  # number of rows
b = 3  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(20,12))

for i in application_data1.columns:
    if application_data1[i].dtypes=="object":

        plt.subplot(a, b, c)
        plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))
        plt.xlabel(i)
        sns.countplot(application_data1[i],palette="Blues_d")
        c = c + 1

fig.tight_layout()
plt.show()


# Some insights from the above charts:
# 
# 1. Cash loans offered are more than revolving loans, at 90%
# 2. 65% Females have taken loans in comparison to 34% male. This is an interesting insight and needs to be studied further.
# 3. 65% applicant dont own cars.
# 4. 69% applicants own living quarters.
# 5. 81% applicants came accompanied for loan application (This might not have any effect on our analysis).
# 6. While most applicants are working class, 18% are pensioners.
# 7. 71% have secondary education and 24% have higher education.
# 8. More than 63% appicants are married.
# 9. 31% have not mentioned their occupation type.

# In[138]:


#Checking the float type columns
application_data1.select_dtypes(include='float64').columns


# In[139]:


#Converting these count columns to int64
ColumnToConvert = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
                   'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
                   'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                   'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
application_data1.loc[:,ColumnToConvert]=application_data1.loc[:,ColumnToConvert].apply(lambda col: col.astype('int',errors='ignore'))


# In[140]:


#Checking the object type columns
ColumnToConvert = list(application_data1.select_dtypes(include='object').columns)


# In[141]:


application_data1.loc[:,ColumnToConvert]=application_data1.loc[:,ColumnToConvert].apply(lambda col: col.astype('str',errors='ignore'))


# In[249]:


#From the remaining columns about 30 are selected based on their description and relevance with problem statement for further analysis
FinalColumns = ['SK_ID_CURR','TARGET','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','AMT_INCOME_RANGE','DAYS_BIRTH_BINS','AMT_CREDIT','AMT_INCOME_TOTAL',
'CREDIT_INCOME_RATIO','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','DAYS_EMPLOYED',
'DAYS_REGISTRATION','FLAG_EMAIL','OCCUPATION_TYPE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT_W_CITY','ORGANIZATION_TYPE',
'SOCIAL_CIRCLE_30_DAYS_DEF_PERC','SOCIAL_CIRCLE_60_DAYS_DEF_PERC','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_MON',
'AMT_REQ_CREDIT_BUREAU_QRT','NAME_CONTRACT_TYPE','AMT_ANNUITY','REGION_RATING_CLIENT','AMT_GOODS_PRICE']


# In[250]:


application_data1_final = application_data1[FinalColumns]


# In[251]:


newapp_0=application_data1_final[application_data1_final.TARGET==0]    # Dataframe with all the data related to non-defaulters
newapp_1=application_data1_final[application_data1_final.TARGET==1]    # Dataframe with all the data related to defaulters


# #   Univariate Categorical Ordered Analysis 

# In[194]:


# function to count plot for categorical variables
def plotuninewapp(var):

    plt.style.use('ggplot')
    sns.despine
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,6))
    
    sns.countplot(x=var, data=newapp_0,ax=ax1)
    ax1.set_ylabel('Total Counts')
    ax1.set_title(f'Distribution of {var} for Non-Defaulters',fontsize=15)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
    
    # Adding the normalized percentage for easier comparision between defaulter and non-defaulter
    for p in ax1.patches:
        ax1.annotate('{:.1f}%'.format((p.get_height()/len(newapp_0))*100), (p.get_x()+0.1, p.get_height()+50))
        
    sns.countplot(x=var, data=newapp_1,ax=ax2)
    ax2.set_ylabel('Total Counts')
    ax2.set_title(f'Distribution of {var} for Defaulters',fontsize=15)    
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")
    
    # Adding the normalized percentage for easier comparision between defaulter and non-defaulter
    for p in ax2.patches:
        ax2.annotate('{:.1f}%'.format((p.get_height()/len(newapp_1))*100), (p.get_x()+0.1, p.get_height()+50))
    
    plt.show()
    


# In[195]:


plotuninewapp('CODE_GENDER')


# > We can see that females contribute almost 67% to the non-defaulters and 57% to the defaulters.
# > <br> **But the rate of defaulting of FEMALE is much lower when compared to their MALE counterparts.**

# In[196]:


plotuninewapp('FLAG_OWN_CAR')


# > We can see that people without cars contribute 65.7% to the non-defaulters while 69.5% to the defaulters.

# In[197]:


plotuninewapp('NAME_INCOME_TYPE')


# > We can see that the students don't default. The reason could be they are not required to pay during the time they are students.
# <br> We can also see that the Businessmen never default.
# <br>Most of the loans are distributed to working class people<br>We also see that working class people contribute almost  51% to non defaulters while they contribute to 61% of the defaulters. Clearly, the chances of defaulting are more in their case.

# In[198]:


plotuninewapp('NAME_FAMILY_STATUS')


# > Married people tend to apply for loans more than others in the group. 
# > <br> But from the graph we see that Single/non Married people contribute 14.5% to Non Defaulters and 18% to the defaulters. So there might be more risk associated with them too.

# In[199]:


plotuninewapp('NAME_HOUSING_TYPE')


# > It is clear from the graph that people who have House/Apartment, tend to apply for loans more than others.
# > <br>People living with parents tend to default more often when compared with others.

# In[252]:


plotuninewapp('DAYS_BIRTH_BINS')


# > We see that [25,30] age group tend to default more often. So they are the riskiest people to loan to.
# > <br> With increasing age group, people tend to default less starting from the age 25.

# In[218]:


plotuninewapp('AMT_INCOME_RANGE')


# > The Very High income group tend to default less often. They contribute 12.4% to the total number of defaulters, while they contribute 15.6% to the Non-Defaulters, which also tells us that there are less number of very high income range people who apply for loans.

# In[203]:


plotuninewapp('NAME_EDUCATION_TYPE')


# > People with Secondary education tend to apply for loans more than others.
# > People with Secondary education and higher education tend to default more.

# In[204]:


plotuninewapp('REGION_RATING_CLIENT')


# # Univariate continuous variable analysis

# In[205]:


# function to dist plot for continuous variables
def plotunidist(var):

    plt.style.use('ggplot')
    sns.despine
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    
    sns.distplot(a=newapp_0[var],ax=ax1)

    ax1.set_title(f'Distribution of {var} for Non-Defaulters',fontsize=15)
            
    sns.distplot(a=newapp_1[var],ax=ax2)
    ax2.set_title(f'Distribution of {var} for Defaulters',fontsize=15)    
        
    plt.show()


# In[206]:


plotunidist('CREDIT_INCOME_RATIO')


# > Credit income ratio the ratio of AMT_CREDIT/AMT_INCOME_TOTAL. 
# >Although there doesn't seem to be a clear distiguish between the group which defaulted vs the group which didn't when compared using the ratio, we can see that when the CREDIT_INCOME_RATIO is more than 50, people default.

# In[208]:


plotunidist('DAYS_EMPLOYED')


# In[207]:


newapp_1['CNT_FAM_MEMBERS'].value_counts()


# In[219]:


plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
newapp_0['CNT_FAM_MEMBERS'].plot.hist(bins=range(15))
plt.title('Distribution of CNT_FAM_MEMBERS for Non-Defaulters',fontsize=15)
plt.xlabel('CNT_FAM_MEMBERS')
plt.ylabel('LOAN APPLICATION COUNT')

plt.subplot(1, 2, 2)
newapp_1['CNT_FAM_MEMBERS'].plot.hist(bins=range(15))
plt.title(f'Distribution of CNT_FAM_MEMBERS for Defaulters',fontsize=15)
plt.xlabel('CNT_FAM_MEMBERS')
plt.ylabel('LOAN APPLICATION COUNT')  

plt.show()


# > We can see that a family of 3 applies loan more often than the other families

# In[220]:


#Getting the top 10 correlation in newapp_0
corr=newapp_0.corr()
corr_df = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool)).unstack().reset_index()
corr_df.columns=['Column1','Column2','Correlation']
corr_df.dropna(subset=['Correlation'],inplace=True)
corr_df['Abs_Correlation']=corr_df['Correlation'].abs()
corr_df = corr_df.sort_values(by=['Abs_Correlation'], ascending=False)
corr_df.head(10)


# In[221]:


#Getting the top 10 correlation newapp_1
corr=newapp_1.corr()
corr_df = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool)).unstack().reset_index()
corr_df.columns=['Column1','Column2','Correlation']
corr_df.dropna(subset=['Correlation'],inplace=True)
corr_df['Abs_Correlation']=corr_df['Correlation'].abs()
corr_df = corr_df.sort_values(by=['Abs_Correlation'], ascending=False)
corr_df.head(10)


# #  Bivariate Analysis of numerical variables

# In[222]:


# function for scatter plot for continuous variables
def plotbivar(var1,var2):

    plt.style.use('ggplot')
    sns.despine
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,6))
    
    sns.scatterplot(x=var1, y=var2,data=newapp_0,ax=ax1)
    ax1.set_xlabel(var1)    
    ax1.set_ylabel(var2)
    ax1.set_title(f'{var1} vs {var2} for Non-Defaulters',fontsize=15)
    
    sns.scatterplot(x=var1, y=var2,data=newapp_1,ax=ax2)
    ax2.set_xlabel(var1)    
    ax2.set_ylabel(var2)
    ax2.set_title(f'{var1} vs {var2} for Defaulters',fontsize=15)
            
    plt.show()


# In[223]:


plotbivar('AMT_CREDIT','CNT_FAM_MEMBERS')


# > We can see that the density in the lower left corner is similar in both the case, so the people are equally likely to default if the family is small and the AMT_CREDIT is low.
# > We can observe that larger families and people with larger AMT_CREDIT default less often

# In[224]:


plotbivar('AMT_GOODS_PRICE','AMT_CREDIT')


# ## Data Analysis For Previous Application Data

# ###  Doing some more routine check

# In[225]:


previous_application.head(10)


# In[228]:


# Removing all the columns with more than 50% of null values
previous_application = previous_application.loc[:,previous_application.isnull().mean()<=0.5]
previous_application.shape


# In[229]:


# function to count plot for categorical variables
def plot_uni(var):

    plt.style.use('ggplot')
    sns.despine
    fig,ax = plt.subplots(1,1,figsize=(15,5))
    
    sns.countplot(x=var, data=previous_application,ax=ax,hue='NAME_CONTRACT_STATUS')
    ax.set_ylabel('Total Counts')
    ax.set_title(f'Distribution of {var}',fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    plt.show()


# In[230]:


plot_uni('NAME_CONTRACT_TYPE')


# > From the above chart, we can infer that, most of the applications are for 'Cash loan' and 'Consumer loan'. Although the cash loans are refused more often than others.

# In[231]:


plot_uni('NAME_PAYMENT_TYPE')


# >From the above chart, we can infer that most of the clients chose to repay the loan using the 'Cash through the bank' option <br> We can also see that 'Non-Cash from your account' & 'Cashless from the account of the employee' options are not at all popular in terms of loan repayment amongst the customers.

# In[232]:


plot_uni('NAME_CLIENT_TYPE')


# >Most of the loan applications are from repeat customers, out of the total applications 70% of customers are repeaters.

# ###  Checking the correlation in the PreviousApplication dataset

# In[233]:


#Getting the top 10 correlation PreviousApplication
corr=previous_application.corr()
corr_df = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool)).unstack().reset_index()
corr_df.columns=['Column1','Column2','Correlation']
corr_df.dropna(subset=['Correlation'],inplace=True)
corr_df['Abs_Correlation']=corr_df['Correlation'].abs()
corr_df = corr_df.sort_values(by=['Abs_Correlation'], ascending=False)
corr_df.head(10)


# ### Using pairplot to perform bivariate analysis on numerical columns 

# In[234]:


#plotting the relation between correlated highly corelated numeric vriables
plt.figure(figsize=[20,8])
sns.pairplot(previous_application[['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_GOODS_PRICE','NAME_CONTRACT_STATUS']], 
             diag_kind = 'kde', 
             plot_kws = {'alpha': 0.4, 's': 80, 'edgecolor': 'k'},
             size = 4)
plt.show()


# > 1. Annuity of previous application has a very high and positive influence over: (Increase of annuity increases below factors) <br>(1) How much credit did client asked on the previous application <br> (2)Final credit amount on the previous application that was approved by the bank <br>(3) Goods price of good that client asked for on the previous application.<br><br>
# >2. For how much credit did client ask on the previous application is highly influenced by the Goods price of good that client has asked for on the previous application<br><br>
# >3. Final credit amount disbursed to the customer previously, after approval is highly influence by the application amount and also the goods price of good that client asked for on the previous application.

# ### Using box plot to do some more bivariate analysis on categorical vs numeric columns

# In[235]:


#by variant analysis function
def plot_by_cat_num(cat, num):

    plt.style.use('ggplot')
    sns.despine
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    
    sns.boxenplot(x=cat,y = num, data=previous_application)
    ax.set_ylabel(f'{num}')
    ax.set_xlabel(f'{cat}')

    ax.set_title(f'{cat} Vs {num}',fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
     
    plt.show()


# In[236]:


#by-varient analysis of Contract status and Annuity of previous appliction
plot_by_cat_num('NAME_CONTRACT_STATUS', 'AMT_ANNUITY')


# > From the above plot we can see that loan application for people with lower AMT_ANNUITY gets canceled or Unused most of the time.<br> We also see that applications with too high AMT ANNUITY also got refused more often than others.

# In[237]:


#by-varient analysis of Contract status and Final credit amount disbursed to the customer previously, after approval
plot_by_cat_num('NAME_CONTRACT_STATUS', 'AMT_CREDIT')


# >We can infer that when the AMT_CREDIT is too low, it get's cancelled/unused most of the time.

# ## Merging the files and analyzing the data

# In[238]:


## Merging the two files to do some analysis
NewLeftPrev = pd.merge(application_data1_final, previous_application, how='left', on=['SK_ID_CURR'])


# In[239]:


NewLeftPrev.shape


# In[240]:


NewLeftPrev.info()


# In[241]:


def plotuni_combined(Varx,Vary):
    # 100% bar chart
    plt.style.use('ggplot')
    sns.despine
    NewDat = NewLeftPrev.pivot_table(values='SK_ID_CURR', index=Varx,columns=Vary,aggfunc='count')
    NewDat=NewDat.div(NewDat.sum(axis=1),axis='rows')*100
    sns.set()
    NewDat.plot(kind='bar',stacked=True,figsize=(15,5))
    plt.title(f'Effect Of {Varx} on Loan Approval')
    plt.xlabel(f'{Varx}')
    plt.ylabel(f'{Vary}%')
    plt.show()


# > We see that car ownership doesn't have any effect on application approval or rejection. But we saw earlier that the people who has a car has lesser chances of default. The bank can add more weightage to car ownership while approving a loan amount

# In[242]:


plotuni_combined('CODE_GENDER','NAME_CONTRACT_STATUS')


# > We see that code gender doesn't have any effect on application approval or rejection. 
# > <br>But we saw earlier that female have lesser chances of default compared to males. The bank can add more weightage to female while approving a loan amount.

# In[243]:


plotuni_combined('TARGET','NAME_CONTRACT_STATUS')


# >We can see that the people who were approved for a loan earlier, defaulted less often where as people who were refused a loan earlier have higher chances of defaulting. 

# #### <font color = blue> **Default cases in Approved Applications"</font>
#     All the below variables were established in analysis of Application dataframe as leading to default.
#     
#   **Default High**<br>
#         'INCOME_GROUP' - Medium income<br> 
#         'AGE_GROUP - 25-35, followed by 35-45<br>
#         'NAME_INCOME_TYPE' - Working <br>
#         'OCCUPATION_TYPE' - Labourers 31%<br>
#         'ORGANIZATION_TYPE' - Business type 3<br>
#         'OWN_CAR_flag' - 31% dont have car<br>
#         'OWN_REALTY_flag' - 70% dont have own home

# # Case Summary
#    
# ### *Defaulters' demography*
#      All the below variables were established in analysis of Application dataframe as leading to default. 
#     Checked these against the Approved loans which have defaults, and it proves to be correct
#         -Medium income
#         -25-35 years ols , followed by 35-45 years age group
#         -Male
#         -Unemployed
#         -Labourers, Salesman, Drivers
#         -Business type 3
#         -Own House - No
#     Other IMPORTANT Factors to be considered
#         -Days last phone number changed - Lower figure points at concern
#         -No of Bureau Hits in last week. Month etc – zero hits is good
#         -Amount income not correspondingly equivalent to Good Bought – Income low and good value high is a concern
#         -Previous applications with Refused, Cancelled, Unused loans also have default which is a matter of concern.   This indicates that the financial company had Refused/Cancelled previous application but has approved the current and is  facing default on these. 
#         
# ### *Credible Applications refused*
#     -Unused applications have lower loan amount. Is this the reason for no usage?
#     -Female applicants should be given extra weightage as defaults are lesser.
#     -60% of defaulters are Working applicants. This does not mean working applicants must be refused. Proper scrutiny of other parameters needed
#     -Previous applications with Refused, Cancelled,Unused loans also have cases where payments are coming on time in current application. This indicates that possibly wrong decisions were done in those cases.

# 

# In[ ]:




