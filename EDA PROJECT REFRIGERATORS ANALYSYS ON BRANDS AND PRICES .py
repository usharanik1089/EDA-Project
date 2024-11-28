#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests


# In[4]:


url="https://www.flipkart.com/search?q=refrigerator&as=on&as-show=on&otracker=AS_Query_OrganicAutoSuggest_6_3_na_na_na&otracker1=AS_Query_OrganicAutoSuggest_6_3_na_na_na&as-pos=6&as-type=RECENT&suggestionId=refrigerator&requestId=97b3fe07-a0ff-4ccc-8853-1ff3476a916e&as-searchtext=refrigerator&p%5B%5D=facets.price_range.from%3D15000&p%5B%5D=facets.price_range.to%3DMax"


# In[5]:


requests.get(url) #finding the website is scrapable or not


# In[6]:


#Header for 500

request_header = {'Content-Type': 'text/html; charset=UTF-8','User-Agent': 'Chrome/101.0.0.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0','Accept-Encoding': 'gzip, deflate, br'}

page = requests.get(url,headers =request_header)


# In[7]:


page


# In[8]:


page.text


# In[9]:


from bs4 import BeautifulSoup


# In[10]:


soup=BeautifulSoup(page.text)


# In[11]:


soup


# In[12]:


soup.find("div",class_ = "KzDlHZ")


# In[13]:


soup.find("div",class_ = "KzDlHZ").text


# In[14]:


for i in soup.find_all("div",class_="KzDlHZ"):
    print(i.text)


# In[15]:


title=[]
for i in soup.find_all("div",class_="KzDlHZ"):
    title.append(i.text)


# In[16]:


title


# In[17]:


len(title)


# In[18]:


#rating
rating=[]
for i in soup.find_all("div",class_="XQDdHH"):
    rating.append(i.text)


# In[19]:


rating


# In[20]:


len(rating)


# In[21]:


for i in soup.find_all("div",class_="tUxRFH"):
    print(i.text)
    print("******"*50)


# In[22]:


import numpy as np


# In[23]:


rating=[]
for i in soup.find_all("div",class_="tUxRFH"):
    r = i.find("div",class_="XQDdHH")
    if r:
        rating.append(r.text)
    else:
        rating.append(np.nan)


# In[24]:


rating


# In[25]:


len(rating)


# In[26]:


import re


# In[27]:


container=soup.find_all("div",class_="tUxRFH")
Brand=[]
Storage=[]
Stars=[]
Doors=[]
Coolingsystem=[]
Rating=[]

for i in container:
    t = i.find("div",class_="KzDlHZ") # t is title of mobiles and class_="KzDlHZ" is tag for title
    text=t.text
    
    
    brand=re.findall("^\w+",text)
    if brand:
        Brand.append(brand[0])
    else:
        Brand.append(np.nan)
        
    storage=re.findall("(\d+\sL)",text)
    if storage:
        Storage.append(storage[0])
    else:
        Storage.append(np.nan)
    
    stars=re.findall(" Door (\d Star) ",text)
    if stars:
        Stars.append(stars[0])
    else:
        Stars.append(np.nan)
        
    doors=re.findall("(Double|Multi|Single|Side by Side)",text)
    if doors:
        Doors.append(doors[0])
    else:
        Doors.append(np.nan)
        
    coolingsystem=re.findall("(Frost Free|Direct Cool)",text)
    if coolingsystem:
        Coolingsystem.append(coolingsystem[0])
    else:
        Coolingsystem.append(np.nan)
    rating = i.find("div",class_="XQDdHH")                                 
    if rating:                                                             
        Rating.append(rating.text)
    else:
        Rating.append(np.nan)
            



# In[28]:


Brand


# In[29]:


len(Brand)


# In[30]:


Storage


# In[31]:


len(Storage)


# In[32]:


Stars


# In[33]:


len(Stars)


# In[34]:


Doors


# In[35]:


len(Doors)


# In[36]:


Coolingsystem


# In[37]:


len(Coolingsystem)


# In[38]:


Rating


# In[39]:


len(Rating)


# In[40]:


for i in container:
    print(i.find("li",class_="J+igdf"))


# In[41]:


import numpy as np
import pandas as pd

Compressor=[]
OriginalPrice=[]
SellingPrice=[]
DiscountPercentage=[]



for i in container:
    print(i.text)
    text=i.text
    
    compressor=re.findall("Reviews(.+)Bu",text)
    if compressor:
        Compressor.append(compressor[0])
    else:
        Compressor.append(np.nan)
        
    originalprice=i.find("div",class_="yRaY8j ZYYwLA")
    
    if originalprice:
        OriginalPrice.append(originalprice.text)
    else:
        OriginalPrice.append(np.nan)
        
    sellingprice=i.find("div",class_="Nx9bqj _4b5DiR")
    if sellingprice:
        SellingPrice.append(sellingprice.text)
    else:
        SellingPrice.append(np.nan)
        
    discountpercentage=i.find("div",class_="UkUFwK")
    if discountpercentage:
        DiscountPercentage.append(discountpercentage.text)
    else:
        DiscountPercentage.append(np.nan)
        
    


# In[42]:


Compressor


# In[43]:


len(Compressor)


# In[44]:


OriginalPrice


# In[45]:


len(OriginalPrice)


# In[46]:


SellingPrice


# In[47]:


len(SellingPrice)


# In[48]:


DiscountPercentage


# In[49]:


len(DiscountPercentage) 


# In[50]:


d={"Brand":Brand,"Storage":Storage,"Stars":Stars,"Doors":Doors,"Coolingsystem":Coolingsystem,"Rating":Rating,"Compressor":Compressor,"OriginalPrice":OriginalPrice,"SellingPrice":SellingPrice,"DiscountPercentage":DiscountPercentage}


# In[51]:


import pandas as pd


# In[52]:


df=pd.DataFrame(d)


# In[53]:


df.to_csv("FRIDGES.csv")


# In[54]:


df


# In[55]:


for i in range(1,30):
    print(f'url-{i}')


# In[56]:


Brand=[]
Storage=[]
Stars=[]
Doors=[]
Coolingsystem=[]
Rating=[]
Compressor=[]
OriginalPrice=[]
SellingPrice=[]
DiscountPercentage=[]

for i in range(1,30):
    
    url=f"https://www.flipkart.com/search?q=refrigerator&as=on&as-show=on&otracker=AS_Query_OrganicAutoSuggest_6_3_na_na_na&otracker1=AS_Query_OrganicAutoSuggest_6_3_na_na_na&as-pos=6&as-type=RECENT&suggestionId=refrigerator&requestId=97b3fe07-a0ff-4ccc-8853-1ff3476a916e&as-searchtext=refrigerator&p%5B%5D=facets.price_range.from%3D15000&p%5B%5D=facets.price_range.to%3DMax&page={i}"    
    
    request_header = {'Content-Type': 'text/html; charset=UTF-8','User-Agent': 'Chrome/101.0.0.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0','Accept-Encoding': 'gzip, deflate, br'}

    page = requests.get(url,headers=request_header)
    
    soup=BeautifulSoup(page.text)
    
    container = soup.find_all("div",class_="tUxRFH")
    
    for i in container:
        t = i.find("div",class_="KzDlHZ") # t is title of mobiles and class_="KzDlHZ" is tag for title
        text=t.text
        
        brand=re.findall("^\w+",text)
        if brand:
            Brand.append(brand[0])
        else:
            Brand.append(np.nan)

        storage=re.findall("(\d+\sL)",text)
        if storage:
            Storage.append(storage[0])
        else:
            Storage.append(np.nan)

        stars=re.findall(" Door (\d Star) ",text)
        if stars:
            Stars.append(stars[0])
        else:
            Stars.append(np.nan)

        doors=re.findall("(Double|Multi|Single|Side by Side)",text)
        if doors:
            Doors.append(doors[0])
        else:
            Doors.append(np.nan)

        coolingsystem=re.findall("(Frost Free|Direct Cool)",text)
        if coolingsystem:
            Coolingsystem.append(coolingsystem[0])
        else:
            Coolingsystem.append(np.nan)
        rating = i.find("div",class_="XQDdHH")                                 
        if rating:                                                             
            Rating.append(rating.text)
        else:
            Rating.append(np.nan)
            

    for i in container:
        print(i.text)
        text=i.text

        compressor=re.findall("Reviews(.+)Bu",text)
        if compressor:
            Compressor.append(compressor[0])
        else:
            Compressor.append(np.nan)

        originalprice=i.find("div",class_="yRaY8j ZYYwLA")

        if originalprice:
            OriginalPrice.append(originalprice.text)
        else:
            OriginalPrice.append(np.nan)

        sellingprice=i.find("div",class_="Nx9bqj _4b5DiR")
        if sellingprice:
            SellingPrice.append(sellingprice.text)
        else:
            SellingPrice.append(np.nan)

        discountpercentage=i.find("div",class_="UkUFwK")
        if discountpercentage:
            DiscountPercentage.append(discountpercentage.text)
        else:
            DiscountPercentage.append(np.nan)



# In[57]:


d={"Brand":Brand,"Storage":Storage,"Stars":Stars,"Doors":Doors,"Coolingsystem":Coolingsystem,"Rating":Rating,"Compressor":Compressor,"OriginalPrice":OriginalPrice,"SellingPrice":SellingPrice,"DiscountPercentage":DiscountPercentage}


# In[58]:


df=pd.DataFrame(d)


# In[59]:


df


# In[60]:


df.to_csv("FRIDGES.csv")


# In[61]:


df


# In[62]:


df.shape


# In[63]:


df.head()


# In[64]:


df.columns


# In[65]:


df.info()


# In[66]:


df.isnull().sum()


# In[67]:


df[df.duplicated()]


# In[68]:


df.drop_duplicates(inplace=True)


# In[69]:


df[df.duplicated()]


# In[70]:


df.to_csv("FRIDGES.csv")


# In[71]:


df


# In[72]:


df.reset_index()


# In[73]:


df.to_csv("FRIDGES.csv")


# In[74]:


df.reset_index()


# In[75]:


df.describe()


# In[76]:


df.head()


# In[77]:


df.info()


# In[78]:


df.isna().sum()


# In[79]:


df[df["Stars"].isna()]


# In[80]:


df[df["Storage"].isna()]


# to remove unwanted column
# df=df.drop("Unnamed: 0",axis=1)
# df

# In[81]:


df[df["Stars"].isna()]


# In[82]:


l=['Rating','OriginalPrice','SellingPrice','DiscountPercentage']

for i in l:
    df[i]=df[i].replace("[^0-9]","",regex=True).astype("float")


# In[83]:


df['Rating']=df['Rating']/10


# In[84]:


df


# In[85]:


df.info()


# In[86]:


df["Storage"]=df["Storage"].str.replace("L","")
df


# In[87]:


df["Stars"]=df["Stars"].str.replace("Star","")
df


# In[88]:


df.info()


# In[89]:


df['Storage'].fillna(0,inplace=True)
df['Storage']=df['Storage'].astype(int)


# In[90]:


df['Stars'].fillna(0,inplace=True)
df['Stars']=df['Stars'].astype(int)


# In[91]:


df.info()


# In[92]:


df["Brand"].isna().sum()


# In[93]:


import seaborn as sns


# In[94]:


sns.boxplot(df['Storage'])


# In[95]:


df["Storage"].mean()


# In[96]:


df["Storage"].median()


# In[97]:


median_storage = df['Storage'].median()
df['Storage'].fillna(median_storage, inplace=True)


# In[98]:


df["Stars"].mean()


# In[99]:


df["Stars"].median()


# In[100]:


median_stars = df['Stars'].median()
df['Stars'].fillna(median_stars, inplace=True)


# In[101]:


df['Storage'].isna().sum()


# In[102]:


df['Stars'].isna().sum()


# In[103]:


df


# In[104]:


df["Doors"].mode().iloc[0]


# In[105]:


df['Doors'] = df.groupby('Brand')['Doors'].transform(lambda x: x.fillna(x.mode().iloc[0]))


# In[106]:


df.isna().sum()


# In[107]:


def fill_missing_rating(df):
    brand_mean_rating = df.groupby('Brand')['Rating'].mean()
    def fill_rating(row):
        if pd.isna(row['Rating']):
            return brand_mean_rating[row['Brand']]
        else:
            return row['Rating']
    df['Rating'] = df.apply(fill_rating, axis=1)
    return df
    


# In[108]:


df_filled = fill_missing_rating(df)

print(df_filled)


# In[109]:


#mode_Doors = df['Doors'].mode()
#df['Stars'].fillna(median_stars, inplace=True)


# In[110]:


df


# In[111]:


df.isna().sum()


# In[112]:


median_ratings = df['Rating'].median()
df['Rating'].fillna(median_ratings, inplace=True)


# In[113]:


df.isna().sum()


# In[114]:


mode_value = df['Compressor'].mode()[0]
df['Compressor'].fillna(mode_value, inplace=True)


# In[115]:


df.isna().sum()


# In[116]:


df


# In[117]:


df.info()


# In[118]:


#df1=df.copy()


# In[119]:


for i in ["OriginalPrice","SellingPrice","DiscountPercentage"]:
    df[i]=df[i].fillna(0).astype(int)
    


# In[120]:


df.info()


# In[121]:


df.isna().sum()


# In[122]:


df["OriginalPrice"].fillna(df["SellingPrice"])


# In[123]:


df["OriginalPrice"].fillna(df["SellingPrice"]).isna().sum()


# In[124]:


df["OriginalPrice"]=df["OriginalPrice"].fillna(df["SellingPrice"])


# In[125]:


df.isna().sum()


# In[126]:


df.info()


# In[127]:


df


# In[128]:


df['OriginalPrice'].fillna(df['SellingPrice'], inplace=True) 


# In[129]:


df


# In[130]:


df.tail(50)


# In[131]:


df.reset_index()


# In[132]:


grouped = df.groupby(['OriginalPrice', 'SellingPrice'])
group_mean = grouped['DiscountPercentage'].mean()

def fill_discount_percentage(row):
    if pd.isna(row['DiscountPercentage']):
        key = (row['OriginalPrice'], row['SellingPrice'])
        return group_mean.get(key, None)
    else:
        return row['DiscountPercentage']
    


# In[133]:


df['DiscountPercentage'] = df.apply(fill_discount_percentage, axis=1)


# In[134]:


df.isna().sum()


# In[135]:


df.info()


# # UNIVARIATE ANALYSYS

# In[136]:


df["Brand"].unique()


# In[137]:


df["Brand"].nunique()


# In[138]:


df["Brand"].value_counts()


# In[139]:


df["Brand"].value_counts().plot(kind="bar")


# In[140]:


#univariate analysis
import seaborn as sns
import matplotlib.pyplot as plt

def bar_plot(data,xl,title_,r=90):   # bar plot for categorical column
    plt.figure(figsize=(12,6))
    sns.barplot(x=data.index,y=data.values,palette='icefire')
    plt.xlabel(xl)
    plt.ylabel("Count")
    plt.title(title_)
    plt.xticks(rotation = r)
    plt.show()

def hist_plot(data,xl,title_):  #plot for numerical column
    plt.figure(figsize=(12,6))
    sns.histplot(data,color="blue",kde=True)
    plt.xlabel(xl)
    plt.ylabel("Frequency")
    plt.title(title_)
    plt.show()

def box_plot(data,xl,title_):
    plt.figure(figsize=(8,5))
    sns.boxplot(data,color="red")
    plt.xlabel(xl)
    plt.ylabel("Outliers")
    plt.title(title_)
    plt.show()
    
def analyze_categorical(col):
    print(f"-------------{col}-------------")
    print("********"*10)

    unique_val=df[col].unique()
    nunique_val=df[col].nunique()
    freq=df[col].value_counts()
    
    print(f"The unique values in {col} are :\n",unique_val)
    print()
    print(f"The no of  unique values in {col} is :\n",nunique_val)
    print()
    print(f"The value counts of {col} is :\n",freq)
    
    if nunique_val<25:
        bar_plot(freq,col,f"The count of {col} using barplot")
    else:
        freq = df[col].value_counts()[:20]
        bar_plot(freq,col,f"The count of Top 20 in {col} using barplot")
        
def analyze_numerical(col):
    print(f"-------------{col}-------------")
    print("********"*10)
    
    descriptive_stats = df[col].describe()
    skewness = df[col].skew()
    nunique_val=df[col].nunique()
    discrete = nunique_val <15
    
    
    print(f"The Descriptive Stats of  {col} are :\n",descriptive_stats)
    print()
    print(f"The skewness of  {col} is :\n",skewness)
    print()
    
    if discrete:
        freq = df[col].value_counts()
        bar_plot(freq,col,f"The count of {col} using barplot")
    else:
        hist_plot(df[col],col,f"The distribution of {col} using Histogram")
        
def univariate_analysis(df):
        cat_col = df.select_dtypes("object").columns
        num_col = df.select_dtypes(exclude = "object").columns
        
        for col  in cat_col:
            analyze_categorical(col)
        for col in num_col:
            analyze_numerical(col)
    
    


    


# In[141]:


bar_plot(df["Brand"].value_counts(),"Brand","Bar plot on Brand")


# In[142]:


hist_plot(df["SellingPrice"],"SellingPrice","Distribution of SellingPrice")


# In[143]:


box_plot(df["SellingPrice"],"SellingPrice","Outliers in SellingPrice")


# In[144]:


analyze_categorical("Brand")


# In[145]:


analyze_numerical("SellingPrice")


# In[146]:


univariate_analysis(df)


# # BI VARIATE ANALYSYS
# ANALYSYS ON TWO COLUMNS AT A TIME

# In[147]:


df


# In[148]:


df.reset_index()


# In[149]:


df.head()


#  NUMERICAL AND NUMERICAL COLUMN ANALYSIS

# In[182]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Storage', y='SellingPrice', data=df, palette='viridis')
plt.title('Relationship between Storage and Selling Price')
plt.xlabel('Storage Capacity (liters)')
plt.ylabel('Selling Price')
plt.grid(True)
plt.show()


# NUMERICAL AND CATEGORICAL (OR) CATEGORICAL AND NUMERICAL COLUMN ANALYSYS

# In[188]:


plt.figure(figsize=(12, 6))
sns.barplot(x='Brand', y='SellingPrice', data=df, palette='viridis')
plt.title('Distribution of Selling Prices by Brand')
plt.xlabel('Brand')
plt.ylabel('Selling Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[191]:


plt.figure(figsize=(12, 6))
sns.barplot(x='Brand', y='DiscountPercentage', data=df, palette='viridis')
plt.title('Distribution of Discount Percentages by Brand')
plt.xlabel('Brand')
plt.ylabel('Discount Percentage')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ANALYSYS ON CATEGORICAL AND CATEGORICAL COLUMNS

# In[194]:


cooling_brand_counts = pd.crosstab(df['Brand'], df['Coolingsystem'])


# In[198]:


plt.figure(figsize=(10, 6))
cooling_brand_counts.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Distribution of Cooling Systems by Brand')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Cooling System')
plt.grid(True)
plt.show()


# In[179]:


sns.heatmap(df.corr(numeric_only=True),annot=True)


# In[ ]:




