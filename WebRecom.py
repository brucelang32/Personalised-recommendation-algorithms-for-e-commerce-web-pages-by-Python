"""
Developer: Wenkai Lang
"""
import pandas as pd
import pymysql as pm
import re
from random import sample


#  access the database and save to csv file
def mysql_access():
    try:
        db = pm.connect(host="localhost", user="root", password="123456", database="project1", charset="utf8")

        print("database is connected successfully.")

        data = pd.read_sql('select * from all_gzdata', con=db)
        db.close()

        data.to_csv('all_gzdata.csv', index=False, encoding='utf-8')
        print('data saved to excel successfully.')
    except pm.Error as e:
        print("a failure to connect database：" + str(e))


def pageviews_statistics():
    db = pm.connect(host="localhost", user="root", password="123456", database="xiangmu1", charset="utf8")
    sql_cmd = "SELECT * FROM all_gzdata"

    # analysis of web page type
    sql = pd.read_sql(sql=sql_cmd, con=db, chunksize=10000)
    counts = [i['fullURLId'].value_counts() for i in sql] 
    counts = counts.copy()
    counts = pd.concat(counts).groupby(level=0).sum() 
    counts = counts.reset_index() 
    counts.columns = ['index', 'num'] 
    counts['type'] = counts['index'].str.extract('(\d{3})')  # extract the first three numbers as category ID
    counts_ = counts[['type', 'num']].groupby('type').sum()
    counts_.sort_values(by='num', ascending=False, inplace=True)
    counts_['ratio'] = counts_.iloc[:, 0] / counts_.iloc[:, 0].sum()
    print(counts_)

    # There is only one type: 107001，thus it can be further subdivided into：homepage, list and content
    def count107(i):
        j = i[['fullURL']][i['fullURLId'].str.contains('107')].copy() 
        j['type'] = None 
        j['type'][j['fullURL'].str.contains('info/.+?/')] = u'homepage'
        j['type'][j['fullURL'].str.contains('info/.+?/.+?')] = u'list'
        j['type'][j['fullURL'].str.contains('/\d+?_*\d+?\.html')] = u'content'
        # print('*********j************')
        # print(j.head(10))
        return j['type'].value_counts()

    sql = pd.read_sql(sql=sql_cmd, con=db, chunksize=10000)

    counts2 = [count107(i) for i in sql]
    counts2 = pd.concat(counts2).groupby(level=0).sum()
    print('*********counts2************')
    print(counts2)
    # percentage
    res107 = pd.DataFrame(counts2)
    # res107.reset_index(inplace=True)
    res107.index.name = u'107type'
    res107.rename(columns={'type': 'num'}, inplace=True)
    res107[u'proportion'] = res107['num'] / res107['num'].sum()
    res107.reset_index(inplace=True)
    print('*************res107******************')
    print(res107)

    def countquestion(i):
        j = i[['fullURLId']][i['fullURL'].str.contains('\?')].copy()  # find URLs containing?
        return j

    sql = pd.read_sql(sql=sql_cmd, con=db, chunksize=10000)

    counts3 = [countquestion(i)['fullURLId'].value_counts() for i in sql]
    counts3 = pd.concat(counts3).groupby(level=0).sum()
    print('*************counts3******************')
    print(counts3)

    # calculate the ercentage of all types and save
    df1 = pd.DataFrame(counts3)
    df1['perc'] = df1['fullURLId'] / df1['fullURLId'].sum() * 100
    df1.sort_values(by='fullURLId', ascending=False, inplace=True)
    print(df1.round(4))

    def page199(i):
        j = i[['fullURL', 'pageTitle']][(i['fullURLId'].str.contains('199')) &
                                        (i['fullURL'].str.contains('\?'))]
        j['pageTitle'].fillna(u'空', inplace=True)
        j['type'] = u'others'
        j['type'][j['pageTitle'].str.contains(u'法律快车-律师助手')] = u'lawtime-LegalAssist'
        j['type'][j['pageTitle'].str.contains(u'咨询发布成功')] = u'LegalAdvicePostedSuccess'
        j['type'][j['pageTitle'].str.contains(u'免费发布法律咨询')] = u'FreeLegalAdvicePosted'
        j['type'][j['pageTitle'].str.contains(u'法律快搜')] = u'快搜'
        j['type'][j['pageTitle'].str.contains(u'法律快车法律经验')] = u'lawtime-LegalExperience'
        j['type'][j['pageTitle'].str.contains(u'法律快车法律咨询')] = u'lawtime-LegalAdvice'
        j['type'][(j['pageTitle'].str.contains(u'_法律快车')) |
                  (j['pageTitle'].str.contains(u'-法律快车'))] = u'lawtime'
        j['type'][j['pageTitle'].str.contains(u'空')] = u'Null'

        return j

    sql = pd.read_sql(sql=sql_cmd, con=db, chunksize=10000)

    counts4 = [page199(i) for i in sql]
    counts4 = pd.concat(counts4)
    d1 = counts4['type'].value_counts()
    print('*************d1******************')
    print(d1)
    d2 = counts4[counts4['type'] == u'others']
    print('*************d2******************')
    print(d2)
    # calculate the percentage of all types and save
    df1_ = pd.DataFrame(d1)
    df1_['perc'] = df1_['type'] / df1_['type'].sum() * 100
    df1_.sort_values(by='type', ascending=False, inplace=True)
    print('*************df1_******************')
    print(df1_)

    def xiaguang(i):  # define the function
        j = i.loc[(i['fullURL'].str.contains('\.html')) == False,
                  ['fullURL', 'fullURLId', 'pageTitle']]
        return j

    # access the database again
    sql = pd.read_sql(sql=sql_cmd, con=db, chunksize=10000)

    counts5 = [xiaguang(i) for i in sql]
    counts5 = pd.concat(counts5)

    xg1 = counts5['fullURLId'].value_counts()
    print('*************xg1******************')
    print(xg1)
    # calculate the percentage
    xg_ = pd.DataFrame(xg1)
    xg_.reset_index(inplace=True)
    xg_.columns = ['index', 'num']
    xg_['perc'] = xg_['num'] / xg_['num'].sum() * 100
    xg_.sort_values(by='num', ascending=False, inplace=True)

    xg_['type'] = xg_['index'].str.extract('(\d{3})')  # extract the first three numbers as category ID

    xgs_ = xg_[['type', 'num']].groupby('type').sum()
    xgs_.sort_values(by='num', ascending=False, inplace=True)
    xgs_['percentage'] = xgs_['num'] / xgs_['num'].sum() * 100
    print('*************xgs_******************')
    print(xgs_.round(4))

    # analysis and count of PageClick
    sql = pd.read_sql(sql=sql_cmd, con=db, chunksize=10000)

    counts1 = [i['realIP'].value_counts() for i in sql]  # calculate the number of occurrences of each IP by chunk
    counts1 = pd.concat(counts1).groupby(level=0).sum()  # level=0 means group by index
    print('*************counts1******************')
    print(counts1)

    counts1_ = pd.DataFrame(counts1)
    print('*************counts1_******************')
    print(counts1_)
    counts1['realIP'] = counts1.index.tolist()

    counts1_[1] = 1
    hit_count = counts1_.groupby('realIP').sum()  # Calculate time of "Different click" appear

    hit_count.columns = [u'user_num']
    hit_count.index.name = u'click_num'

    # Calculate num of users 1-7 and 7+
    hit_count.sort_index(inplace=True)
    hit_count_7 = hit_count.iloc[:7, :]
    time = hit_count.iloc[7:, 0].sum()  
    hit_count_7 = hit_count_7.append([{u'user_num': time}], ignore_index=True)
    hit_count_7.index = ['1', '2', '3', '4', '5', '6', '7', '7+']
    hit_count_7[u'user_percentage'] = hit_count_7[u'user_num'] / hit_count_7[u'user_num'].sum()
    print('*************hit_count_7******************')
    print(hit_count_7)

    # analyze user's single behaviour
    all_gzdata = pd.read_sql(sql=sql_cmd, con=db)

    # count realIP
    real_count = pd.DataFrame(all_gzdata.groupby("realIP")["realIP"].count())
    real_count.columns = ["count"]
    real_count["realIP0"] = real_count.index.tolist()
    user_one = real_count[(real_count["count"] == 1)]
    print('*************user_one******************')
    print(user_one)
    real_one = pd.merge(user_one, all_gzdata, left_on='realIP', right_on='realIP')

    # count webpage type
    URL_count = pd.DataFrame(real_one.groupby("fullURLId")["fullURLId"].count())
    URL_count.columns = ["count"]
    URL_count.sort_values(by='count', ascending=False, inplace=True)
    URL_count_4 = URL_count.iloc[:4, :]
    time = URL_count_4.iloc[4:, 0].sum()
    URLindex = URL_count_4.index.values
    URL_count_4 = URL_count_4.append([{'count': time}], ignore_index=True)
    URL_count_4.index = [URLindex[0], URLindex[1], URLindex[2], URLindex[3], 'others']
    URL_count_4[u'percentage'] = URL_count_4['count'] / URL_count_4['count'].sum()
    print('*************URL_count_4******************')
    print(URL_count_4)

    fullURL_count = pd.DataFrame(real_one.groupby("fullURL")["fullURL"].count())
    fullURL_count.columns = ["count"]
    fullURL_count["fullURL"] = fullURL_count.index.tolist()
    fullURL_count.sort_values(by='count', ascending=False, inplace=True) 
    print('*************fullURL_count.head(10)******************')
    print(fullURL_count.head(10))
    fullURL_count.to_excel('fullURL_count.xlsx')


def web_pretreatment():
    db = pm.connect(host="localhost", user="root", password="123456", database="xiangmu1", charset="utf8")
    data = pd.read_sql('select * from all_gzdata', con=db)
    db.close() 

    index107 = [re.search('107', str(i)) != None for i in data.loc[:, 'fullURLId']]
    data_107 = data.loc[index107, :]

    index = [re.search('hunyin', str(i)) != None for i in data_107.loc[:, 'fullURL']]
    data_hunyin = data_107.loc[index, :]

    info = data_hunyin.loc[:, ['realIP', 'fullURL']]

    # Remove ? and after
    da = [re.sub('\?.*', '', str(i)) for i in info.loc[:, 'fullURL']]
    info.loc[:, 'fullURL'] = da  # transfer fullURL in info to da
    # Remove pages without html
    index = [re.search('\.html', str(i)) != None for i in info.loc[:, 'fullURL']]
    index.count(True) 
    info1 = info.loc[index, :]
    print('*************info1******************')
    print(info1.head())

    # Find full page and preview page
    index = [re.search('/\d+_\d+\.html', i) != None for i in info1.loc[:, 'fullURL']]
    index1 = [i == False for i in index]
    info1_1 = info1.loc[index, :] 
    info1_2 = info1.loc[index1, :]  
    # get full URL of preview webpage
    da = [re.sub('_\d+\.html', '.html', str(i)) for i in info1_1.loc[:, 'fullURL']]
    info1_1.loc[:, 'fullURL'] = da
    # merge webpage preview and full
    frames = [info1_1, info1_2]
    info2 = pd.concat(frames)
    info3 = info2.drop_duplicates()
    info3.iloc[:, 0] = [str(index) for index in info3.iloc[:, 0]]
    info3.iloc[:, 1] = [str(index) for index in info3.iloc[:, 1]]
    print('*************info3******************')
    print(len(info3))

    # Filter IP meet certain number of views
    IP_count = info3['realIP'].value_counts()
    IP = list(IP_count.index)
    count = list(IP_count.values)
    IP_count = pd.DataFrame({'IP': IP, 'count': count})
    n = 2
    index = IP_count.loc[:, 'count'] > n
    IP_index = IP_count.loc[index, 'IP']
    print('*************IP_index******************')
    print(IP_index.head())

    # Split the IP set to training and test set
    index_tr = sample(range(0, len(IP_index)), int(len(IP_index) * 0.8))  # 或者np.random.sample
    index_te = [i for i in range(0, len(IP_index)) if i not in index_tr]
    IP_tr = IP_index[index_tr]
    IP_te = IP_index[index_te]
    # Split the dataset into training and test set
    index_tr = [i in list(IP_tr) for i in info3.loc[:, 'realIP']]
    index_te = [i in list(IP_te) for i in info3.loc[:, 'realIP']]
    data_tr = info3.loc[index_tr, :]
    data_te = info3.loc[index_te, :]
    print('*************len(data_tr)******************')
    print(len(data_tr))
    IP_tr = data_tr.iloc[:, 0]
    url_tr = data_tr.iloc[:, 1]
    IP_tr = list(set(IP_tr))
    url_tr = list(set(url_tr))
    print('*************len(url_tr)******************')
    print(len(url_tr))

    # Calculate the matrix using training dataset
    UI_matrix_tr = pd.DataFrame(0, index=IP_tr, columns=url_tr)
    # Calculate user-item matrix
    for i in data_tr.index:
        UI_matrix_tr.loc[data_tr.loc[i, 'realIP'], data_tr.loc[i, 'fullURL']] = 1
    sum(UI_matrix_tr.sum(axis=1))
    print('*************sum(UI_matrix_tr.sum(axis=1))******************')
    print(sum(UI_matrix_tr.sum(axis=1)))

    # calculate item matrix
    Item_matrix_tr = pd.DataFrame(0, index=url_tr, columns=url_tr)
    for i in Item_matrix_tr.index:
        for j in Item_matrix_tr.index:
            a = sum(UI_matrix_tr.loc[:, [i, j]].sum(axis=1) == 2)
            b = sum(UI_matrix_tr.loc[:, [i, j]].sum(axis=1) != 0)
            Item_matrix_tr.loc[i, j] = a / b
    print('*************Item_matrix_tr.head(10)******************')
    print(Item_matrix_tr.head(10))

    # Make matrix diagonal 0
    for i in Item_matrix_tr.index:
        Item_matrix_tr.loc[i, i] = 0

    # evaluate the matrix using test dataset
    IP_te = data_te.iloc[:, 0]
    url_te = data_te.iloc[:, 1]
    IP_te = list(set(IP_te))
    url_te = list(set(url_te))

    # user-item matrix of test dataset
    UI_matrix_te = pd.DataFrame(0, index=IP_te, columns=url_te)
    for i in data_te.index:
        UI_matrix_te.loc[data_te.loc[i, 'realIP'], data_te.loc[i, 'fullURL']] = 1

    # recommend in test dataset
    Res = pd.DataFrame('NaN', index=data_te.index,
                       columns=['IP', 'URI_viewed', 'URI_recomm', 'T/F'])
    Res.loc[:, 'IP'] = list(data_te.iloc[:, 0])
    Res.loc[:, 'URI_viewed'] = list(data_te.iloc[:, 1])

    # Start recommendation
    for i in Res.index:
        if Res.loc[i, 'URI_viewed'] in list(Item_matrix_tr.index):
            Res.loc[i, 'URI_recomm'] = Item_matrix_tr.loc[Res.loc[i, 'URI_viewed'], :].argmax()
            if Res.loc[i, 'URI_recomm'] in url_te:
                Res.loc[i, 'T/F'] = UI_matrix_te.loc[Res.loc[i, 'IP'],
                                                     Res.loc[i, 'URI_recomm']] == 1
            else:
                Res.loc[i, 'T/F'] = False

    Res.to_csv('Result.csv', index=False, encoding='utf8')
    print('*************Res.head(10)******************')
    print(Res.head(10))


def model_evaluate():
    Res = pd.read_csv('Result.csv', keep_default_na=False, encoding='utf-8')
    Pre = round(sum(Res.loc[:, 'T/F'] == 'True') / (len(Res.index) - sum(Res.loc[:, 'T/F'] == 'NaN')), 4)
    print('Acc:', Pre)
    Rec = round(sum(Res.loc[:, 'T/F'] == 'True') / (sum(Res.loc[:, 'T/F'] == 'True') + sum(Res.loc[:, 'T/F'] == 'NaN')),
                4)
    print('Recall:', Rec)
    F1 = round(2 * Pre * Rec / (Pre + Rec), 4)
    print('F1:', F1)


if __name__ == '__main__':
    mysql_access()
    pageviews_statistics()
    web_pretreatment()
    model_evaluate()
