import re, pymysql, time
import pandas as pd
import sys
sys.path.append('../../')
import config 

def get_cate(itemIds):
    cats=list()
    item_ids=[]
    # Connect to the database
    connection = pymysql.connect(host=config.DATABASE_CONFIG['host'],
                                 user=config.DATABASE_CONFIG['user'],
                                 password=config.DATABASE_CONFIG['password'],
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    with connection.cursor() as cursor:
        sql = "SELECT catId,newsId FROM news.news_resource where sourceNews='Soha' and newsId IN ({}) ".format(",".join(itemIds))
        cursor.execute(sql)
        try:
            for row in cursor.fetchall():
                cats.append(row['catId'])
                item_ids.append(row['newsId'])
        except Exception as e:
            print(e)
        finally:
            connection.close()
    return cats , item_ids 

def get_content(ids):
    content = []
    news_ids=[]
    # Connect to the database
    connection = pymysql.connect(host=config.DATABASE_CONFIG['host'],
                                 user=config.DATABASE_CONFIG['user'],
                                 password=config.DATABASE_CONFIG['password'],
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    with connection.cursor() as cursor:
        sql = "SELECT content_token,news_id FROM recsys.news_token where domain='Soha' and news_id IN ({})".format(",".join(ids))
        cursor.execute(sql)
        try:
            for row in cursor.fetchall():
                content.append(row['content_token'])
                news_ids.append(row['news_id'])
        except Exception as e:
            print(e)
    
        finally:
            connection.close()

    return content,news_ids





