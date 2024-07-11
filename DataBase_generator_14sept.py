import datetime
import random
import time
import mysql.connector as sql

class dbconnect():
    def __init__(self, hname, usr, pwd, datab, tablenm):
        self.db = sql.connect(
            host=hname,
            user=usr,
            password=pwd,
            database=datab,
        )
        if self.db.is_connected() == False:
            print("not connected")
        if self.db.is_connected() == True:
            print(" connected")
        self.curs = self.db.cursor()
        self.tablename = tablenm
      

    def add_dbdata(self, data):
        try:
            self.curs.execute(f"INSERT INTO {self.tablename} (plate_text) VALUES (%s)", (data,))
            self.db.commit()
            print("Added data to database!")
        except Exception as e:
            print("Data push failed:", e)
