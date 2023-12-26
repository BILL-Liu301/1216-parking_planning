import os
import time
import smtplib
from threading import Thread
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

from .base_path import path_result


class Sending_email(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.f = input("请输入源头邮箱（学号）：") + '@mail.scut.edu.cn'
        self.p = input("请输入专用密码：")
        self.t = input("请输入目标邮箱（163邮箱）：") + '@163.com'
        if self.t.split('@')[0] == 'copy':
            self.t = self.f
        os.system('CLS')
        print(f"{self.f} ----> {self.t}")
        self.flag_sending = True

    def send_email(self):
        con = smtplib.SMTP_SSL('smtp.mail.scut.edu.cn', 465)
        con.login(self.f, self.p)
        msg = MIMEMultipart()
        subject = Header('倒车入库轨迹规划模型训练', 'utf-8').encode()
        msg['Subject'] = subject
        msg['From'] = f'{self.f} <{self.f}m>'
        msg['To'] = self.t
        result = open(path_result, 'r').read()
        text = MIMEText(result, 'plain', 'utf-8')
        msg.attach(text)
        con.sendmail(self.f, self.t, msg.as_string())
        con.quit()

    def run(self):
        while self.flag_sending:
            try:
                self.send_email()
            except Exception as e:
                pass
            time.sleep(60)
