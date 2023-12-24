import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# 设置SMTP服务器和端口
smtp_server = 'smtp.edu.com'
port = 465  # 通常为587或465

# 创建邮件对象
msg = MIMEMultipart()
msg['From'] = '202320100998@mail.scut.edu.cn'  # 发件人邮箱
msg['To'] = '13528767608@163.com'  # 收件人邮箱
msg['Subject'] = '主题'  # 邮件主题

# 邮件正文
body = '这是邮件正文内容'  # 邮件正文内容
msg.attach(MIMEText(body, 'plain'))

# 连接SMTP服务器并发送邮件
with smtplib.SMTP(smtp_server, port) as server:
    server.starttls()  # 启动TLS加密
    server.login(msg['From'], '@SCUTARacing2019')  # 邮箱登录密码
    server.sendmail(msg['From'], msg['To'], msg.as_string())