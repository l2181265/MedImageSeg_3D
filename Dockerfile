FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN apt-get update && apt-get install -y openssh-server
RUN apt-get install -y sudo screen git nano
RUN mkdir /var/run/sshd
RUN groupadd --gid 1024 docker_user_group \
  && useradd -m -s /bin/bash -u 1001 --groups docker_user_group lvyan \
  && echo "lvyan:lvyan" | chpasswd \
  && adduser lvyan sudo

 #RUN cat /dev/null > /etc/bash.bashrc #for tf docker ssh debug bug

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"

RUN echo "export VISIBLE=now" >> /etc/profile
RUN echo 'export PATH="/opt/conda/bin:$PATH"'  >> /etc/profile

WORKDIR /workspace
COPY ./   /workspace

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pip -U

RUN pip install -r requirements.txt

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
