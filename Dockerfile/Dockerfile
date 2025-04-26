FROM python:3.11-slim-bullseye

LABEL maintainer="murphl3 <lmurphy3@luc.edu>" \
	version="0.1" \
	description="A completely functional Apache Hadoop ecosystem, meant for learning and development in a small package."

ARG USERNAME=user \
	UID=1000 \
	GID=1000

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
	HADOOP_HOME=/opt/hadoop-3.3.6 \
	SPARK_HOME=/opt/spark-3.5.5-bin-without-hadoop \
	HIVE_HOME=/opt/apache-hive-3.1.3-bin \
	HBASE_HOME=/opt/hbase-2.4.18 \
	HUE_HOME=/opt/hue-4.11.0 \
	OOZIE_HOME=/opt/oozie-5.2.1 \
	ZOOKEEPER_HOME=/opt/apache-zookeeper-3.8.4-bin \
	SQOOP_HOME=/opt/sqoop-1.4.7 \
	SOLR_HOME=/opt/solr-9.8.1-slim \
	IMPALA_HOME=/opt/apache-impala-4.4.1 \
	LANG=en_US.UTF-8 \
	HOME=/home/${USERNAME}
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin:${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin:${SPARK_HOME}/bin:${HIVE_HOME}/bin:${HBASE_HOME}/bin:${HUE_HOME}/bin:${OOZIE_HOME}/bin:${ZOOKEEPER_HOME}/bin
RUN echo -e 'export JAVA_HOME=${JAVA_HOME}\nexport HADOOP_HOME=${HADOOP_HOME}\nexport SPARK_HOME=${SPARK_HOME}\nexport HIVE_HOME=${HIVE_HOME}\nexport HBASE_HOME=${HBASE_HOME}\nexport HUE_HOME=${HUE_HOME}\nexport OOZIE_HOME=${OOZIE_HOME}\nexport ZOOKEEPER_HOME=${ZOOKEEPER_HOME}\nexport SQOOP_HOME=${SQOOP_HOME}\nexport SOLR_HOME=${SOLR_HOME}\nexport IMPALA_HOME=${IMPALA_HOME}\nexport PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin:${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin:${SPARK_HOME}/bin:${HIVE_HOME}/bin:${HBASE_HOME}/bin:${HUE_HOME}/bin:${OOZIE_HOME}/bin:${ZOOKEEPER_HOME}/bin' >> /etc/profile.d/global_env.sh && \
	chmod +x /etc/profile.d/global_env.sh && \
	groupadd --gid ${GID} ${USERNAME} && \
	useradd --uid ${UID} --gid ${GID} --shell /bin/bash --create-home ${USERNAME} && \
	echo -e 'export JAVA_HOME=${JAVA_HOME}\nexport HADOOP_HOME=${HADOOP_HOME}\nexport SPARK_HOME=${SPARK_HOME}\nexport HIVE_HOME=${HIVE_HOME}\nexport HBASE_HOME=${HBASE_HOME}\nexport HUE_HOME=${HUE_HOME}\nexport OOZIE_HOME=${OOZIE_HOME}\nexport ZOOKEEPER_HOME=${ZOOKEEPER_HOME}\nexport SQOOP_HOME=${SQOOP_HOME}\nexport SOLR_HOME=${SOLR_HOME}\nexport IMPALA_HOME=${IMPALA_HOME}\nexport PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin:${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin:${SPARK_HOME}/bin:${HIVE_HOME}/bin:${HBASE_HOME}/bin:${HUE_HOME}/bin:${OOZIE_HOME}/bin:${ZOOKEEPER_HOME}/bin' >> ~/.bashrc && \
	DEBIAN_FRONTEND=noninteractive apt-get update && \
	DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y \
		curl \
		git \
		gnupg \
		openssh-client \
		openssh-server \
		sudo \
		krb5-kdc \
		krb5-admin-server \
		mariadb-server \
		openjdk-11-jre-headless && \
	rm -rf /var/lib/apt/lists/*  && \
	usermod -aG sudo ${USERNAME} && \
	echo '%sudo ALL=(ALL) NOPASSWD:ALL\nDefaults env_keep += "JAVA_HOME HADOOP_HOME SPARK_HOME HIVE_HOME HBASE_HOME HUE_HOME OOZIE_HOME ZOOKEEPER_HOME SQOOP_HOME SOLR_HOME IMPALA_HOME LANG HOME PATH"' >> /etc/sudoers && \
	cd /tmp && \
	curl -fSL https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz | tar -xzf - -C /opt && \
	chown -R ${USERNAME}:${USERNAME} ${HADOOP_HOME} && \
	curl -fSL https://downloads.apache.org/spark/spark-3.5.5/spark-3.5.5-bin-without-hadoop.tgz | tar -xzf - -C /opt && \
	chown -R ${USERNAME}:${USERNAME} ${SPARK_HOME} && \
	curl -fSL https://archive.apache.org/dist/hive/hive-3.1.3/apache-hive-3.1.3-bin.tar.gz | tar -xzf - -C /opt && \
	chown -R ${USERNAME}:${USERNAME} ${HIVE_HOME} && \
	curl -fSL https://downloads.apache.org/hbase/2.4.18/hbase-2.4.18-bin.tar.gz | tar -xzf - -C /opt && \
	chown -R ${USERNAME}:${USERNAME} ${HBASE_HOME} && \
	curl -fSL https://cdn.gethue.com/downloads/hue-4.11.0.tgz | tar -xzf - -C /opt && \
	chown -R ${USERNAME}:${USERNAME} ${HUE_HOME} && \
	curl -fSL https://archive.apache.org/dist/oozie/5.2.1/oozie-5.2.1.tar.gz | tar -xzf - -C /opt && \
	chown -R ${USERNAME}:${USERNAME} ${OOZIE_HOME} && \
	curl -fSL https://downloads.apache.org/zookeeper/zookeeper-3.8.4/apache-zookeeper-3.8.4-bin.tar.gz | tar -xzf - -C /opt && \
	chown -R ${USERNAME}:${USERNAME} ${ZOOKEEPER_HOME} && \
	curl -fSL https://archive.apache.org/dist/sqoop/1.4.7/sqoop-1.4.7.tar.gz | tar -xzf - -C /opt && \
	chown -R ${USERNAME}:${USERNAME} ${SQOOP_HOME} && \
	curl -fSL https://downloads.apache.org/solr/solr/9.8.1/solr-9.8.1-slim.tgz | tar -xzf - -C /opt && \
	chown -R ${USERNAME}:${USERNAME} ${SOLR_HOME} && \
	curl -fSL https://downloads.apache.org/impala/4.4.1/apache-impala-4.4.1.tar.gz | tar -xzf - -C /opt && \
	chown -R ${USERNAME}:${USERNAME} ${IMPALA_HOME} && \
	rm -rf /tmp/* /root/.cache && \
	echo "${USERNAME}:password" | chpasswd && \
	mkdir /var/run/sshd && \
	sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
	sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
	echo '#!/bin/bash\nhdfs namenode -format -force &&\nsudo service ssh start &&\nstart-dfs.sh &&\nstart-yarn.sh &&\nexec /bin/bash -l' > /usr/bin/startup.sh && \
	chmod +x /usr/bin/startup.sh && \
	mkdir -p /var/run/sshd && \
	ssh-keygen -A && \
	mkdir -p /hadoop/dfs/name && \
	mkdir -p /hadoop/dfs/data && \
	chown -R ${USERNAME}:${USERNAME} /hadoop && \
	sed -i 's?<configuration>?<configuration>\n\t<property>\n\t\t<name>fs.defaultFS</name>\n\t\t<value>hdfs://localhost:9000</value>\n\t</property>?' ${HADOOP_HOME}/etc/hadoop/core-site.xml && \
	sed -i 's?<configuration>?<configuration>\n\t<property>\n\t\t<name>dfs.replication</name>\n\t\t<value>1</value>\n\t</property>\n\t<property>\n\t\t<name>dfs.namenode.name.dir</name>\n\t\t<value>file:///hadoop/dfs/name</value>\n\t</property>\n\t<property>\n\t\t<name>dfs.datanode.data.dir</name>\n\t\t<value>file:///hadoop/dfs/data</value>\n\t</property>?' ${HADOOP_HOME}/etc/hadoop/hdfs-site.xml && \
	echo "export JAVA_HOME=${JAVA_HOME}" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh && \
	sed -i 's?<configuration>?<configuration>\n\t<property>\n\t\t<name>yarn.resourcemanager.hostname</name>\n\t\t<value>localhost</value>\n\t</property>\n\t<property>\n\t\t<name>yarn.nodemanager.aux-services</name>\n\t\t<value>mapreduce_shuffle</value>\n\t</property>\n\t<property>\n\t\t<name>yarn.nodemanager.aux-services.mapreduce_shuffle.class</name>\n\t\t<value>org.apache.hadoop.mapred.ShuffleHandler</value>\n\t</property>\n\t<property>\n\t\t<name>yarn.nodemanager.resource.memory-mb</name>\n\t\t<value>4096</value>\n\t</property>?' ${HADOOP_HOME}/etc/hadoop/yarn-site.xml && \
	sed -i 's?<configuration>?<configuration>\n\t<property>\n\t\t<name>mapreduce.framework.name</name>\n\t\t<value>yarn</value>\n\t</property>\n\t<property>\n\t\t<name>mapreduce.map.memory.mb</name>\n\t\t<value>1024</value>\n\t</property>\n\t<property>\n\t\t<name>mapreduce.reduce.memory.mb</name>\n\t\t<value>1024</value>\n\t</property><property>\n\t\t<name>yarn.app.mapreduce.am.env</name>\n\t\t<value>HADOOP_MAPRED_HOME=${HADOOP_HOME}</value>\n\t</property><property>\n\t\t<name>mapreduce.map.env</name>\n\t\t<value>HADOOP_MAPRED_HOME=${HADOOP_HOME}</value>\n\t</property><property>\n\t\t<name>mapreduce.reduce.env</name>\n\t\t<value>HADOOP_MAPRED_HOME=${HADOOP_HOME}</value>\n\t</property>?' ${HADOOP_HOME}/etc/hadoop/mapred-site.xml

EXPOSE 22

WORKDIR ${HOME}

USER ${USERNAME}

RUN ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && \
	cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
	chmod 700 ~/.ssh && \
	chmod 600 ~/.ssh/authorized_keys

ENTRYPOINT ["/usr/bin/startup.sh"]