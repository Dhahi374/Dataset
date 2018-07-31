function processOutput {
                read VAR1 < t.txt
		read  VAR2 < a.txt
		echo "$VAR2,   $VAR1" >> out.csv
}

# 1st remove files from last run if any
rm -rf a.txt t.txt out.csv
# we want only real time
TIMEFORMAT=%R
for i in seq 1 30;
do
(time sudo perf_3.18 stat -B /opt/spark/bin/spark-shell --master spark://192.168.1.105:7077 -i TestScala.scala 2>&1 | grep "insns per cycle" | awk '{print $1;}'  >a.txt ) > t.txt 2>&1
processOutput
done
echo "Two Nodes Size Done"

ssh pi@192.168.1.116 sudo /opt/spark/sbin/start-slave.sh 192.168.1.105:7077
ssh pi@192.168.1.116 sudo /opt/spark/sbin/start-slave.sh 192.168.1.105:7077

echo "New Node has been added"

for i in seq 1 30;
do
(time sudo perf_3.18 stat -B /opt/spark/bin/spark-shell --master spark://192.168.1.105:7077 -i TestScala.scala 2>&1 | grep "insns per cycle" | awk '{print $1;}'  >a.txt ) > t.txt 2>&1
processOutput
done
echo "Three Nodes Size Done"
ssh pi@192.168.1.121 sudo /opt/spark/sbin/start-slave.sh 192.168.1.105:7077
ssh pi@192.168.1.121 sudo /opt/spark/sbin/start-slave.sh 192.168.1.105:7077

echo "New Node has been added"

