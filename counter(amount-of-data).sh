#!/bin/bash

# Set name of the network interface
NIC='eth0'

# Set monitoring port
PORT='7077'

# Full pathes to the programs required
SOFT='/usr/bin/sudo /opt/spark/bin/spark-shell --master spark://192.168.1.115:7077'
IPTABLES='/usr/bin/sudo /sbin/iptables'
GREP='/bin/grep'
AWK='/usr/bin/awk'

# Make iptables accounting rules
$IPTABLES -F
$IPTABLES -A INPUT -i $NIC -p tcp -m tcp --sport $PORT -j ACCEPT
$IPTABLES -A OUTPUT -o $NIC -p tcp -m tcp --dport $PORT -j ACCEPT

$SOFT

# Gather collected traffic data
echo
echo Traffic at interface $NIC and $PORT is:
$IPTABLES -nvxL | $GREP $NIC | $GREP $PORT | $AWK '{if (NR==1) print "  - incoming: " $2 " Bytes";}'
$IPTABLES -nvxL | $GREP $NIC | $GREP $PORT | $AWK '{if (NR==2) print "  - outgoing: " $2 " Bytes";}'
echo
