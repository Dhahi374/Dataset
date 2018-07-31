#! /bin/bash
echo "Hi This Is My Bash Script to Login into pi2 and to run DT benchmark for MPI Appicton"
ssh pi@192.168.1.7 'sudo perf_3.18 record -e instructions -a -o test2.dat &'  # run the profiler on cloud server (Raspberry Pi) an$
ssh pi@192.168.1.7 mpirun -f machinefile -n 40 ~/NPB3.3/NPB3.3-MPI/bin/dt.A BH
#ab -n 1000 -c 5 http://130.209.247.51/index.php  # run this benchmark on client (Ubuntu PC)
echo "MPI Job for the benchmrk DT is DONE .."
# PERF_ID=`ssh ... ` # find the Process Id of perf
ssh pi@192.168.1.7 PERF_ID=`ps aux | grep perf_3 | awk '{print $2;'}`
echo "PROCESSOR ID HAS BEEN GATHERED :"
ssh pi@192.168.1.7 echo $PERF_ID
ssh pi@192.168.1.7 sudo kill $PERF_ID 
echo "A PERF_ID HAS BEEN KILLED ..."
ssh pi@192.168.1.7 'sudo perf_3.18 report -i test2.dat | head -n 5 | grep Event'
# perf report - but use --stdio and head and grep so you only show the total number of events
sleep 2
echo "Thank you"
exit


