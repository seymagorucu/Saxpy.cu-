# Cuda-Parallel-programming-
Create appropriate number of threads and blocks based on the N number entered


We have up to 1024 threads and 2097151 blocks for the x dimension. I tried to create the most appropriate number of threads and blocks if the number was not more than the maximum number of threads according to the number of N entered.


If N is less than or equal to 1024, we only use one block.

![image](https://user-images.githubusercontent.com/35656598/145712231-d9b261ce-f3e0-4095-8e07-4f0937a106fa.png)

If we enter 1025, it creates 2 block  513 threads instead of 2 block 1024 threads.
![image](https://user-images.githubusercontent.com/35656598/145712245-d09cb2be-c68e-4d36-bf65-7946621d7180.png)


The maximum number of threads is 1024, but when we enter 4000, I create 1000 threads in each block, avoiding extra threads.

![image](https://user-images.githubusercontent.com/35656598/145712263-444e86c9-e966-44a5-a3bb-52b01670fc79.png)

1019 * 23 = 23437   Our N valu is 23432 thus, the most appropriate number of threads was formed within N number.

![image](https://user-images.githubusercontent.com/35656598/145712281-8e53e8d6-ce39-4eee-8dc5-2890d5a02ffb.png)
