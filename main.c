#include<stdio.h>
#include<string.h>
#include<math.h>
#include<stdlib.h>
#include<random>
#include <omp.h>
#include"mkl.h"

#include"timeutils.h"
#include"network_definition.h"
#include"mnist.h"
#include"machinelearning_function.h"

int main(void)
{
	int i;
	int recog =0;
	struct network * net;
	
	net = (struct network *) malloc(sizeof(struct network));//network 를 할당 합니다.

	init(net);
	mnist_load(net);/* reader 함수는 철저하게 mnist 를 중심으로 짜여있는 code입니다. 
				   다른 data set을 원하시면 함수를 새로 만들어 그 함수가 실행 뒤엔
				   network 구조체의 train_q,train_a,test_q,test_a 배열들이

				   전부 원하는 input 데이터로 가득 차있게 만드 십시오.*/

	train(net,net->threads,net->modes);
	return 0;

}
