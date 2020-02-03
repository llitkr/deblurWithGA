// fitness는 float형으로 고정(double이 더 정확하긴 한데 오버헤드 때문에 float으로 함),
// fitness는 낮을 수록 좋은 것
// 이미지를 복사할 때 = 연산자를 쓰면 주소를 공유하게 되고 이미지변수.clone()을 써야 제대로 복사가 됨. 혹은 이미지변수.copyto(새 이미지변수)로 하기

#include <iostream>
#include <opencv2./core/core.hpp>
#include <opencv2./highgui/highgui.hpp>
#include <opencv2./imgproc/imgproc.hpp>
#include <ctime>
#include <string>
#include <fstream>

#define RR 10
#define SOURCE "grad.png"			// 실험용 파일명
#define blurSize 25
#define blurPower 30
#define iniRAN 99					// 새 픽셀 초기화시 기존 픽셀의 블러값에 얼마나 영향받을지에 대한 상수, 100일 경우 거의 무조건 기존 블러값에 따름, 0일 경우 반전, 50일 경우 완전 랜덤
#define PR 10						// 성능평가용 총 라운드

#define GEN 10
#define POP 100						// 왠만하면 짝수로 하기(중간에 i+=2 구간이 있음)
#define CP 3		// 교차 포인트 갯수
#define CR 0.9		// 교차 확률
#define MT (0.01 + ri*0.05)		// 변이 확률
#define EL 0.3		// 엘리티즘을 통해 보존할 해의 비율
#define NP 0.2		// 새로운 세대를 투입할 비율

using namespace cv;
using namespace std;

int makeRandomRange(int range);		// -range ~ +range 사이의 범위 랜덤생성

Mat_<bool> Initailize(Mat_<uchar> source);				// 초기해집단 생성 함수

float getFitness(Mat_<uchar> bluredSource, Mat_<bool> NewImage);		// 해 품질 평가 함수

float getDif(uchar a, uchar b);			// 두 수의 차이를 구하는 함수

float getSqr(float num);				// 제곱 구하기

void sort(Mat_<bool> *&pop, float *&fitness);							// 해 배열 정렬

void swap(Mat_<bool> *a, Mat_<bool> *b);

void swapF(float &a, float &b);

void makeCutRandom(int *&cut, int size);

int cross(Mat_<bool> nA, Mat_<bool> nB, Mat_<bool> oA, Mat_<bool> oB);

int tour(float *&fitness, float val);						// 토너먼트 선택(전체 품질 배열과 선택 품질 인자

void mutation(Mat_<bool> image, Mat_<uchar> bluredImage, int i, int j);						// 변이 함수

bool makeBW(uchar pixel);

Mat_<uchar> makeGray(Mat_<bool> image);

int ri = 0;		// 라운드수를 더할 때 쓰는 변수

int main(void)
{
	//		파일 가져오기 단계
	ofstream fout;
//	fout.open("result.txt");
//	if (!fout.fail())
//		cout << "실패!" << endl;

	Mat source;
	int i = 0, j = 0, k=0;
	Mat_<bool> *population;
	Mat_<bool> *newPopulation;
	
	// 랜덤 함수를 위한 선작업

	srand(time(NULL));

	// 이미지 파일 불러오기
	source = imread(SOURCE);
	float *fitness;
	float *newFitness;

	fitness = new float[POP];
	newFitness = new float[POP];

	// 이미지 파일 체크
	if(!source.data)
	{
		cout << "이미지 파일을 불러올 수 없습니다.";
		return -1;
	}

	Mat_<uchar> BWSource(source.rows, source.cols);

	for (i = 0; i < source.rows; i++)
		for (j = 0; j < source.cols; j++)
			if (source.at<cv::Vec3b>(i, j)[0] > 128)
				BWSource(i, j) = 255;
			else
				BWSource(i, j) = 0;

//	새_이미지 = source;

//	imshow("원본 이미지", graySource);

//	waitKey(0);

//	Mat test;

//	GaussianBlur(source, source, Size(5, 5), 50);
	imwrite("OC\\BW0(원본 흑백).bmp", BWSource);
	GaussianBlur(BWSource, BWSource, Size(blurSize, blurSize), blurPower);
	float maxFit = FLT_MAX;
	float minFit = -1;
	int round = 0;
	float everFit = 0;
	Mat_<bool> bestFit;
	float ChBest = 0;		// 베스트핏이 바뀌었는지 확인하는 변수

//	for (i = 0; i < source.rows; i++)
//		for (j = 0; j < source.cols; j++)
//			cout << "(" << i << ", " << j << ") = " << getDif(source.at<cv::Vec3b>(i, j)[0], test.at<cv::Vec3b>(i, j)[0]) << "\t";

	imwrite("OC\\BW0(원본 블러).bmp", BWSource);
	while (ri < PR)
	{
		population = new Mat_<bool>[POP];
		newPopulation = new Mat_<bool>[POP];
		maxFit = FLT_MAX;
		minFit = -1;
		round = 0;
		everFit = 0;
		
		ChBest = 0;		// 베스트핏이 바뀌었는지 확인하는 변수

		for (i = 0; i < POP; i++)
		{
			population[i] = Initailize(BWSource);
//			cout << i + 1 << "번째 이미지 생성" << endl;
			fitness[i] = getFitness(BWSource, population[i]);
//			cout << "그림의 품질 : " << fitness[i] << endl;
			if (fitness[i] < maxFit)
			{
				maxFit = fitness[i];
				bestFit = population[i].clone();
			}

			if (fitness[i] > minFit)
				minFit = fitness[i];
			//		waitKey(0);
		}

		sort(population, fitness);

		for (i = 0; i < POP; i++)
			newPopulation[i] = population[i].clone();

		while (round++ < GEN)			// 총 세대수만큼 반복
		{
			//		cout << round << "번째 라운드*********************" << endl;

			//		*******************엘리티즘을 위한 정렬*******************

			sort(population, fitness);
			

			//		*******************교차*******************
			
			everFit = 0;

			int pe = POP * EL;

			if (pe % 2 == 1)
				pe--;
			for (i = 0; i < pe; i++)
			{
				newPopulation[i] = population[i].clone();
				//			imshow("디스플레이", newPopulation[i]);
				//			imshow("디스플레이2", newPopulation[i + 1]);
				//			waitKey(0);
				newFitness[i] = fitness[i];
			}
			
			for (i = pe; i < POP; i += 2)
			{
				j = tour(fitness, fitness[i]);
				k = tour(fitness, fitness[i + 1]);
				
				if (!j)
					j = i;
				if (!k)
					k = i + 1;
				if (rand() % 100 <= CR * 100)
				{
					cross(newPopulation[i], newPopulation[i + 1], population[j], population[k]);
				}
				else
				{
					
					newPopulation[i] = population[k].clone();
					newPopulation[i + 1] = population[j].clone();
					
				}
			}
			

			//		*******************변이*******************

			for (i = POP * EL; i < POP; i++)
			{
				//			cout << i+1 << "번째 개체 처리중" << endl;
				for (j = 0; j < BWSource.rows; j++)
				{
					for (k = 0; k < BWSource.cols; k++)
					{
						if (rand() % 100 <= MT * 100)
							mutation(newPopulation[i], BWSource, j, k);
						else
						{
							if (k == 0)
							{
								if (newPopulation[i](j, k) != newPopulation[i](j, BWSource.cols - 1))
									mutation(newPopulation[i], BWSource, j, k);
							}
						}
					}
				}
				newFitness[i] = getFitness(BWSource, newPopulation[i]);
				if (newFitness[i] < maxFit)
				{
					maxFit = newFitness[i];
					bestFit = newPopulation[i].clone();
				}
			}

			sort(newPopulation, newFitness);

			//		*******************새 인구 투입*******************

			for (i = POP * (1 - NP); i < POP; i++)
			{
				newPopulation[i] = Initailize(BWSource);
				newFitness[i] = getFitness(BWSource, newPopulation[i]);
			}

			for (i = 0; i < POP; i++)
				everFit += newFitness[i];

			//		*******************새 세대로 옮기기*******************
			for (i = 0; i < POP; i++)
			{
				population[i] = newPopulation[i].clone();
				fitness[i] = newFitness[i];
			}
			if (maxFit != ChBest)
			{
				cout << round << "번 세대의 평균 품질 : " << everFit / POP << ", \t최대 품질 : " << maxFit << "\t\t";
				cout << "(바뀜)" << endl;
				string save = "OC\\BW";
				ChBest = maxFit;
				save.append(to_string(round));
				save.append("_");
				save.append(to_string((int)maxFit));
				save.append(".bmp");
				imwrite(save, makeGray(bestFit));
				//			imshow("디스플레이", makeGray(bestFit));
				//			waitKey(0);
			}
			else
			{
				cout << round << "번 세대의 평균 품질 : " << everFit / POP << ", \t최대 품질 : " << maxFit << "\t\t";
				cout << "\r" << flush;
			}

			//		for (i = 0; i < POP; i++)
			//			population[i] = Initailize(BWSource);
			//		for (i = 0; i < POP; i++)
			//			fitness[i] = getFitness(BWSource, population[i]);

			//		maxFit = FLT_MAX;
			//		cout << "saveing :: " << save << endl;
			//		imshow("이미지", bestFit);
			//		waitKey(0);
		}

		cout << "최대 품질 : " << maxFit << ", 최소 품질 : " << minFit << endl;

		fout << ri + 1 << "번째 최댓값(" << MT << ") : " << maxFit << endl;

		cout << "최대 품질 이미지 디스플레이" << endl;

//		imshow("베스트샷", bestFit);
		//	imwrite("C:\\Users\\YungHee Lee\\Pictures\\OC\\Cyaron.jpg", bestFit);
		cout << "쓰기 완료" << endl;
		// 키 입력 기다리기
		delete []population;
		delete []newPopulation;
		ri++;
	}

	return 0;
}

int makeRandomRange(int range)
{
	int result = rand() % range;
	result -= range / 2;
	return result;
}

Mat_<bool> Initailize(Mat_<uchar> source)
{
	int i, j;
	Mat_<bool> newImage(source.rows, source.cols);

	for (i = 0; i < source.rows; i++)
		for (j = 0; j < source.cols; j++)
		{
			int rn = rand() % 100;
			if (source(i, j)>128)
			{
				if (iniRAN > rn)
					newImage(i, j) = 1;
				else
					newImage(i, j) = 0;
			}
			else
			{
				if (iniRAN > rn)
					newImage(i, j) = 0;
				else
					newImage(i, j) = 1;
			}
//			int rn = rand() % 2;
//			newImage(i, j) = rn % 2;
		}

	return newImage;
}

float getFitness(Mat_<uchar> bluredSource, Mat_<bool> NewImage)
{
	int i, j;
	float fitness = 0;
	Mat_<uchar> bluredNewImage(NewImage.rows, NewImage.cols);
	
	for (i = 0; i < NewImage.rows; i++)
		for (j = 0; j < NewImage.cols; j++)
			if (NewImage(i, j))
				bluredNewImage(i, j) = 255;
			else
				bluredNewImage(i, j) = 0;

	GaussianBlur(bluredNewImage, bluredNewImage, Size(blurSize, blurSize), blurPower);

	for (i = 0; i < bluredSource.rows; i++)
		for (j = 0; j < bluredSource.cols; j++)
			fitness += getSqr(getDif(bluredNewImage(i, j), bluredSource(i, j)));
	return (fitness * 100) / (bluredSource.rows * bluredSource.cols);
}

float getDif(uchar a, uchar b)
{
	if (a >= b)
		return a - b;
	else
		return b - a;
}

float getSqr(float num)
{
	return num*num;
}

void sort(Mat_<bool> *&pop, float *&fitness)
{
	int i = 0, j = 0;
	float max;
	int maxn;
	for (i = 0; i < POP - 1; i++)
	{
		max = 999999999;
		maxn = -1;
		for (j = i + 1; j < POP; j++)
		{
			if (max > fitness[j])
			{
				max = fitness[j];
				maxn = j;
			}
		}
		if (maxn == -1)
			continue;
		swap(pop[i], pop[maxn]);
		swapF(fitness[i], fitness[maxn]);
	}
}

void swap(Mat_<uchar> *a, Mat_<uchar> *b)
{
	Mat_<bool> temp = (*a).clone();
	*a = (*b).clone();
	*b = temp.clone();

	return;
}

void swapF(float &a, float &b)
{
	float temp = a;
	a = b;
	b = temp;

	return;
}

void makeCutRandom(int *&cut, int size)
{
	cut = new int[CP + 1];
	int i = 0, j = 0;
	for (i = 0; i <= CP; i++)
	{
		cut[i] = rand() * rand() % size + 1;
		for (j = 0; j<i; j++)
			if (cut[i] == cut[j])
			{
				i--;
				break;
			}
	}
	for (i = 0; i < CP - 1; i++)
	{
		int minN = -1;
		int min = INT_MAX;
		for (j = i; j<CP; j++)
		{
			if (cut[j] < min)
			{
				minN = j;
				min = cut[j];
			}
		}
		int temp = cut[i];
		cut[i] = cut[minN];
		cut[minN] = temp;
	}

	return;
}

int cross(Mat_<bool> nA, Mat_<bool> nB, Mat_<bool> oA, Mat_<bool> oB)
{
	int *cut;
	int i = 0, j = 0, k = 0, m = 0;
	
	makeCutRandom(cut, oA.rows * oA.cols);

	for (i = 0; i < oA.rows; i++)
	{
		for (j = 0; j < oA.cols; j++)
		{
			if (k % 2)
			{
				nA(i, j) = oB(i, j);
				nB(i, j) = oA(i, j);
			}
			else
			{
				nA(i, j) = oA(i, j);
				nB(i, j) = oB(i, j);
			}
			if (cut[k] == m)
				k++;
			m++;
		}
	}

	delete []cut;

	return 0;
}

int tour(float *&fitness, float val)
{
	int t = 0, i = 0;

	t = rand() % POP;

	if (fitness[t] < val)
		return t;
	else
		return 0;
}

void mutation(Mat_<bool> image, Mat_<uchar> bluredImage, int i, int j)
{
	int rn = rand() % 255;
	if (bluredImage(i, j)>rn)
	{
		image(i, j) = 1;
	}
	else
	{
		image(i, j) = 0;
	}
}

bool makeBW(uchar pixel)
{
	if (pixel > 128)
		return 1;
	else
		return 0;
}

Mat_<uchar> makeGray(Mat_<bool> image)
{
	Mat_<uchar> newImage(image.rows, image.cols);
	int i, j;

	for (i = 0; i < image.rows; i++)
		for (j = 0; j < image.cols; j++)
			if (image(i, j))
				newImage(i, j) = 255;
			else
				newImage(i, j) = 0;

	return newImage;
}