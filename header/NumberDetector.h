#pragma once
#include <opencv2/opencv.hpp>
#include "BlobLabeling.h"


using namespace std;
using namespace cv;

class NumberDetector {
private:
	IplImage *_Number[10];
	IplImage *input_img;
	int numberDetectDebug = 0;
	cv::Ptr<cv::ml::ANN_MLP> nnetwork;
	int resultNum[3][2];
	int digit;
	char _numName5[300];

private:	
	int annPredict(IplImage* roi);
	void BubbleSorting(int num[3][2]);

public:
	NumberDetector(string network_path);
	NumberDetector(string network_path, int flag);
	int NumberDetector::detect(Mat input);
	
};
