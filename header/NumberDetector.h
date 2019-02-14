#pragma once
#include <opencv2/opencv.hpp>
#include "BlobLabeling.h"

#define NETWORK_PATH "annfileishere2.yml"
#define NUMBER_DETECT_FLAG 0
#define SHOW_FLAG 1

using namespace std;
using namespace cv;

class NumberDetector {
private:
	IplImage *_Number[10];
	IplImage *input_img;
	cv::Ptr<cv::ml::ANN_MLP> nnetwork;
	int resultNum[3][2];
	int digit;

private:	
	int annPredict(IplImage* roi);
	void BubbleSorting(int num[3][2]);

public:
	NumberDetector(string network_path);
	int detect(Mat input);
	
};
