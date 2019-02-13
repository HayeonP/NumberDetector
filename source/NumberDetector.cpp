#include "NumberDetector.h"

int NumberDetector::annPredict(IplImage* roi)
{
	int result_num = 0;

	Mat classificationResult(1, 10, CV_32FC1);
	Point max_loc = { 0, 0 };

	Mat test(1, 28 * 38, CV_32FC1);
	Mat test_classification = Mat::zeros(1, 10, CV_32FC1);

	int correct_class = 0;
	int wrong_class = 0;
	int false_positives[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	for (int i = 0; i < roi->height; i++)
	{
		for (int j = 0; j < roi->width; j++)
		{
			int p = (unsigned char)(roi->imageData[i * roi->widthStep + j]);
			test.at<float>(0, i * roi->widthStep + j) = p;
			/*cout << "P : " << p << endl;
			cout << "idx : " << i * roi->widthStep + j << endl;
			cout << "test : " << test.at<float>(0, i * roi->widthStep + j)<<endl;*/
		}
	}

	Mat test_sample = test.clone();

	nnetwork->predict(test_sample, classificationResult);

	for (int i = 0; i < 10; i++)
	{
		float pro = classificationResult.at<float>(0, i);

		printf("%d\t------ : %f\n", i, pro);
	}

	minMaxLoc(classificationResult, NULL, NULL, NULL, &max_loc);

	result_num = max_loc.x;
	printf("\t result_num = %d\n\n", result_num);

	test.release();
	classificationResult.release();

	return result_num;
}


void NumberDetector::BubbleSorting(int num[3][2])
{
	int temp = 0;
	int temp2 = 0;

	for (int i = 0; i < digit; i++)
	{
		for (int j = 0; j < digit - i - 1; j++)
		{
			if (num[j][0] > num[j + 1][0])
			{
				temp = num[j][0];
				temp2 = num[j][1];
				num[j][0] = num[j + 1][0];
				num[j][1] = num[j + 1][1];
				num[j + 1][0] = temp;
				num[j + 1][1] = temp2;
			}
		}
	}
}

NumberDetector::NumberDetector(string network_path){
	nnetwork = ml::ANN_MLP::load(network_path);
}

NumberDetector::NumberDetector(string network_path, int flag) {
	nnetwork = ml::ANN_MLP::load(network_path);
	numberDetectDebug = flag;
}


int NumberDetector::detect(Mat input)
{
	int speed = 0;
	int low_limit = 0;
	IplImage *image = &IplImage(input);

	bool low_lim = false;
	bool flg = false;

	IplImage* gray = cvCreateImage(cvGetSize(image), 8, 1);
	IplImage* resize = cvCreateImage(cvSize(110, 110), 8, 1);
	IplImage* binary = cvCreateImage(cvGetSize(resize), 8, 1);
	cvSetZero(binary);

	cvCvtColor(image, gray, CV_BGR2GRAY);
	cvResize(gray, resize);
	cvAdaptiveThreshold(resize, binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 21, 10);

	CBlobLabeling label;
	label.SetParam(binary, 50);
	label.DoLabeling();

	digit = 0;

	for (int i = 0; i < 3; i++)
	{
		resultNum[i][0] = 0;
		resultNum[i][1] = 0;
	}

	IplImage* color = cvCreateImage(cvGetSize(binary), 8, 3);
	cvCvtColor(binary, color, CV_GRAY2BGR);
	if (numberDetectDebug) printf("Number of lables : %d\n ", label.m_nBlobs);
	for (int i = 0; i < label.m_nBlobs; i++)
	{
		sprintf(_numName5, "tem/im_%d.bmp", i);

		cvRectangle(color, cvPoint(label.m_recBlobs[i].x, label.m_recBlobs[i].y), cvPoint(label.m_recBlobs[i].x + label.m_recBlobs[i].width, label.m_recBlobs[i].y + label.m_recBlobs[i].height), cvScalar(0, 255, 255), 1, 8, 0);
		printf(" width = %d \\ height = %d\n", label.m_recBlobs[i].width, label.m_recBlobs[i].height);
		printf(" x = %d \\ y = %d\n", label.m_recBlobs[i].x, label.m_recBlobs[i].y);

		if (label.m_recBlobs[i].x > 10 && label.m_recBlobs[i].y > 10 && ((label.m_recBlobs[i].x + label.m_recBlobs[i].width) < resize->width - 10) && ((label.m_recBlobs[i].y + label.m_recBlobs[i].height) < resize->height - 10))
		{
			if ((label.m_recBlobs[i].width >= 5) && (label.m_recBlobs[i].width < 40) && (label.m_recBlobs[i].height >= 25) && (label.m_recBlobs[i].height < 80))
			{
				printf("#2");
				if ((label.m_recBlobs[i].width <= label.m_recBlobs[i].height) && (label.m_recBlobs[i].width * 4 > label.m_recBlobs[i].height))
				{
					printf("#3");
					if ((label.m_recBlobs[i].height - label.m_recBlobs[i + 1].height) < 8 || (label.m_recBlobs[i].height - label.m_recBlobs[i + 1].height) > -8)
					{
						printf("#4");
						if (label.m_recBlobs[i].x < label.m_recBlobs[i + 1].x)
						{
							printf("#5");
							if (label.m_recBlobs[i].x + label.m_recBlobs[i].width < label.m_recBlobs[i + 1].x)
							{
								printf("#6");
								cvRectangle(color, cvPoint(label.m_recBlobs[i].x, label.m_recBlobs[i].y), cvPoint(label.m_recBlobs[i].x + label.m_recBlobs[i].width, label.m_recBlobs[i].y + label.m_recBlobs[i].height), cvScalar(0, 255, 0), 1, 8, 0);

								flg = true;
							}
						}

						if (label.m_recBlobs[i + 1].x < label.m_recBlobs[i].x)
						{
							printf("#-5");
							if (label.m_recBlobs[i + 1].x + label.m_recBlobs[i + 1].width < label.m_recBlobs[i].x)
							{
								printf("#6");
								cvRectangle(color, cvPoint(label.m_recBlobs[i].x, label.m_recBlobs[i].y), cvPoint(label.m_recBlobs[i].x + label.m_recBlobs[i].width, label.m_recBlobs[i].y + label.m_recBlobs[i].height), cvScalar(0, 255, 0), 1, 8, 0);

								flg = true;
							}
						}

						printf("\n");

						if (flg == true)
						{
							int matchingNum = 0;

							IplImage* roi_temp2 = cvCreateImage(cvSize(28, 38), 8, 1);
							IplImage* roi_temp3 = cvCreateImage(cvSize(label.m_recBlobs[i].width, label.m_recBlobs[i].height), 8, 1);
							IplImage* roi_temp4 = cvCreateImage(cvSize(18, 28), 8, 1);

							cvSetImageROI(binary, label.m_recBlobs[i]);
							cvCopy(binary, roi_temp3);

							cvResize(roi_temp3, roi_temp4);

							cvCopyMakeBorder(roi_temp4, roi_temp2, cvPoint(5, 5), IPL_BORDER_CONSTANT, CV_RGB(0, 0, 0));
							cvThreshold(roi_temp2, roi_temp2, 50, 255, CV_THRESH_BINARY);

							cvShowImage("roi_temp2", roi_temp2);
							cvSaveImage(_numName5, roi_temp2);
							matchingNum = annPredict(roi_temp2);

							cvReleaseImage(&roi_temp3);
							cvReleaseImage(&roi_temp2);
							cvReleaseImage(&roi_temp4);

							if (digit < 3)
							{
								resultNum[digit][0] = label.m_recBlobs[i].x;
								resultNum[digit][1] = matchingNum;
								digit++;
							}
						}
					}
				}
			}

			if (label.m_recBlobs[i].width >= 36 && label.m_recBlobs[i].height < 20)
			{
				low_limit = -1;
				low_lim = true;
			}
		}
	}

	BubbleSorting(resultNum);

	if (digit == 3)			//技磊府老 锭
	{
		if (resultNum[0][1] == 1 && resultNum[2][1] == 0)
		{
			speed = 100 + resultNum[1][1] * 10;
		}
	}
	else if (digit == 2)	//滴磊府老 锭
	{
		if (resultNum[0][1] == 1)
		{
		}
		if (resultNum[1][1] == 0)
		{
			speed = resultNum[0][1] * 10;
		}
	}
	else
	{
	}

	if (low_limit < 0)
	{
		speed = speed*(-1);
	}

	if (numberDetectDebug) cvShowImage("color", color);
						
	cvReleaseImage(&color);
	cvReleaseImage(&gray);
	cvReleaseImage(&binary);
	cvReleaseImage(&resize);

	if(numberDetectDebug) cvShowImage("image", image);
	if (numberDetectDebug) waitKey();

	return speed;
}


