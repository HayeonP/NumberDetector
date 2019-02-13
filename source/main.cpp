#include <iostream>
#include "NumberDetector.h"

#define NETWORK_PATH "annfileishere2.yml"
#define NUMBER_DETECT_FLAG 0

using namespace std;

int main() {
	Mat input = imread("D:\\Hayeon\\DB\\190122_b\\190122_b_774.jpg");
	NumberDetector numberDetector(NETWORK_PATH, NUMBER_DETECT_FLAG);
	int speed = numberDetector.detect(input);
	cout << speed << endl;

	return 0;
}

