#include <iostream>
#include "NumberDetector.h"

//
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <filesystem>
//

using namespace std;

vector<string> get_filepaths_in_directory(string path);

int main() {
	// Mat input = imread("D:\\Users\\VisionWork\\Documents\\카카오톡 받은 파일\\KakaoTalk_20190214_154149171.jpg");
	//Mat input = imread("D:\\Users\\VisionWork\\Desktop\\박하연\\Projects\\TrainingDataGenerator\\YOLODataRefiner\\YOLODataRefiner\\Labels\\16_speed_50.jpg");
	NumberDetector numberDetector(NETWORK_PATH);
	
	vector<string> path = get_filepaths_in_directory("D:\\Users\\VisionWork\\Desktop\\박하연\\DB\\test");
	vector<string> image_path,txt_path;


	for (auto it = path.begin(); it != path.end(); ++it) {
		string filename = *it;
		istringstream ss(filename);
		while (getline(ss, filename, '\\')) {}

		if (filename == "train.txt") continue;

		ss = istringstream(filename);
		while (getline(ss, filename, '.')) {}

		if (filename == "txt") txt_path.push_back(*it);
		else image_path.push_back(*it);

	}

	for (int i = 0; i < image_path.size(); i++) {
		Mat src, roi;

		src = imread(image_path[i]);

		ifstream in(txt_path[i]);
		string input_s;
		Rect location;

		while (!in.eof()) {
			cout << "!" << endl;
			getline(in, input_s);
			stringstream ss(input_s);

			string value;
			int label;
			double x_center, y_center, width, height;

			ss.seekg(0, ios::end);
			if (ss.tellg() <= 1) break;
			ss.seekg(0, 0);

			ss >> label;
			ss >> x_center;
			ss >> y_center;
			ss >> width;
			ss >> height;
			
			 location = Rect(src.cols * (x_center - 0.5*width), src.rows* (y_center - 0.5*height),
				 src.cols * width, src.rows * height);
		}

		roi = src(location);
		numberDetector.detect(roi);
	
	}

	return 0;
}

vector<string> get_filepaths_in_directory(string path) {
	vector<string> paths;

	for (auto& p1 : std::experimental::filesystem::directory_iterator(path)) {
		paths.push_back(p1.path().string());
	}

	return paths;
}