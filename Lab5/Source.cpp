#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "opencv2/calib3d.hpp"
#include "panoramic_utils.h"
#include "PanoramicImage.h"
#include <utility>

using namespace std;
using namespace cv;

int main() {
	PanoramicImage *pi = new PanoramicImage();
	vector<String> fn;
	vector<Mat> data;
	pi->addImage("C:/Users/albma/Desktop/Università/Computer Vision/labs/lab5/dolomites/*png", fn, data);

	pi->cylProject(pi, data);

	for (int i = 0; i < data.size(); i++)
		cvtColor(data[i], data[i], COLOR_BGR2GRAY);

	int a = 0;
	int c = 0;

	vector<Mat> homo;

	for (int b = 1; b <= data.size() - 1; b++) {
		Mat H = pi->cal_homography_matrix(data[a], data[b], 2000, 3);
		homo.push_back(H);
		a++;
	}

	int g = 0;
	vector<pair<double, double>> p;
	for (int b=1; b<=data.size()-1;b++) {
		p.push_back(pi->getDistances(data[g], data[b], 2000, 3));
		g++;
	}

	//getting all the info about the first image as a test
	int columns = data[0].cols;
	int rows = data[0].rows;
	for (int i = 0; i < p.size(); i++) {
		columns = columns + p[i].first;
		rows = rows + p[i].second;
	}

	int h = 0;
	vector<Mat> stitched;
	for (int b = 1; b <= data.size() - 1; b++) {
		stitched.push_back(pi->crop_image(pi->stitch_image(data[h], data[b], homo[h])));
	}

	//test: I first stich the first 2 images, then I add the others in the next for loop
	Mat F = Mat(stitched[0].rows, columns, CV_8UC1);

	stitched[0].copyTo(pi->crop_image(F.rowRange(0, stitched[0].rows).colRange(0, stitched[0].cols)));

	int start_distance = 0;

	for (int i = 1; i < stitched.size();i++) {
		start_distance = start_distance + p[i - 1].first;
		stitched[i].copyTo(F.rowRange(0, stitched[i].rows).colRange(start_distance, start_distance + stitched[i].cols));
	}

	imshow("FINE", F);

	waitKey(0);
	return 0;
}