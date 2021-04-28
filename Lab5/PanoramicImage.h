#pragma once
#include "panoramic_utils.h"
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <utility> 

using namespace std;
using namespace cv;

class PanoramicImage : public PanoramicUtils
{
public:PanoramicImage();
	   void addImage(const string& path, vector<String>& fn, vector<Mat>& data);
	   void cylProject(PanoramicImage*, vector<Mat>& data);
	   Mat cal_homography_matrix(Mat image1, Mat image2, int features, int layers);
	   pair<double, double> getDistances(Mat image1, Mat image2, int features, int layers);
	   Mat crop_image(Mat result);
	   Mat stitch_image(Mat image1, Mat image2, Mat H);	  
};

