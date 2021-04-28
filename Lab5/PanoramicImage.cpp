#include "PanoramicImage.h"

PanoramicImage::PanoramicImage() :PanoramicUtils() {}

void PanoramicImage::addImage(const string & path, vector<String>& fn, vector<Mat>& data)
{
	glob(path, fn, true);
	for (size_t k = 0; k < fn.size(); ++k)
	{
		Mat im = imread(fn[k]);
		if (im.empty()) continue; //only proceed if sucsessful
		data.push_back(im);
	}
}
void PanoramicImage::cylProject(PanoramicImage* pu, vector<Mat>& data)
{
	for (int i = 0; i < data.size(); i++)
		pu->cylindricalProj(data[i], 27);
}

Mat PanoramicImage::cal_homography_matrix(Mat image1, Mat image2, int features, int layers) {

	Ptr<ORB> detector = ORB::create(features);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector->detect(image1, keypoints_1);
	detector->detect(image2, keypoints_2);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(image1, keypoints_1, img_keypoints_1, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(image2, keypoints_2, img_keypoints_2, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//-- Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	detector->compute(image1, keypoints_1, descriptors_1);
	detector->compute(image2, keypoints_2, descriptors_2);


	//-- Step 3: Matching descriptor vectors using BFMatcher :
	BFMatcher matcher;
	std::vector<std::vector<cv::DMatch>> matches;

	//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
	//std::vector< DMatch > good_matches;
	matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches

	vector<cv::DMatch> good_matches;
	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.8; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
	}

	cv::Mat H;
	std::vector< Point2f > obj;
	std::vector< Point2f > scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}


	// Find the Homography Matrix for img 1 and img2
	Mat mask;
	H = findHomography(obj, scene, RANSAC, 3.0, mask);
	//H is the transformation. Using ransac to remove the outliers
	return H;
}

pair<double, double> PanoramicImage::getDistances(Mat image1, Mat image2, int features, int layers)
{
	pair<double, double> p;
	Ptr<ORB> detector = ORB::create(features);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector->detect(image1, keypoints_1);
	detector->detect(image2, keypoints_2);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(image1, keypoints_1, img_keypoints_1, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(image2, keypoints_2, img_keypoints_2, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//-- Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	detector->compute(image1, keypoints_1, descriptors_1);
	detector->compute(image2, keypoints_2, descriptors_2);


	//-- Step 3: Matching descriptor vectors using BFMatcher :
	BFMatcher matcher;
	std::vector<std::vector<cv::DMatch>> matches;

	//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
	//std::vector< DMatch > good_matches;
	matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);  // Find two nearest matches

	vector<cv::DMatch> good_matches;
	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.8; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
	}

	cv::Mat H;
	std::vector< Point2f > obj;
	std::vector< Point2f > scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}


	// Find the Homography Matrix for img 1 and img2
	Mat mask;
	H = findHomography(obj, scene, RANSAC, 3.0, mask);
	double avg_dx = 0;
	double avg_dy = 0;
	int c = 0;

	// Somewhere before I did: findHomography(points1, points2, CV_RANSAC, 3, mask)
	for (int i = 0; i < good_matches.size(); i++)
	{
		// Select only the inliers (mask entry set to 1)
		if ((int)mask.at<uchar>(i, 0) != 0)
		{
			avg_dx += (keypoints_1[good_matches[i].queryIdx].pt.x - keypoints_2[good_matches[i].trainIdx].pt.x);
			avg_dy += (keypoints_1[good_matches[i].queryIdx].pt.y - keypoints_2[good_matches[i].trainIdx].pt.y);
			c++;
		}
	}
	cout << "[" << abs(avg_dx / c) << ", " << avg_dy / c << "]" << endl;

	p.first = avg_dx / c;
	p.second = avg_dy / c;

	return p;
}

Mat PanoramicImage::crop_image(Mat result) {
	// //Finding the largest contour i.e remove the black region from image 

	Mat img_gray = result.clone();
	threshold(img_gray, img_gray, 25, 255, THRESH_BINARY); //Threshold the gray 

	vector<vector<Point> > contours; // Vector for storing contour 
	vector<Vec4i> hierarchy;
	findContours(img_gray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // Find the contours in the image 
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;


	for (int i = 0; i < contours.size(); i++) // iterate through each contour.  
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour 
		if (a > largest_area) {
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour 
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour 

		}

	}
	result = result(Rect(bounding_rect.x, bounding_rect.y, bounding_rect.width, bounding_rect.height));
	return result;

}

Mat PanoramicImage::stitch_image(Mat image1, Mat image2, Mat H)
{
	cv::Mat result;
	cv::warpPerspective(image1, result, H, cv::Size(image2.cols + image1.cols, image1.rows), 3);

	cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
	image2.copyTo(half);

	return result;

}
;