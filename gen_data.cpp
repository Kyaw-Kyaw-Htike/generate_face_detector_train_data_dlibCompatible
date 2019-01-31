// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "opencv2/opencv.hpp"
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include "dlib/image_processing/frontal_face_detector.h"
#include <dlib/opencv/cv_image.h>
#include <QtCore/QDirIterator>
#include <QtCore/QDir>

using namespace std;

dlib::rectangle enlarge_rect(const dlib::rectangle& r, double ntimes)
{
	float x = r.top();
	float y = r.left();
	float w = r.width();
	float h = r.height();

	float x_center = x + w / 2;
	float y_center = y + h / 2;

	float w_new = w * ntimes;
	float h_new = h * ntimes;
	float x_new = x_center - w_new / 2;
	float y_new = y_center - h_new / 2;

	return dlib::rectangle(x_new, y_new, x_new + w_new - 1, y_new + h_new - 1);
}



	int main(int argc, char* argv[])
	{
		std::vector<QString> dirs_search;
		dirs_search.push_back("D:/Research/Datasets/Face/Dlib");
		dirs_search.push_back("D:/Research/Datasets/Face/Recognition/lfw/lfw");

		QString dir_out = "D:/Research/Datasets/Face/Detection/Collection_80x80_pos_crops/Data";

		cv::Size size_save(80, 80);

		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

		QStringList extensions_search;
		extensions_search << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp" << "*.tif" << "*.tiff";

		unsigned int count_num_images = 0;

		for (size_t i = 0; i < dirs_search.size(); i++)
		{
			QDirIterator it(dirs_search[i], extensions_search, QDir::Files, QDirIterator::Subdirectories);

			while (it.hasNext())
			{
				QString s = it.next();
				QFileInfo f(s);
				std::string fpath_img = f.absoluteFilePath().toStdString();
				//printf("Processing image: %s\n", fpath_img.c_str());

				cv::Mat img_cv = cv::imread(fpath_img);
				int width_img = img_cv.cols;
				int height_img = img_cv.rows;
				dlib::cv_image<dlib::bgr_pixel> img_dlib(img_cv);
				std::vector<dlib::rectangle> dets = detector(img_dlib);

				if (dets.size() != 1) continue;

				for (size_t j = 0; j < dets.size(); j++)
				{
					count_num_images++;
					dlib::rectangle rect_dlib = dets[j];
					rect_dlib = enlarge_rect(rect_dlib, 0.3);
					int x = rect_dlib.top();
					int y = rect_dlib.left();
					int w = rect_dlib.width();
					int h = rect_dlib.height();
					if (x < 0 || y < 0 || x + w >= width_img || y + h >= height_img) continue;
					cv::Rect rect_cv(x, y, w, h);
					//printf("Bbox width = %d, height = %d\n", rect_dlib.width(), rect_dlib.height());
					cv::rectangle(img_cv, rect_cv, cv::Scalar(255, 0, 0), 3);
					cv::Mat roi(img_cv, rect_cv);
					cv::resize(roi, roi, size_save);
					QString img_save_name = QString("face_pos_crop_%1.png").arg(count_num_images, 5, 10, QChar('0'));
					std::string fpath_save = QDir::cleanPath(dir_out + QDir::separator() + img_save_name).toStdString();
					//cv::imwrite(fpath_save, roi);
				}

				cv::imshow("win", img_cv);
				cv::waitKey(0);

			}


		}





		return 0;
	}
}









